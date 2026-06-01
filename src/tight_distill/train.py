from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from tight_distill.config import load_config
from tight_distill.data import OPTION_LABELS, load_encoded_dataset
from tight_distill.distill_trainer import DistillationCollator, DistillationConfig, DistillationTrainerMixin
from tight_distill.noise import maybe_register_noise_hook
from tight_distill.utils import ensure_dir, save_yaml, set_seed, write_run_manifest


class DistillationTrainer(DistillationTrainerMixin, Trainer):
    pass


def parse_torch_dtype(value: str | None) -> Any:
    if value in (None, "auto"):
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return mapping[value]


def maybe_make_bnb_config(model_cfg: dict[str, Any]) -> BitsAndBytesConfig | None:
    if not model_cfg.get("load_in_4bit", False):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def maybe_apply_lora(model, lora_cfg: dict[str, Any]):
    if not lora_cfg.get("enabled", False):
        return model
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def build_dataset(encoded_rows) -> Dataset:
    rows = [
        {
            "input_ids": row.input_ids,
            "attention_mask": row.attention_mask,
            "labels": row.labels,
            "distill_index": row.distill_index,
            "teacher_option_probs": row.teacher_option_probs,
            "example_id": row.example_id,
        }
        for row in encoded_rows
    ]
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a reproducible MCQ distillation run.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.experiment["seed"]))

    output_dir = ensure_dir(cfg.experiment["output_dir"])
    save_yaml(output_dir / "resolved_config.yaml", cfg.raw)
    write_run_manifest(output_dir, cfg.raw)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model["student_model_name_or_path"],
        trust_remote_code=cfg.model.get("trust_remote_code", False),
        local_files_only=bool(cfg.model.get("local_files_only", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = maybe_make_bnb_config(cfg.model)
    torch_dtype = parse_torch_dtype(cfg.model.get("torch_dtype"))
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model["student_model_name_or_path"],
        trust_remote_code=cfg.model.get("trust_remote_code", False),
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        local_files_only=bool(cfg.model.get("local_files_only", False)),
    )
    if cfg.model.get("gradient_checkpointing", False):
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model.gradient_checkpointing_enable()
    model = maybe_apply_lora(model, cfg.lora)
    noise_hook_handle = maybe_register_noise_hook(model, cfg.noise)

    option_labels = cfg.distillation.get("option_labels", OPTION_LABELS)
    option_token_ids = []
    for label in option_labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Option label {label} is not a single token for this tokenizer.")
        option_token_ids.append(token_ids[0])

    train_rows = load_encoded_dataset(
        dataset_path=cfg.data["train_path"],
        teacher_cache_path=cfg.data["train_teacher_path"],
        tokenizer=tokenizer,
        template=cfg.data["template"],
        max_prompt_length=int(cfg.data["max_prompt_length"]),
        max_target_length=int(cfg.data["max_target_length"]),
        option_labels=option_labels,
    )
    val_rows = load_encoded_dataset(
        dataset_path=cfg.data["val_path"],
        teacher_cache_path=cfg.data["val_teacher_path"],
        tokenizer=tokenizer,
        template=cfg.data["template"],
        max_prompt_length=int(cfg.data["max_prompt_length"]),
        max_target_length=int(cfg.data["max_target_length"]),
        option_labels=option_labels,
    )

    train_dataset = build_dataset(train_rows)
    val_dataset = build_dataset(val_rows)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=int(cfg.training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg.training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg.training["gradient_accumulation_steps"]),
        learning_rate=float(cfg.training["learning_rate"]),
        num_train_epochs=float(cfg.training["num_train_epochs"]),
        weight_decay=float(cfg.training["weight_decay"]),
        warmup_ratio=float(cfg.training["warmup_ratio"]),
        logging_steps=int(cfg.training["logging_steps"]),
        save_steps=int(cfg.training["save_steps"]),
        eval_steps=int(cfg.training["eval_steps"]),
        save_total_limit=int(cfg.training["save_total_limit"]),
        max_grad_norm=float(cfg.training["max_grad_norm"]),
        report_to=cfg.training.get("report_to", []),
        bf16=bool(cfg.training.get("bf16", False)),
        fp16=bool(cfg.training.get("fp16", False)),
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        remove_unused_columns=False,
        load_best_model_at_end=False,
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DistillationCollator(pad_token_id=tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    trainer.distill_cfg = DistillationConfig(
        enabled=bool(cfg.distillation.get("enabled", True)),
        method=str(cfg.distillation.get("method", "kl")),
        alpha=float(cfg.distillation["alpha"]),
        temperature=float(cfg.distillation["temperature"]),
        option_token_ids=option_token_ids,
        bayes_eps=float(cfg.distillation.get("bayes_eps", 1e-6)),
        bayes_scale_min=float(cfg.distillation.get("bayes_scale_min", 1e-3)),
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    if noise_hook_handle is not None:
        noise_hook_handle.remove()


if __name__ == "__main__":
    main()
