from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tight_distill.config import load_config
from tight_distill.data import (
    OPTION_LABELS,
    build_chat_messages,
    extract_answer_label,
    format_prompt,
    get_prompt_mode,
    read_jsonl,
)
from tight_distill.utils import ensure_dir, save_json


def parse_torch_dtype(value: str | None):
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


def maybe_make_bnb_config(load_in_4bit: bool, load_in_8bit: bool, compute_dtype: str | None):
    if load_in_4bit and load_in_8bit:
        raise ValueError("Only one of --load-in-4bit or --load-in-8bit may be set.")
    if not load_in_4bit and not load_in_8bit:
        return None
    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=parse_torch_dtype(compute_dtype),
    )


def load_eval_model(
    checkpoint: str,
    trust_remote_code: bool,
    quantization_config: BitsAndBytesConfig | None,
    local_files_only: bool,
):
    checkpoint_path = Path(checkpoint)
    adapter_config_path = checkpoint_path / "adapter_config.json"
    common_kwargs = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": "auto",
        "device_map": "auto",
        "local_files_only": local_files_only,
    }
    if quantization_config is not None:
        common_kwargs["quantization_config"] = quantization_config

    if adapter_config_path.exists():
        from peft import AutoPeftModelForCausalLM

        return AutoPeftModelForCausalLM.from_pretrained(
            checkpoint,
            **common_kwargs,
        )
    return AutoModelForCausalLM.from_pretrained(
        checkpoint,
        **common_kwargs,
    )


def render_prompt(tokenizer, row: dict, template: dict[str, str], include_answer_stub: bool = True) -> str:
    prompt_mode = get_prompt_mode(template)
    if prompt_mode == "chat_template":
        messages = build_chat_messages(row, template, include_answer_stub=include_answer_stub)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=include_answer_stub,
        )
    if prompt_mode != "plain":
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")
    return format_prompt(row, template, include_answer_stub=include_answer_stub)


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_prompt_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> list[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
    outputs = model.generate(**encoded, **generation_kwargs)

    generated_texts: list[str] = []
    # For decoder-only models with left padding, generate() returns sequences
    # that include the full padded prompt width before newly generated tokens.
    prompt_width = int(encoded["input_ids"].shape[1])
    for row_idx in range(outputs.shape[0]):
        new_tokens = outputs[row_idx, prompt_width:]
        generated_texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return generated_texts


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an MCQ checkpoint by generation and answer extraction.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument(
        "--quant-compute-dtype",
        type=str,
        default="bf16",
        help="Compute dtype for 4-bit quantized evaluation.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="eval",
        help="Prefix for saved metrics/prediction files, e.g. 'val' or 'test'.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_path = args.dataset or Path(cfg.data["val_path"])
    default_option_labels = cfg.distillation.get("option_labels", OPTION_LABELS)
    quantization_config = maybe_make_bnb_config(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        compute_dtype=args.quant_compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        trust_remote_code=cfg.model.get("trust_remote_code", False),
        local_files_only=bool(cfg.model.get("local_files_only", False)),
    )
    model = load_eval_model(
        checkpoint=args.checkpoint,
        trust_remote_code=cfg.model.get("trust_remote_code", False),
        quantization_config=quantization_config,
        local_files_only=bool(cfg.model.get("local_files_only", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    rows = read_jsonl(dataset_path)
    predictions = []
    correct = 0
    parsed = 0
    batch_size = int(cfg.evaluation.get("batch_size", 1))
    max_prompt_length = int(cfg.data.get("max_prompt_length", 1024))
    max_new_tokens = int(cfg.evaluation.get("max_new_tokens", cfg.data.get("max_target_length", 256)))
    do_sample = bool(cfg.evaluation.get("do_sample", False))
    temperature = float(cfg.evaluation.get("temperature", 0.7))

    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        prompts = [render_prompt(tokenizer, row, cfg.data["template"], include_answer_stub=True) for row in batch_rows]
        generated_texts = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        for row, generated_text in zip(batch_rows, generated_texts):
            row_option_labels = [label for label in default_option_labels if label in row["options"]]
            pred = extract_answer_label(generated_text, row_option_labels)
            is_correct = pred == row["answer"]
            correct += int(is_correct)
            parsed += int(pred is not None)
            predictions.append(
                {
                    "id": row["id"],
                    "prediction": pred,
                    "gold": row["answer"],
                    "correct": is_correct,
                    "generated_text": generated_text,
                    "answer_found": pred is not None,
                }
            )

    accuracy = correct / max(len(rows), 1)
    output_dir = ensure_dir(cfg.experiment["output_dir"])
    metrics = {
        "accuracy": accuracy,
        "num_examples": len(rows),
        "parsed_answers": parsed,
        "parse_rate": parsed / max(len(rows), 1),
        "evaluation_mode": "generate_and_extract",
    }
    metrics_path = output_dir / f"{args.output_prefix}_metrics.json"
    predictions_path = output_dir / f"{args.output_prefix}_predictions.jsonl"
    save_json(metrics_path, metrics)
    if cfg.evaluation.get("save_predictions", True):
        with predictions_path.open("w", encoding="utf-8") as f:
            for prediction in predictions:
                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
