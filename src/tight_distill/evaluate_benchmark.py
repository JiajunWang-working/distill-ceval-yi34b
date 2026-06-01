from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tight_distill.config import load_config
from tight_distill.data import get_prompt_mode, read_jsonl
from tight_distill.utils import ensure_dir, save_json


DEFAULT_CEVAL_SUBJECT_MAPPING = {
    "advanced_mathematics": ["Advanced Mathematics", "高等数学", "STEM"],
    "basic_medicine": ["Basic Medicine", "基础医学", "STEM"],
    "clinical_medicine": ["Clinical Medicine", "临床医学", "STEM"],
    "college_chemistry": ["College Chemistry", "大学化学", "STEM"],
    "college_physics": ["College Physics", "大学物理", "STEM"],
    "computer_architecture": ["Computer Architecture", "计算机组成", "STEM"],
}


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


def _format_options(options: dict[str, str]) -> str:
    labels = sorted(options.keys())
    return "\n".join(f"{label}. {options[label].strip()}" for label in labels)


@lru_cache(maxsize=8)
def _load_subject_mapping(mapping_path: str | None) -> dict[str, Any]:
    if not mapping_path:
        return {}
    path = Path(mapping_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_subject_metadata(row: dict[str, Any], template: dict[str, Any]) -> dict[str, str]:
    subject = str(row.get("subject", "")).strip()
    mapping_path = template.get("benchmark_subject_mapping_path")
    subject_mapping = _load_subject_mapping(str(mapping_path)) if mapping_path else {}
    subject_meta = subject_mapping.get(subject, DEFAULT_CEVAL_SUBJECT_MAPPING.get(subject))

    subject_name_en = subject
    subject_name_zh = subject
    subject_category = ""
    if isinstance(subject_meta, list) and len(subject_meta) >= 3:
        subject_name_en = str(subject_meta[0]).strip() or subject
        subject_name_zh = str(subject_meta[1]).strip() or subject
        subject_category = str(subject_meta[2]).strip()
    elif isinstance(subject_meta, dict):
        subject_name_en = str(subject_meta.get("en", subject)).strip() or subject
        subject_name_zh = str(subject_meta.get("zh", subject)).strip() or subject
        subject_category = str(subject_meta.get("category", "")).strip()

    return {
        "subject": subject,
        "subject_name_en": subject_name_en,
        "subject_name_zh": subject_name_zh,
        "subject_category": subject_category,
    }


def _build_format_context(
    row: dict[str, Any],
    template: dict[str, Any],
    answer: str = "",
) -> dict[str, str]:
    subject_meta = _get_subject_metadata(row, template)
    option_items = {
        str(label).strip(): str(text).strip()
        for label, text in row["options"].items()
    }
    return {
        "id": str(row.get("id", "")).strip(),
        "question": str(row["question"]).strip(),
        "options": _format_options(row["options"]),
        "answer": answer,
        "answer_prefix": _get_answer_prefix(template),
        **option_items,
        **subject_meta,
    }


def _format_template_text(text: str, row: dict[str, Any], template: dict[str, Any]) -> str:
    return text.format(**_build_format_context(row=row, template=template))


def _get_answer_prefix(template: dict[str, Any]) -> str:
    return template.get("benchmark_answer_prefix", template.get("answer_prefix", "答案："))


def _get_system_prefix(template: dict[str, Any]) -> str:
    return template.get(
        "benchmark_system_prefix",
        template.get("system_prefix", "你是一个严谨的中文单项选择题助手。"),
    )


def _get_instruction(template: dict[str, Any]) -> str:
    return template.get(
        "benchmark_instruction",
        "请直接给出最可能的正确选项字母，不要解释。",
    )


def _get_user_prefix(template: dict[str, Any]) -> str:
    return template.get(
        "benchmark_user_prefix",
        template.get("user_prefix", "下面是一道单项选择题。"),
    )


def _get_few_shot_seed(template: dict[str, Any]) -> int:
    return int(template.get("benchmark_few_shot_seed", 42))


def _get_subject_key(row: dict[str, Any]) -> str:
    return str(row.get("subject", "__unknown__"))


def render_benchmark_example(
    row: dict[str, Any],
    template: dict[str, Any],
    include_answer: bool,
    answer_override: str | None = None,
) -> str:
    answer_prefix = _get_answer_prefix(template)
    answer = answer_override or row["answer"]
    example_format = template.get(
        "benchmark_example_with_answer_format" if include_answer else "benchmark_example_without_answer_format"
    ) or template.get("benchmark_example_format")
    if example_format:
        return example_format.format(
            **_build_format_context(
                row=row,
                template=template,
                answer=answer if include_answer else "",
            )
        )
    example = (
        f"题目：{row['question'].strip()}\n"
        f"选项：\n{_format_options(row['options'])}\n"
        f"{answer_prefix}"
    )
    if include_answer:
        example += f"{answer}"
    return example


def build_few_shot_block(
    target_row: dict[str, Any],
    few_shot_rows: list[dict[str, Any]],
    num_shots: int,
    template: dict[str, Any],
) -> str:
    if num_shots <= 0:
        return ""
    same_subject_rows = [row for row in few_shot_rows if _get_subject_key(row) == _get_subject_key(target_row)]
    use_subject_ordered_shots = bool(template.get("benchmark_use_subject_ordered_few_shot", False))
    if use_subject_ordered_shots:
        candidate_rows = [row for row in same_subject_rows if row["id"] != target_row["id"]]
        sampled = candidate_rows[: min(num_shots, len(candidate_rows))]
    else:
        rng = random.Random(_get_few_shot_seed(template) + hash(target_row["id"]) % (10**6))
        candidate_rows = same_subject_rows if len(same_subject_rows) >= num_shots else few_shot_rows
        candidate_rows = [row for row in candidate_rows if row["id"] != target_row["id"]]
        if not candidate_rows:
            return ""
        sampled = candidate_rows[:]
        rng.shuffle(sampled)
        sampled = sampled[: min(num_shots, len(sampled))]
    if not sampled:
        return ""
    return "\n\n".join(
        render_benchmark_example(row, template=template, include_answer=True) for row in sampled
    )


def build_plain_prompt(
    row: dict[str, Any],
    template: dict[str, Any],
    few_shot_rows: list[dict[str, Any]],
    num_shots: int,
) -> str:
    parts = [
        _format_template_text(_get_system_prefix(template).strip(), row=row, template=template),
        _format_template_text(_get_instruction(template).strip(), row=row, template=template),
    ]
    few_shot_block = build_few_shot_block(row, few_shot_rows=few_shot_rows, num_shots=num_shots, template=template)
    if few_shot_block:
        parts.append(few_shot_block)
    parts.append(render_benchmark_example(row, template=template, include_answer=False))
    return "\n\n".join(part for part in parts if part)


def build_chat_messages(
    row: dict[str, Any],
    template: dict[str, Any],
    few_shot_rows: list[dict[str, Any]],
    num_shots: int,
) -> list[dict[str, str]]:
    system_prefix = _format_template_text(_get_system_prefix(template).strip(), row=row, template=template)
    user_prefix = _format_template_text(_get_user_prefix(template).strip(), row=row, template=template)
    instruction = _format_template_text(_get_instruction(template).strip(), row=row, template=template)
    few_shot_block = build_few_shot_block(row, few_shot_rows=few_shot_rows, num_shots=num_shots, template=template)

    user_parts = [user_prefix, instruction]
    if few_shot_block:
        user_parts.append("以下是示例：\n" + few_shot_block)
    user_parts.append(render_benchmark_example(row, template=template, include_answer=False))
    messages: list[dict[str, str]] = []
    if system_prefix:
        messages.append({"role": "system", "content": system_prefix})
    messages.append({"role": "user", "content": "\n\n".join(part for part in user_parts if part)})
    return messages


def render_benchmark_prompt(
    tokenizer,
    row: dict[str, Any],
    template: dict[str, Any],
    few_shot_rows: list[dict[str, Any]],
    num_shots: int,
) -> str:
    prompt_mode = get_prompt_mode(template)
    if prompt_mode == "chat_template":
        messages = build_chat_messages(
            row=row,
            template=template,
            few_shot_rows=few_shot_rows,
            num_shots=num_shots,
        )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if prompt_mode != "plain":
        raise ValueError(f"Unsupported prompt_mode for benchmark evaluation: {prompt_mode}")
    return build_plain_prompt(row=row, template=template, few_shot_rows=few_shot_rows, num_shots=num_shots)


def group_rows_by_subject(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_get_subject_key(row)].append(row)
    return grouped


def get_single_token_option_ids(tokenizer, option_labels: list[str]) -> list[int] | None:
    option_token_ids: list[int] = []
    for label in option_labels:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            return None
        option_token_ids.append(token_ids[0])
    return option_token_ids


@torch.no_grad()
def score_option_candidates(
    model,
    tokenizer,
    prompt: str,
    option_labels: list[str],
) -> tuple[dict[str, float], str]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    option_token_ids = get_single_token_option_ids(tokenizer, option_labels)
    if option_token_ids is not None:
        encoded_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded_prompt = {key: value.to(model.device) for key, value in encoded_prompt.items()}
        outputs = model(**encoded_prompt)
        next_token_logits = outputs.logits[:, -1, :].squeeze(0)
        candidate_logits = next_token_logits[option_token_ids]
        candidate_probs = torch.softmax(candidate_logits, dim=-1)
        probs = {
            label: float(prob.item())
            for label, prob in zip(option_labels, candidate_probs)
        }
        prediction = option_labels[int(torch.argmax(candidate_logits).item())]
        return probs, prediction

    encoded_prompt = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_len = int(encoded_prompt["input_ids"].shape[1])
    encoded_prompt = {key: value.to(model.device) for key, value in encoded_prompt.items()}

    scores: dict[str, float] = {}
    for label in option_labels:
        continuation = label
        encoded_full = tokenizer(prompt + continuation, return_tensors="pt", add_special_tokens=False)
        full_ids = encoded_full["input_ids"].to(model.device)
        attention_mask = encoded_full["attention_mask"].to(model.device)
        outputs = model(input_ids=full_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        target_ids = full_ids[:, 1:]

        option_token_start = max(prompt_len - 1, 0)
        option_logits = logits[:, option_token_start:, :]
        option_targets = target_ids[:, option_token_start:]
        log_probs = torch.log_softmax(option_logits, dim=-1)
        gathered = torch.gather(log_probs, dim=-1, index=option_targets.unsqueeze(-1)).squeeze(-1)
        scores[label] = float(gathered.sum().item())

    prediction = max(scores.items(), key=lambda item: item[1])[0]
    max_logit = max(scores.values())
    probs = {label: math.exp(score - max_logit) for label, score in scores.items()}
    total = sum(probs.values())
    if total > 0:
        probs = {label: value / total for label, value in probs.items()}
    return probs, prediction


@torch.no_grad()
def score_option_candidates_batch(
    model,
    tokenizer,
    prompts: list[str],
    option_labels: list[str],
) -> list[tuple[dict[str, float], str]]:
    option_token_ids = get_single_token_option_ids(tokenizer, option_labels)
    if option_token_ids is None:
        return [
            score_option_candidates(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                option_labels=option_labels,
            )
            for prompt in prompts
        ]

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    outputs = model(**encoded)
    attention_mask = encoded["attention_mask"]
    # `padding_side` may be left for decoder-only chat/instruct models.
    # In that case `attention_mask.sum() - 1` points to the end of the
    # unpadded sequence length, not the last valid token position in the
    # padded batch tensor. We need the index of the rightmost non-pad token.
    last_positions = attention_mask.shape[1] - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)
    batch_indices = torch.arange(attention_mask.shape[0], device=model.device)
    next_token_logits = outputs.logits[batch_indices, last_positions, :]
    candidate_logits = next_token_logits[:, option_token_ids]
    candidate_probs = torch.softmax(candidate_logits, dim=-1)
    predictions = torch.argmax(candidate_logits, dim=-1)

    results: list[tuple[dict[str, float], str]] = []
    for row_probs, row_pred_idx in zip(candidate_probs, predictions):
        probs = {
            label: float(prob.item())
            for label, prob in zip(option_labels, row_probs)
        }
        pred = option_labels[int(row_pred_idx.item())]
        results.append((probs, pred))
    return results


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark-style MCQ evaluation with option likelihood and few-shot support.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--few-shot-dataset", type=Path, default=None)
    parser.add_argument("--num-shots", type=int, default=0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--quant-compute-dtype", type=str, default="bf16")
    parser.add_argument("--output-prefix", type=str, default="benchmark_eval")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_path = args.dataset or Path(cfg.data["val_path"])
    few_shot_path = args.few_shot_dataset or Path(cfg.data["train_path"])
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
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(dataset_path)
    few_shot_rows = read_jsonl(few_shot_path)
    predictions = []
    correct = 0
    batch_size = int(cfg.evaluation.get("batch_size", 1))

    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        batch_option_labels = [sorted(row["options"].keys()) for row in batch_rows]
        if not all(labels == batch_option_labels[0] for labels in batch_option_labels):
            for row, option_labels in zip(batch_rows, batch_option_labels):
                prompt = render_benchmark_prompt(
                    tokenizer=tokenizer,
                    row=row,
                    template=cfg.data["template"],
                    few_shot_rows=few_shot_rows,
                    num_shots=args.num_shots,
                )
                option_probs, pred = score_option_candidates(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    option_labels=option_labels,
                )
                is_correct = pred == row["answer"]
                correct += int(is_correct)
                predictions.append(
                    {
                        "id": row["id"],
                        "subject": row.get("subject"),
                        "prediction": pred,
                        "gold": row["answer"],
                        "correct": is_correct,
                        "option_probs": option_probs,
                        "prompt": prompt,
                    }
                )
            continue

        prompts = [
            render_benchmark_prompt(
                tokenizer=tokenizer,
                row=row,
                template=cfg.data["template"],
                few_shot_rows=few_shot_rows,
                num_shots=args.num_shots,
            )
            for row in batch_rows
        ]
        scored = score_option_candidates_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            option_labels=batch_option_labels[0],
        )
        for row, prompt, (option_probs, pred) in zip(batch_rows, prompts, scored):
            is_correct = pred == row["answer"]
            correct += int(is_correct)
            predictions.append(
                {
                    "id": row["id"],
                    "subject": row.get("subject"),
                    "prediction": pred,
                    "gold": row["answer"],
                    "correct": is_correct,
                    "option_probs": option_probs,
                    "prompt": prompt,
                }
            )

    accuracy = correct / max(len(rows), 1)
    output_dir = ensure_dir(cfg.experiment["output_dir"])
    metrics = {
        "accuracy": accuracy,
        "num_examples": len(rows),
        "num_shots": args.num_shots,
        "evaluation_mode": "option_likelihood_benchmark",
        "dataset_path": str(dataset_path),
        "few_shot_dataset_path": str(few_shot_path),
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
