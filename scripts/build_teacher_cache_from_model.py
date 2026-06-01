#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tight_distill.data import canonicalize_target_text, extract_answer_label, format_prompt

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_teacher_target(row: dict, template: dict[str, str]) -> str:
    return canonicalize_target_text(
        raw_target=None,
        fallback_answer=row["answer"],
        fallback_rationale=row.get("explanation", ""),
        template=template,
        option_labels=list(row["options"].keys()),
    )


def make_quantization_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


@torch.no_grad()
def predict_option_probs_from_ids(model, input_ids: list[int], option_token_ids: list[int]) -> dict[int, float]:
    encoded = {
        "input_ids": torch.tensor([input_ids], device=model.device),
        "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long, device=model.device),
    }
    outputs = model(**encoded)
    next_token_logits = outputs.logits[:, -1, :].squeeze(0)
    option_logits = next_token_logits[option_token_ids]
    probs = torch.softmax(option_logits, dim=-1)
    return {token_id: float(prob.item()) for token_id, prob in zip(option_token_ids, probs)}


@torch.no_grad()
def generate_teacher_target(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    fallback_answer: str,
    option_labels: list[str],
    template: dict[str, str],
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
    outputs = model.generate(**encoded, **generation_kwargs)
    new_tokens = outputs[0, encoded["input_ids"].shape[1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return canonicalize_target_text(
        raw_target=text,
        fallback_answer=fallback_answer,
        fallback_rationale="",
        template=template,
        option_labels=option_labels,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build teacher cache from a local teacher model.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-model", type=str, required=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--system-prefix", type=str, default="你是一个严谨的中文选择题助手。")
    parser.add_argument("--instruction", type=str, default="请阅读题目并完成推理。")
    parser.add_argument("--response-prefix", type=str, default="请作答：")
    parser.add_argument(
        "--response-format-instruction",
        type=str,
        default='请先给出简明解析过程，最后一行使用“最终答案：<选项>”作答。',
    )
    parser.add_argument("--final-answer-prefix", "--answer-prefix", dest="final_answer_prefix", type=str, default="最终答案：")
    parser.add_argument("--rationale-prefix", type=str, default="解析：")
    parser.add_argument("--teacher-target-source", choices=["gold", "generate"], default="gold")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-target-length", type=int, default=256)
    args = parser.parse_args()

    rows = read_jsonl(args.dataset)
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    option_token_ids = []
    option_labels = ["A", "B", "C", "D", "E", "F"]
    for label in option_labels:
        label_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(label_ids) != 1:
            raise ValueError(f"Option label {label} is not a single token for this tokenizer.")
        option_token_ids.append(label_ids[0])

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        trust_remote_code=args.trust_remote_code,
        quantization_config=make_quantization_config(args.load_in_4bit),
        torch_dtype="auto",
        device_map="auto",
        local_files_only=args.local_files_only,
    )
    template = {
        "system_prefix": args.system_prefix,
        "instruction": args.instruction,
        "response_prefix": args.response_prefix,
        "response_format_instruction": args.response_format_instruction,
        "final_answer_prefix": args.final_answer_prefix,
        "rationale_prefix": args.rationale_prefix,
    }

    with args.output.open("w", encoding="utf-8") as f:
        for index, row in enumerate(rows, start=1):
            prompt = format_prompt(row=row, template=template, include_answer_stub=True)
            teacher_target = build_teacher_target(row, template)
            if args.teacher_target_source == "generate":
                teacher_target = generate_teacher_target(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    fallback_answer=row["answer"],
                    option_labels=list(row["options"].keys()),
                    template=template,
                )
            row_option_labels = list(row["options"].keys())
            teacher_answer = extract_answer_label(teacher_target, row_option_labels)
            if teacher_answer is None:
                raise ValueError(f"Could not extract final answer from teacher target for {row['id']}")

            target_prefix_text = teacher_target[: -len(teacher_answer)]
            prompt_ids = tokenizer.encode(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_prompt_length,
            )
            target_prefix_ids = tokenizer.encode(
                target_prefix_text,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_target_length,
            )
            probe_input_ids = prompt_ids + target_prefix_ids
            if not probe_input_ids:
                raise ValueError(f"Probe prompt is empty for {row['id']}")

            row_token_ids = []
            for label in row_option_labels:
                label_ids = tokenizer.encode(label, add_special_tokens=False)
                if len(label_ids) != 1:
                    raise ValueError(f"Option label {label} is not a single token for this tokenizer.")
                row_token_ids.append(label_ids[0])
            option_prob_by_id = predict_option_probs_from_ids(
                model=model,
                input_ids=probe_input_ids,
                option_token_ids=row_token_ids,
            )
            option_probs = {
                label: option_prob_by_id[token_id]
                for label, token_id in zip(row_option_labels, row_token_ids)
            }
            record = {
                "id": row["id"],
                "teacher_target": teacher_target,
                "option_probs": option_probs,
                "source": f"local-model:{args.teacher_model}",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if index % 10 == 0 or index == len(rows):
                print(f"Processed {index}/{len(rows)} examples")
    print(f"Wrote teacher cache to {args.output}")


if __name__ == "__main__":
    main()
