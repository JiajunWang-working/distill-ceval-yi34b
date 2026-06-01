from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OPTION_LABELS = ["A", "B", "C", "D"]


@dataclass
class EncodedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    distill_index: int
    teacher_option_probs: list[float]
    example_id: str


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_teacher_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {row["id"]: row for row in rows}


def normalize_option_probs(option_probs: dict[str, float], option_labels: list[str]) -> list[float]:
    probs = [float(option_probs.get(label, 0.0)) for label in option_labels]
    total = sum(probs)
    if total <= 0:
        return [1.0 / len(option_labels)] * len(option_labels)
    return [value / total for value in probs]


def get_rationale_prefix(template: dict[str, str]) -> str:
    return template.get("rationale_prefix", "解析：")


def get_final_answer_prefix(template: dict[str, str]) -> str:
    return template.get("final_answer_prefix", template.get("answer_prefix", "最终答案："))


def get_response_prefix(template: dict[str, str]) -> str:
    return template.get("response_prefix", "请作答：")


def get_response_format_instruction(template: dict[str, str]) -> str:
    return template.get(
        "response_format_instruction",
        '请先给出简明解析过程，最后一行使用“最终答案：<选项>”作答。',
    )


def get_prompt_mode(template: dict[str, str]) -> str:
    return template.get("prompt_mode", "plain")


def get_user_prefix(template: dict[str, str]) -> str:
    return template.get("user_prefix", "")


def get_assistant_prefix(template: dict[str, str]) -> str:
    return template.get("assistant_prefix", get_response_prefix(template))


def extract_answer_label(text: str, option_labels: list[str]) -> str | None:
    allowed = {label.upper() for label in option_labels}
    upper_text = text.upper()
    patterns = [
        r"(?:最终答案|答案|故选|选择|选)\s*[:：]?\s*[\(\[（]?\s*([A-F])\s*[\)\]）]?",
        r"(?:应选|应当选|应选择|故应选|故应选择|正确选项是|正确答案是)\s*[:：]?\s*[\(\[（]?\s*([A-F])\s*[\)\]）]?",
        r"^\s*[\(\[（]?\s*([A-F])\s*[\)\]）]?\s*[。．.]?\s*$",
    ]

    matches: list[tuple[int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, upper_text, flags=re.IGNORECASE | re.MULTILINE):
            label = match.group(1).upper()
            if label in allowed:
                matches.append((match.start(), label))
    if matches:
        matches.sort(key=lambda item: item[0])
        return matches[-1][1]
    return None


def _is_answer_only_line(line: str, option_labels: list[str]) -> bool:
    return extract_answer_label(line, option_labels) is not None and (
        re.fullmatch(
            r"\s*(?:最终答案|答案|故选|选择|选)?\s*[:：]?\s*[\(\[（]?\s*[A-F]\s*[\)\]）]?\s*[。．.]?\s*",
            line,
            flags=re.IGNORECASE,
        )
        is not None
    )


def _clean_rationale_text(text: str, option_labels: list[str], template: dict[str, str]) -> str:
    rationale_prefix = get_rationale_prefix(template)
    lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n") if line.strip()]
    kept_lines: list[str] = []
    for line in lines:
        if _is_answer_only_line(line, option_labels):
            continue
        kept_lines.append(line)

    rationale = "\n".join(kept_lines).strip()
    if rationale.startswith(rationale_prefix):
        rationale = rationale[len(rationale_prefix) :].lstrip()
    else:
        rationale = re.sub(r"^解析\s*[:：]\s*", "", rationale, count=1)

    final_answer_label = extract_answer_label(text, option_labels)
    if final_answer_label is not None:
        rationale = re.sub(
            rf"(?:最终答案|答案|故选|选择|选)\s*[:：]?\s*[\(\[（]?\s*{re.escape(final_answer_label)}\s*[\)\]）]?\s*[。．.]?\s*$",
            "",
            rationale,
            flags=re.IGNORECASE,
        ).strip()
    return rationale


def compose_target(answer: str, rationale: str, template: dict[str, str]) -> tuple[str, str]:
    rationale = rationale.strip()
    final_answer_prefix = get_final_answer_prefix(template)
    if rationale:
        prefix_text = f"{get_rationale_prefix(template)}{rationale}\n{final_answer_prefix}"
    else:
        prefix_text = final_answer_prefix
    return f"{prefix_text}{answer}", prefix_text


def canonicalize_target_text(
    raw_target: str | None,
    fallback_answer: str,
    fallback_rationale: str,
    template: dict[str, str],
    option_labels: list[str],
) -> str:
    if raw_target is None:
        raw_target = ""
    raw_target = str(raw_target).strip()
    answer = extract_answer_label(raw_target, option_labels) or fallback_answer
    rationale = _clean_rationale_text(raw_target, option_labels, template)
    if not rationale:
        rationale = fallback_rationale.strip()
    target_text, _ = compose_target(answer=answer, rationale=rationale, template=template)
    return target_text


def format_prompt(row: dict[str, Any], template: dict[str, str], include_answer_stub: bool = True) -> str:
    option_labels = list(row["options"].keys())
    options = "\n".join(f"{label}. {row['options'][label]}" for label in option_labels)
    parts = [
        get_user_prefix(template).strip(),
        template.get("system_prefix", "").strip(),
        template.get("instruction", "").strip(),
        get_response_format_instruction(template).strip(),
        f"题目：{row['question'].strip()}",
        f"选项：\n{options}",
    ]
    prompt = "\n\n".join(part for part in parts if part)
    if include_answer_stub:
        prompt += f"\n\n{get_assistant_prefix(template)}"
    return prompt


def build_chat_messages(row: dict[str, Any], template: dict[str, str], include_answer_stub: bool = True) -> list[dict[str, str]]:
    user_prompt = format_prompt(row, {**template, "system_prefix": ""}, include_answer_stub=False)
    messages: list[dict[str, str]] = []
    system_prefix = template.get("system_prefix", "").strip()
    if system_prefix:
        messages.append({"role": "system", "content": system_prefix})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def format_target(
    row: dict[str, Any],
    teacher_row: dict[str, Any] | None,
    option_labels: list[str],
    template: dict[str, str],
) -> tuple[str, str]:
    if teacher_row and teacher_row.get("teacher_target"):
        target_text = canonicalize_target_text(
            raw_target=teacher_row.get("teacher_target"),
            fallback_answer=row["answer"],
            fallback_rationale=row.get("explanation", ""),
            template=template,
            option_labels=option_labels,
        )
    else:
        target_text = canonicalize_target_text(
            raw_target=None,
            fallback_answer=row["answer"],
            fallback_rationale=row.get("explanation", ""),
            template=template,
            option_labels=option_labels,
        )

    distill_answer = extract_answer_label(target_text, option_labels)
    if distill_answer is None:
        raise ValueError(f"Could not extract final answer from target for example {row['id']}")
    return target_text, distill_answer


def encode_supervised_example(
    row: dict[str, Any],
    teacher_row: dict[str, Any] | None,
    tokenizer: Any,
    template: dict[str, str],
    max_prompt_length: int,
    max_target_length: int,
    option_labels: list[str],
) -> EncodedExample:
    prompt_text = format_prompt(row, template, include_answer_stub=True)
    target_text, distill_answer = format_target(row, teacher_row, option_labels, template)
    target_prefix_text = target_text[: -len(distill_answer)]

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False, truncation=True, max_length=max_prompt_length)
    answer_ids = tokenizer.encode(distill_answer, add_special_tokens=False)
    if not answer_ids:
        raise ValueError(f"Final answer tokenization is empty for example {row['id']}")
    if len(answer_ids) > max_target_length:
        raise ValueError(f"Final answer is longer than max_target_length for example {row['id']}")

    target_prefix_ids_full = tokenizer.encode(
        target_prefix_text,
        add_special_tokens=False,
    )
    max_prefix_length = max(max_target_length - len(answer_ids), 0)
    if len(target_prefix_ids_full) > max_prefix_length:
        target_prefix_ids = target_prefix_ids_full[-max_prefix_length:] if max_prefix_length > 0 else []
    else:
        target_prefix_ids = target_prefix_ids_full
    target_ids = target_prefix_ids + answer_ids

    if not target_ids:
        raise ValueError(f"Target is empty for example {row['id']}")
    if len(target_prefix_ids) >= len(target_ids):
        raise ValueError(f"Could not isolate final-answer token in target for example {row['id']}")

    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        target_ids = target_ids + [eos_id]

    input_ids = prompt_ids + target_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + target_ids
    distill_index = max(len(prompt_ids) + len(target_prefix_ids) - 1, 0)

    teacher_option_probs = normalize_option_probs(
        (teacher_row or {}).get("option_probs", {}),
        option_labels,
    )

    return EncodedExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        distill_index=distill_index,
        teacher_option_probs=teacher_option_probs,
        example_id=row["id"],
    )


def load_encoded_dataset(
    dataset_path: str | Path,
    teacher_cache_path: str | Path,
    tokenizer: Any,
    template: dict[str, str],
    max_prompt_length: int,
    max_target_length: int,
    option_labels: list[str],
) -> list[EncodedExample]:
    rows = read_jsonl(dataset_path)
    teacher_cache = load_teacher_cache(teacher_cache_path)
    encoded = []
    for row in rows:
        teacher_row = teacher_cache.get(row["id"])
        encoded.append(
            encode_supervised_example(
                row=row,
                teacher_row=teacher_row,
                tokenizer=tokenizer,
                template=template,
                max_prompt_length=max_prompt_length,
                max_target_length=max_target_length,
                option_labels=option_labels,
            )
        )
    return encoded
