from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class DistillationConfig:
    enabled: bool
    method: str
    alpha: float
    temperature: float
    option_token_ids: list[int]
    bayes_eps: float = 1e-6
    bayes_scale_min: float = 1e-3


class DistillationCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        distill_index = []
        teacher_option_probs = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)
            distill_index.append(feature["distill_index"])
            teacher_option_probs.append(feature["teacher_option_probs"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "distill_index": torch.tensor(distill_index, dtype=torch.long),
            "teacher_option_probs": torch.tensor(teacher_option_probs, dtype=torch.float),
        }


class DistillationTrainerMixin:
    distill_cfg: DistillationConfig

    def _compute_kl_kd_loss(self, option_logits: torch.Tensor, teacher_option_probs: torch.Tensor) -> torch.Tensor:
        temperature = self.distill_cfg.temperature
        log_probs = F.log_softmax(option_logits / temperature, dim=-1)
        target_probs = teacher_option_probs.to(option_logits.device)
        return F.kl_div(log_probs, target_probs, reduction="batchmean")

    def _compute_bayes_kd_loss(self, option_logits: torch.Tensor, teacher_option_probs: torch.Tensor) -> torch.Tensor:
        target_probs = teacher_option_probs.to(option_logits.device).clamp_min(self.distill_cfg.bayes_eps)
        target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)

        teacher_logits = torch.log(target_probs)
        teacher_scale = torch.std(teacher_logits, dim=-1, keepdim=True, unbiased=False).clamp_min(
            self.distill_cfg.bayes_scale_min
        )
        student_scale = torch.std(option_logits, dim=-1, keepdim=True, unbiased=False).clamp_min(
            self.distill_cfg.bayes_scale_min
        )

        temperature = self.distill_cfg.temperature
        scaled_teacher_probs = torch.softmax(teacher_logits / teacher_scale, dim=-1)
        scaled_student_log_probs = F.log_softmax((option_logits / temperature) / student_scale, dim=-1)
        return F.kl_div(scaled_student_log_probs, scaled_teacher_probs, reduction="batchmean")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # type: ignore[override]
        labels = inputs["labels"]
        distill_index = inputs.pop("distill_index")
        teacher_option_probs = inputs.pop("teacher_option_probs")
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        sft_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        total_loss = sft_loss
        kd_loss = torch.zeros((), device=logits.device)
        if self.distill_cfg.enabled:
            batch_indices = torch.arange(logits.size(0), device=logits.device)
            selected_logits = logits[batch_indices, distill_index, :]
            option_logits = selected_logits[:, self.distill_cfg.option_token_ids]
            if self.distill_cfg.method == "kl":
                kd_loss = self._compute_kl_kd_loss(option_logits, teacher_option_probs)
            elif self.distill_cfg.method == "bayes":
                kd_loss = self._compute_bayes_kd_loss(option_logits, teacher_option_probs)
            else:
                raise ValueError(f"Unsupported distillation method: {self.distill_cfg.method}")
            temperature = self.distill_cfg.temperature
            total_loss = (1.0 - self.distill_cfg.alpha) * sft_loss + self.distill_cfg.alpha * (temperature ** 2) * kd_loss

        outputs.loss = total_loss
        outputs.sft_loss = sft_loss.detach()
        outputs.kd_loss = kd_loss.detach()
        return (total_loss, outputs) if return_outputs else total_loss
