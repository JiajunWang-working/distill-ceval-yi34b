from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]

    @property
    def experiment(self) -> dict[str, Any]:
        return self.raw["experiment"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def lora(self) -> dict[str, Any]:
        return self.raw.get("lora", {})

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def distillation(self) -> dict[str, Any]:
        return self.raw["distillation"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def noise(self) -> dict[str, Any]:
        return self.raw.get("noise", {})


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = _expand_env_values(raw)
    return Config(raw=raw)


def _expand_env_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_values(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_expand_env_values(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value
