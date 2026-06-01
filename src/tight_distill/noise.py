from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class NoiseConfig:
    enabled: bool
    target_module: str
    distribution: str
    scale: float


def resolve_module_by_suffix(model, target_module: str):
    exact_match = None
    suffix_matches = []
    for name, module in model.named_modules():
        if name == target_module:
            exact_match = (name, module)
            break
        if name.endswith(target_module):
            suffix_matches.append((name, module))
    if exact_match is not None:
        return exact_match
    if not suffix_matches:
        raise ValueError(f"Could not find noise target module matching '{target_module}'.")
    if len(suffix_matches) > 1:
        match_names = ", ".join(name for name, _ in suffix_matches[:5])
        raise ValueError(
            f"Noise target '{target_module}' is ambiguous. Matching modules include: {match_names}"
        )
    return suffix_matches[0]


def _sample_noise(hidden_states: torch.Tensor, distribution: str, scale: float) -> torch.Tensor:
    if distribution == "gaussian":
        return torch.randn_like(hidden_states) * scale
    if distribution == "uniform":
        return torch.empty_like(hidden_states).uniform_(-scale, scale)
    raise ValueError(f"Unsupported noise distribution: {distribution}")


def maybe_register_noise_hook(model, noise_cfg_dict: dict[str, Any]):
    if not noise_cfg_dict or not noise_cfg_dict.get("enabled", False):
        return None

    noise_cfg = NoiseConfig(
        enabled=True,
        target_module=str(noise_cfg_dict["target_module"]),
        distribution=str(noise_cfg_dict.get("distribution", "gaussian")),
        scale=float(noise_cfg_dict["scale"]),
    )
    matched_name, target_module = resolve_module_by_suffix(model, noise_cfg.target_module)

    def forward_hook(_module, _inputs, output):
        if not getattr(_module, "training", False):
            return output

        if isinstance(output, tuple):
            hidden_states = output[0]
            noisy_hidden_states = hidden_states + _sample_noise(hidden_states, noise_cfg.distribution, noise_cfg.scale)
            return (noisy_hidden_states, *output[1:])
        if isinstance(output, torch.Tensor):
            return output + _sample_noise(output, noise_cfg.distribution, noise_cfg.scale)
        return output

    handle = target_module.register_forward_hook(forward_hook)
    print(
        f"Registered noise injection on module '{matched_name}' "
        f"with distribution={noise_cfg.distribution} scale={noise_cfg.scale}"
    )
    return handle
