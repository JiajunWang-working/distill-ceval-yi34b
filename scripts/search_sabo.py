#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_command(cmd: list[str], env: dict[str, str], cwd: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def fit_rbf_gp(xs: np.ndarray, ys: np.ndarray, length_scale: float = 0.35, noise: float = 1e-6):
    if len(xs) == 0:
        raise ValueError("Cannot fit GP with zero observations.")
    diffs = xs[:, None, :] - xs[None, :, :]
    sqdist = np.sum(diffs * diffs, axis=-1)
    kernel = np.exp(-0.5 * sqdist / (length_scale ** 2))
    kernel = kernel + noise * np.eye(len(xs))
    centered = ys - ys.mean()
    alpha = np.linalg.solve(kernel, centered)
    kernel_inv = np.linalg.inv(kernel)
    return ys.mean(), xs, alpha, kernel_inv, length_scale


def predict_rbf_gp(model, x_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_y, x_train, alpha, kernel_inv, length_scale = model
    diffs = x_query[:, None, :] - x_train[None, :, :]
    sqdist = np.sum(diffs * diffs, axis=-1)
    k_star = np.exp(-0.5 * sqdist / (length_scale ** 2))
    mean = mean_y + k_star @ alpha

    k_self = np.ones(len(x_query))
    var = k_self - np.einsum("ij,jk,ik->i", k_star, kernel_inv, k_star)
    var = np.clip(var, 1e-8, None)
    return mean, np.sqrt(var)


def sample_log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    if low <= 0 or high <= 0:
        raise ValueError("Log-uniform sampling requires positive bounds.")
    if math.isclose(low, high):
        return float(low)
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def normalize_linear(value: float, low: float, high: float) -> float:
    if math.isclose(low, high):
        return 0.5
    return float((value - low) / (high - low))


def normalize_log(value: float, low: float, high: float) -> float:
    if low <= 0 or high <= 0:
        raise ValueError("Log normalization requires positive bounds.")
    if math.isclose(low, high):
        return 0.5
    return float((np.log(value) - np.log(low)) / (np.log(high) - np.log(low)))


def parse_epoch_choices(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--epoch-choices was provided but no valid values were found.")
    return values


def sample_candidate(
    rng: np.random.Generator,
    alpha_range: tuple[float, float],
    temp_range: tuple[float, float],
    noise_scale_range: tuple[float, float],
    learning_rate_range: tuple[float, float],
    epoch_range: tuple[float, float],
    epoch_choices: list[float] | None,
) -> dict:
    epoch_budget = (
        float(rng.choice(epoch_choices))
        if epoch_choices is not None
        else float(rng.uniform(epoch_range[0], epoch_range[1]))
    )
    return {
        "alpha": float(rng.uniform(alpha_range[0], alpha_range[1])),
        "temperature": float(rng.uniform(temp_range[0], temp_range[1])),
        "noise_scale": float(rng.uniform(noise_scale_range[0], noise_scale_range[1])),
        "learning_rate": sample_log_uniform(rng, learning_rate_range[0], learning_rate_range[1]),
        "num_train_epochs": epoch_budget,
    }


def normalize_candidate(
    candidate: dict,
    alpha_range: tuple[float, float],
    temp_range: tuple[float, float],
    noise_scale_range: tuple[float, float],
    learning_rate_range: tuple[float, float],
    epoch_range: tuple[float, float],
) -> np.ndarray:
    alpha = normalize_linear(candidate["alpha"], alpha_range[0], alpha_range[1])
    temp = normalize_linear(candidate["temperature"], temp_range[0], temp_range[1])
    noise_scale = normalize_linear(candidate["noise_scale"], noise_scale_range[0], noise_scale_range[1])
    learning_rate = normalize_log(candidate["learning_rate"], learning_rate_range[0], learning_rate_range[1])
    epochs = normalize_linear(candidate["num_train_epochs"], epoch_range[0], epoch_range[1])
    return np.array([alpha, temp, noise_scale, learning_rate, epochs], dtype=np.float64)


def propose_candidate(
    rng: np.random.Generator,
    observations: list[dict],
    alpha_range: tuple[float, float],
    temp_range: tuple[float, float],
    noise_scale_range: tuple[float, float],
    learning_rate_range: tuple[float, float],
    epoch_range: tuple[float, float],
    epoch_choices: list[float] | None,
    warmup_trials: int,
    acquisition_beta: float,
    candidate_pool_size: int,
) -> dict:
    if len(observations) < warmup_trials:
        return sample_candidate(
            rng,
            alpha_range,
            temp_range,
            noise_scale_range,
            learning_rate_range,
            epoch_range,
            epoch_choices,
        )

    xs = np.stack(
        [
            normalize_candidate(
                obs,
                alpha_range,
                temp_range,
                noise_scale_range,
                learning_rate_range,
                epoch_range,
            )
            for obs in observations
        ],
        axis=0,
    )
    ys = np.array([obs["val_accuracy"] for obs in observations], dtype=np.float64)
    gp_model = fit_rbf_gp(xs, ys)

    pool = [
        sample_candidate(
            rng,
            alpha_range,
            temp_range,
            noise_scale_range,
            learning_rate_range,
            epoch_range,
            epoch_choices,
        )
        for _ in range(candidate_pool_size)
    ]
    pool_x = np.stack(
        [
            normalize_candidate(
                candidate,
                alpha_range,
                temp_range,
                noise_scale_range,
                learning_rate_range,
                epoch_range,
            )
            for candidate in pool
        ],
        axis=0,
    )
    mean, std = predict_rbf_gp(gp_model, pool_x)
    acquisition = mean + acquisition_beta * std
    best_idx = int(np.argmax(acquisition))
    return pool[best_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small likelihood-based SABO search over distillation hyperparameters.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com")
    parser.add_argument("--num-trials", type=int, default=5)
    parser.add_argument("--warmup-trials", type=int, default=2)
    parser.add_argument("--search-epochs", type=float, default=1.0)
    parser.add_argument("--alpha-min", type=float, default=0.2)
    parser.add_argument("--alpha-max", type=float, default=0.8)
    parser.add_argument("--temp-min", type=float, default=1.0)
    parser.add_argument("--temp-max", type=float, default=3.0)
    parser.add_argument("--noise-scale-min", type=float, default=None)
    parser.add_argument("--noise-scale-max", type=float, default=None)
    parser.add_argument("--lr-min", type=float, default=None)
    parser.add_argument("--lr-max", type=float, default=None)
    parser.add_argument("--epoch-min", type=float, default=None)
    parser.add_argument("--epoch-max", type=float, default=None)
    parser.add_argument(
        "--epoch-choices",
        type=str,
        default=None,
        help="Comma-separated list of candidate epoch budgets, e.g. '2.0,2.5,3.0'. Overrides epoch-min/epoch-max if set.",
    )
    parser.add_argument("--acquisition-beta", type=float, default=0.5)
    parser.add_argument("--candidate-pool-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-intermediate-eval", action="store_true")
    parser.add_argument("--disable-intermediate-checkpoints", action="store_true")
    parser.add_argument(
        "--eval-mode",
        type=str,
        choices=["generation", "benchmark"],
        default="generation",
        help="Validation evaluator used for SABO search.",
    )
    parser.add_argument("--few-shot-dataset", type=Path, default=None)
    parser.add_argument("--num-shots", type=int, default=0)
    parser.add_argument("--eval-load-in-4bit", action="store_true")
    parser.add_argument("--eval-load-in-8bit", action="store_true")
    parser.add_argument("--eval-quant-compute-dtype", type=str, default="bf16")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_ENDPOINT"] = args.hf_endpoint
    env.setdefault("PYTHONUNBUFFERED", "1")

    base_cfg = load_yaml(args.base_config)
    if not base_cfg.get("distillation", {}).get("enabled", True):
        raise ValueError("Base config must have distillation enabled for SABO search.")

    alpha_range = (args.alpha_min, args.alpha_max)
    temp_range = (args.temp_min, args.temp_max)
    base_noise_scale = float(base_cfg.get("noise", {}).get("scale", 0.0))
    base_learning_rate = float(base_cfg["training"]["learning_rate"])
    base_num_train_epochs = float(base_cfg["training"]["num_train_epochs"])
    noise_scale_range = (
        base_noise_scale if args.noise_scale_min is None else args.noise_scale_min,
        base_noise_scale if args.noise_scale_max is None else args.noise_scale_max,
    )
    learning_rate_range = (
        base_learning_rate if args.lr_min is None else args.lr_min,
        base_learning_rate if args.lr_max is None else args.lr_max,
    )
    epoch_choices = parse_epoch_choices(args.epoch_choices)
    if epoch_choices is None:
        default_epoch = float(args.search_epochs)
        epoch_range = (
            default_epoch if args.epoch_min is None else args.epoch_min,
            default_epoch if args.epoch_max is None else args.epoch_max,
        )
    else:
        epoch_range = (min(epoch_choices), max(epoch_choices))

    if noise_scale_range[0] > noise_scale_range[1]:
        raise ValueError("noise scale min must be <= max")
    if learning_rate_range[0] > learning_rate_range[1]:
        raise ValueError("lr min must be <= max")
    if learning_rate_range[0] <= 0:
        raise ValueError("learning rate bounds must be > 0")
    if epoch_range[0] > epoch_range[1]:
        raise ValueError("epoch min must be <= max")

    rng = np.random.default_rng(args.seed)
    observations: list[dict] = []
    observations_path = output_dir / "observations.jsonl"

    for trial_idx in range(args.num_trials):
        candidate = propose_candidate(
            rng=rng,
            observations=observations,
            alpha_range=alpha_range,
            temp_range=temp_range,
            noise_scale_range=noise_scale_range,
            learning_rate_range=learning_rate_range,
            epoch_range=epoch_range,
            epoch_choices=epoch_choices,
            warmup_trials=args.warmup_trials,
            acquisition_beta=args.acquisition_beta,
            candidate_pool_size=args.candidate_pool_size,
        )

        trial_name = f"trial_{trial_idx + 1:02d}"
        trial_cfg = deepcopy(base_cfg)
        trial_cfg["experiment"]["name"] = f"{base_cfg['experiment']['name']}_sabo_{trial_name}"
        trial_cfg["experiment"]["output_dir"] = str(output_dir / trial_name / "run")
        trial_cfg["distillation"]["method"] = "bayes"
        trial_cfg["distillation"]["alpha"] = round(candidate["alpha"], 4)
        trial_cfg["distillation"]["temperature"] = round(candidate["temperature"], 4)
        if "noise" not in trial_cfg:
            trial_cfg["noise"] = {
                "enabled": False,
                "target_module": "model.norm",
                "distribution": "uniform",
                "scale": 0.0,
            }
        trial_cfg["noise"]["enabled"] = bool(candidate["noise_scale"] > 0.0)
        trial_cfg["noise"]["scale"] = round(candidate["noise_scale"], 4)
        trial_cfg["training"]["learning_rate"] = float(candidate["learning_rate"])
        trial_cfg["training"]["num_train_epochs"] = float(candidate["num_train_epochs"])
        if args.disable_intermediate_eval:
            trial_cfg["training"]["eval_steps"] = 10**12
        if args.disable_intermediate_checkpoints:
            trial_cfg["training"]["save_steps"] = 10**12
            trial_cfg["training"]["save_total_limit"] = 1
        trial_cfg["training"]["logging_steps"] = int(min(max(int(trial_cfg["training"].get("logging_steps", 5)), 5), 20))

        trial_dir = trials_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_cfg_path = trial_dir / "config.yaml"
        save_yaml(trial_cfg_path, trial_cfg)
        print(
            json.dumps(
                {
                    "event": "trial_start",
                    "trial": trial_name,
                    "alpha": trial_cfg["distillation"]["alpha"],
                    "temperature": trial_cfg["distillation"]["temperature"],
                    "noise_scale": float(trial_cfg["noise"]["scale"]),
                    "learning_rate": float(trial_cfg["training"]["learning_rate"]),
                    "num_train_epochs": float(trial_cfg["training"]["num_train_epochs"]),
                    "config_path": str(trial_cfg_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        run_command(
            [args.python, str(repo_root / "scripts/train_distill.py"), "--config", str(trial_cfg_path)],
            env=env,
            cwd=repo_root,
        )
        run_command(
            (
                [
                    args.python,
                    str(repo_root / "scripts/evaluate_benchmark_mcq.py"),
                    "--config",
                    str(trial_cfg_path),
                    "--checkpoint",
                    str(Path(trial_cfg["experiment"]["output_dir"]) / "final"),
                    "--dataset",
                    str(Path(trial_cfg["data"]["val_path"])),
                    "--few-shot-dataset",
                    str(args.few_shot_dataset or Path(trial_cfg["data"]["train_path"])),
                    "--num-shots",
                    str(args.num_shots),
                    "--quant-compute-dtype",
                    args.eval_quant_compute_dtype,
                    "--output-prefix",
                    "val_search",
                ]
                + (["--load-in-4bit"] if args.eval_load_in_4bit else [])
                + (["--load-in-8bit"] if args.eval_load_in_8bit else [])
            )
            if args.eval_mode == "benchmark"
            else [
                args.python,
                str(repo_root / "scripts/evaluate_mcq.py"),
                "--config",
                str(trial_cfg_path),
                "--checkpoint",
                str(Path(trial_cfg["experiment"]["output_dir"]) / "final"),
                "--dataset",
                str(Path(trial_cfg["data"]["val_path"])),
                "--output-prefix",
                "val_search",
            ],
            env=env,
            cwd=repo_root,
        )

        metrics_path = Path(trial_cfg["experiment"]["output_dir"]) / "val_search_metrics.json"
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        observation = {
            "trial": trial_name,
            "alpha": trial_cfg["distillation"]["alpha"],
            "temperature": trial_cfg["distillation"]["temperature"],
            "noise_scale": float(trial_cfg["noise"]["scale"]),
            "learning_rate": float(trial_cfg["training"]["learning_rate"]),
            "num_train_epochs": float(trial_cfg["training"]["num_train_epochs"]),
            "val_accuracy": float(metrics["accuracy"]),
            "num_examples": int(metrics["num_examples"]),
            "config_path": str(trial_cfg_path),
            "output_dir": trial_cfg["experiment"]["output_dir"],
        }
        observations.append(observation)
        with observations_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(observation, ensure_ascii=False) + "\n")
        print(json.dumps(observation, ensure_ascii=False, indent=2), flush=True)

        partial_summary = {
            "base_config": str(args.base_config),
            "num_trials": args.num_trials,
            "completed_trials": len(observations),
            "warmup_trials": args.warmup_trials,
            "search_epochs": args.search_epochs,
            "alpha_range": list(alpha_range),
            "temperature_range": list(temp_range),
            "noise_scale_range": list(noise_scale_range),
            "learning_rate_range": list(learning_rate_range),
            "epoch_range": list(epoch_range),
            "epoch_choices": epoch_choices,
            "best": max(observations, key=lambda item: item["val_accuracy"]),
            "trials": sorted(observations, key=lambda item: item["val_accuracy"], reverse=True),
        }
        save_json(output_dir / "summary.partial.json", partial_summary)

    observations = sorted(observations, key=lambda item: item["val_accuracy"], reverse=True)
    summary = {
        "base_config": str(args.base_config),
        "num_trials": args.num_trials,
        "warmup_trials": args.warmup_trials,
        "search_epochs": args.search_epochs,
        "alpha_range": list(alpha_range),
        "temperature_range": list(temp_range),
        "noise_scale_range": list(noise_scale_range),
        "learning_rate_range": list(learning_rate_range),
        "epoch_range": list(epoch_range),
        "epoch_choices": epoch_choices,
        "best": observations[0],
        "trials": observations,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary["best"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
