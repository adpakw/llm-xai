import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from sae_lens.saes.sae import TrainStepInput
from sae_lens.saes.standard_sae import StandardTrainingSAE, StandardTrainingSAEConfig
from tqdm.auto import trange


MODEL_NAME = "Qwen/Qwen3-1.7B"
HOOK_NAME = "blocks.14.hook_resid_post"
ACTIVATION_FILE = Path("artifacts/qwen3_1_7b_activations_layer14/activation_shards/train_activations_00000.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-file", type=Path, default=ACTIVATION_FILE)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen3_1_7b_sae_layer14"))
    parser.add_argument("--d-sae", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--l1-coeff", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--max-train-tokens", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dump(path: Path, data: dict[str, Any]) -> None:
    def default(value: Any) -> str:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.dtype):
            return str(value).replace("torch.", "")
        raise TypeError(type(value).__name__)

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=default), encoding="utf-8")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty. Use --overwrite or choose another directory.")
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)


def load_acts(path: Path, max_tokens: int | None) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", mmap=True)
    acts = payload["acts"] if isinstance(payload, dict) else payload
    if acts.ndim != 2:
        raise ValueError(f"Expected [num_tokens, d_model] activations, got shape {tuple(acts.shape)}")
    if max_tokens is not None:
        acts = acts[:max_tokens]
    return acts.float()


def split_indices(num_rows: int, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_rows, generator=generator)
    val_size = int(num_rows * val_fraction)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    if len(train_idx) == 0:
        raise ValueError("Train split is empty. Lower --val-fraction or use more activations.")
    return train_idx, val_idx


def sample_batch(acts: torch.Tensor, indices: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
    choice = torch.randint(0, len(indices), (batch_size,))
    rows = indices[choice]
    return acts[rows].to(device, non_blocking=True)


def train_step_input(x: torch.Tensor, l1_coeff: float, step: int, is_logging_step: bool) -> TrainStepInput:
    return TrainStepInput(
        sae_in=x,
        coefficients={"l1": l1_coeff},
        dead_neuron_mask=None,
        n_training_steps=step,
        is_logging_step=is_logging_step,
    )


def metric_value(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def output_metrics(output: Any) -> dict[str, float]:
    feature_acts = output.feature_acts.detach()
    active_features = (feature_acts > 0).any(dim=0)
    dead_feature_count = feature_acts.shape[1] - int(active_features.sum().cpu())
    metrics = {
        "loss": metric_value(output.loss),
        "mse": metric_value(output.losses["mse_loss"]),
        "l1_loss": metric_value(output.losses["l1_loss"]),
        "l0": float((feature_acts > 0).float().sum(dim=1).mean().cpu()),
        "feature_mean": float(feature_acts.mean().cpu()),
        "dead_feature_count": float(dead_feature_count),
        "dead_feature_rate": float(dead_feature_count / feature_acts.shape[1]),
    }
    return metrics


def compute_baseline_mse(acts: torch.Tensor, train_idx: torch.Tensor) -> float:
    train_acts = acts[train_idx]
    mean_acts = train_acts.mean(dim=0)  # [d_model]
    mse_baseline = ((train_acts - mean_acts) ** 2).mean()
    return metric_value(mse_baseline)


@torch.no_grad()
def evaluate(
    sae: StandardTrainingSAE,
    acts: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    l1_coeff: float,
    device: str,
    step: int,
    baseline_mse: float,
) -> dict[str, float]:
    if len(indices) == 0:
        return {"val_loss": float("nan"), "val_mse": float("nan"), "val_l1_loss": float("nan"),
                "val_l0": float("nan"), "val_feature_mean": float("nan"),
                "val_dead_feature_count": float("nan"), "val_dead_feature_rate": float("nan"),
                "val_activation_reconstruction_score": float("nan")}
    x = sample_batch(acts, indices, min(batch_size, len(indices)), device)
    output = sae.training_forward_pass(train_step_input(x, l1_coeff, step, is_logging_step=True))
    metrics = output_metrics(output)
    val_score = 1.0 - metrics["mse"] / baseline_mse if baseline_mse > 0 else 0.0
    return {
        "val_loss": metrics["loss"],
        "val_mse": metrics["mse"],
        "val_l1_loss": metrics["l1_loss"],
        "val_l0": metrics["l0"],
        "val_feature_mean": metrics["feature_mean"],
        "val_dead_feature_count": metrics["dead_feature_count"],
        "val_dead_feature_rate": metrics["dead_feature_rate"],
        "val_activation_reconstruction_score": val_score,
    }


def save_sae(sae: StandardTrainingSAE, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    sae.save_model(path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prepare_output_dir(args.output_dir, args.overwrite)

    print(f"Loading activations: {args.activation_file}")
    acts = load_acts(args.activation_file, args.max_train_tokens)
    train_idx, val_idx = split_indices(len(acts), args.val_fraction, args.seed)

    # Baseline: MSE if we always predict the mean of the training activations
    baseline_mse = compute_baseline_mse(acts, train_idx)
    print(f"Baseline MSE (mean predictor): {baseline_mse:.6f}")

    cfg = StandardTrainingSAEConfig(
        d_in=acts.shape[1],
        d_sae=args.d_sae,
        dtype="float32",
        device=device,
        l1_coefficient=args.l1_coeff,
        lp_norm=1.0,
    )
    sae = StandardTrainingSAE(cfg).to(device)
    sae.b_dec.data.copy_(acts[train_idx[: min(len(train_idx), 100_000)]].mean(dim=0).to(device))
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)

    config = vars(args) | {
        "created_at": now(),
        "model_name": MODEL_NAME,
        "hook_name": HOOK_NAME,
        "d_in": acts.shape[1],
        "device": device,
        "num_tokens": len(acts),
        "num_train_tokens": len(train_idx),
        "num_val_tokens": len(val_idx),
        "baseline_mse": baseline_mse,
        "sae_lens_class": "StandardTrainingSAE",
        "objective": "SAE-Lens standard loss: reconstruction MSE plus L1 feature sparsity",
    }
    json_dump(args.output_dir / "config.json", config)

    fieldnames = [
        "step",
        "loss",
        "mse",
        "l1_loss",
        "l0",
        "feature_mean",
        "activation_reconstruction_score",
        "dead_feature_count",
        "dead_feature_rate",
        "val_loss",
        "val_mse",
        "val_l1_loss",
        "val_l0",
        "val_feature_mean",
        "val_activation_reconstruction_score",
        "val_dead_feature_count",
        "val_dead_feature_rate",
    ]
    last_metrics: dict[str, float] = {}

    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for step in trange(1, args.steps + 1):
            is_logging_step = step == 1 or step == args.steps or step % args.eval_every == 0
            x = sample_batch(acts, train_idx, args.batch_size, device)
            output = sae.training_forward_pass(train_step_input(x, args.l1_coeff, step, is_logging_step))

            optimizer.zero_grad(set_to_none=True)
            output.loss.backward()
            optimizer.step()

            if is_logging_step:
                train_metrics = output_metrics(output)
                train_score = 1.0 - train_metrics["mse"] / baseline_mse if baseline_mse > 0 else 0.0
                val_metrics = evaluate(sae, acts, val_idx, args.batch_size, args.l1_coeff, device, step, baseline_mse)
                last_metrics = {
                    **train_metrics,
                    "activation_reconstruction_score": train_score,
                    **val_metrics,
                }
                writer.writerow({"step": step, **last_metrics})
                handle.flush()

            if step % args.save_every == 0 and step != args.steps:
                save_sae(sae, args.output_dir / "checkpoints" / f"step_{step}")

    save_sae(sae, args.output_dir / "sae")
    summary = {
        "created_at": now(),
        "sae_dir": "sae",
        "metrics_path": "metrics.csv",
        "final_step": args.steps,
        "final_metrics": last_metrics,
        "config": config,
    }
    json_dump(args.output_dir / "summary.json", summary)
    print(f"Wrote SAE artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()