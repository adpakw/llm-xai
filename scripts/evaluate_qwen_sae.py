import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import torch
from sae_lens import SAE
from tqdm.auto import tqdm


MODEL_NAME = "Qwen/Qwen3-1.7B"
HOOK_NAME = "blocks.14.hook_resid_post"
SAE_DIR = Path("artifacts/qwen3_1_7b_sae_layer14/sae")
ACTIVATION_FILE = Path("artifacts/qwen3_1_7b_activations_layer14/activation_shards/train_activations_00000.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Qwen SAE on saved activations.")
    parser.add_argument("--sae-dir", type=Path, default=SAE_DIR)
    parser.add_argument("--activation-file", type=Path, default=ACTIVATION_FILE)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen3_1_7b_sae_layer14_eval"))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--dead-threshold", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty. Use --overwrite or choose another directory.")
    path.mkdir(parents=True, exist_ok=True)


def load_acts(path: Path, max_tokens: int | None) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", mmap=True)
    acts = payload["acts"] if isinstance(payload, dict) else payload
    if acts.ndim != 2:
        raise ValueError(f"Expected [tokens, d_model] activations, got {tuple(acts.shape)}")
    if max_tokens is not None:
        acts = acts[:max_tokens]
    return acts.float()


def batched_ranges(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


def json_dump(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_feature_summary(path: Path, feature_rows: list[dict[str, Any]]) -> None:
    fieldnames = ["feature_id", "max_activation", "mean_activation", "fire_rate", "is_dead"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)


@torch.inference_mode()
def evaluate_sae(
    sae: SAE,
    acts: torch.Tensor,
    batch_size: int,
    dead_threshold: float,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_tokens, d_in = acts.shape
    mean_activation = acts.mean(dim=0)
    baseline_mse = float(((acts - mean_activation) ** 2).mean())

    d_sae = sae.cfg.d_sae
    squared_error_sum = 0.0
    l2_error_sum = 0.0
    l0_sum = 0.0
    feature_l1_sum = 0.0
    feature_sum = torch.zeros(d_sae)
    feature_max = torch.zeros(d_sae)
    fire_count = torch.zeros(d_sae)

    for start, end in tqdm(list(batched_ranges(n_tokens, batch_size)), desc="evaluate sae"):
        x = acts[start:end].to(device, non_blocking=True)
        feature_acts = sae.encode(x).detach().float()
        recon = sae.decode(feature_acts).detach().float()
        error = recon - x.float()

        squared_error_sum += float(error.square().sum().cpu())
        l2_error_sum += float(torch.linalg.vector_norm(error, dim=1).sum().cpu())
        l0_sum += float((feature_acts > dead_threshold).sum(dim=1).sum().cpu())
        feature_l1_sum += float(feature_acts.abs().sum(dim=1).sum().cpu())

        feature_acts_cpu = feature_acts.cpu()
        feature_sum += feature_acts_cpu.sum(dim=0)
        feature_max = torch.maximum(feature_max, feature_acts_cpu.max(dim=0).values)
        fire_count += (feature_acts_cpu > dead_threshold).sum(dim=0)

    mse = squared_error_sum / (n_tokens * d_in)
    feature_fire_rate = fire_count / n_tokens
    dead_mask = feature_fire_rate == 0
    activation_score = 1.0 - mse / baseline_mse if baseline_mse > 0 else 0.0

    metrics = {
        "model_name": MODEL_NAME,
        "hook_name": HOOK_NAME,
        "num_tokens": n_tokens,
        "d_in": d_in,
        "d_sae": d_sae,
        "baseline_mse": baseline_mse,
        "reconstruction_mse": mse,
        "reconstruction_l2": l2_error_sum / n_tokens,
        "l0": l0_sum / n_tokens,
        "l1": feature_l1_sum / n_tokens,
        "mean_fire_rate": float(feature_fire_rate.mean()),
        "dead_feature_count": int(dead_mask.sum()),
        "dead_feature_rate": float(dead_mask.float().mean()),
        "activation_reconstruction_score": activation_score,
        "dead_threshold": dead_threshold,
    }

    feature_rows = []
    for feature_id in range(d_sae):
        feature_rows.append(
            {
                "feature_id": feature_id,
                "max_activation": float(feature_max[feature_id]),
                "mean_activation": float(feature_sum[feature_id] / n_tokens),
                "fire_rate": float(feature_fire_rate[feature_id]),
                "is_dead": bool(dead_mask[feature_id]),
            }
        )
    return metrics, feature_rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    prepare_output_dir(args.output_dir, args.overwrite)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading activations: {args.activation_file}")
    acts = load_acts(args.activation_file, args.max_tokens)

    print(f"Loading SAE: {args.sae_dir}")
    sae = SAE.load_from_disk(args.sae_dir, device=device)
    sae.eval()

    metrics, feature_rows = evaluate_sae(sae, acts, args.batch_size, args.dead_threshold, device)
    json_dump(args.output_dir / "sae_metrics.json", metrics | {"sae_dir": str(args.sae_dir)})
    write_feature_summary(args.output_dir / "feature_summary.csv", feature_rows)
    print(f"Wrote SAE metrics: {args.output_dir}")


if __name__ == "__main__":
    main()
