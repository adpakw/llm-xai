import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer


MODEL_NAME = "Qwen/Qwen3-1.7B"
HOOK_NAME = "blocks.14.hook_resid_post"
SAE_DIR = Path("artifacts/qwen3_1_7b_sae_layer14/sae")
LABELS_CSV = Path("artifacts/qwen3_1_7b_sae_layer14_autolabels/feature_labels.csv")


DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clamp trained SAE features during Qwen generation.")
    p.add_argument("--feature-id", type=int, nargs="+", required=True)
    p.add_argument("--steering-values", type=float, nargs="+", default=[20.0, 80.0, 160.0])
    p.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "The city council discussed the new public transport plan because",
            "The animal moved quietly through the room and",
            "In this essay, I will explain why",
        ],
    )
    p.add_argument("--sae-dir", type=Path, default=SAE_DIR)
    p.add_argument("--labels-csv", type=Path, default=LABELS_CSV)
    p.add_argument("--output-json", type=Path, default=Path("artifacts/qwen3_1_7b_sae_steering/results.json"))
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def default_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return "cuda:2" if torch.cuda.device_count() > 2 else "cuda"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labels(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["feature_id"]): row for row in csv.DictReader(handle)}


def generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "stop_at_eos": False,
        "verbose": False,
    }
    if args.temperature <= 0:
        kwargs["do_sample"] = False
    else:
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p
    return kwargs


@torch.no_grad()
def selected_feature_stats(
    model: HookedTransformer,
    sae: SAE,
    prompt: str,
    feature_ids: list[int],
    device: str,
) -> list[dict[str, Any]]:
    tokens = model.to_tokens(prompt, prepend_bos=True).to(device)
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name == HOOK_NAME, return_type=None)
    acts = cache[HOOK_NAME][0]
    feature_acts = sae.encode(acts.to(next(sae.parameters()).device)).detach().float().cpu()
    str_tokens = model.to_str_tokens(tokens[0].detach().cpu())

    rows = []
    for feature_id in feature_ids:
        values = feature_acts[:, feature_id]
        max_value, max_pos = values.max(dim=0)
        rows.append(
            {
                "feature_id": int(feature_id),
                "last_token_activation": float(values[-1]),
                "max_prompt_activation": float(max_value),
                "max_prompt_token_pos": int(max_pos),
                "max_prompt_token": str_tokens[int(max_pos)],
            }
        )
    return rows


def make_feature_clamp_hook(sae: SAE, feature_ids: list[int], steering_value: float):
    sae_device = next(sae.parameters()).device

    @torch.no_grad()
    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:
        last = act[:, -1:, :]
        flat = last.reshape(-1, last.shape[-1]).to(sae_device)
        features = sae.encode(flat)
        recon = sae.decode(features)

        patched = features.clone()
        patched[:, feature_ids] = float(steering_value)
        patched_recon = sae.decode(patched)

        out = act.clone()
        delta = (patched_recon - recon).reshape_as(last).to(act.device, act.dtype)
        out[:, -1:, :] = out[:, -1:, :] + delta
        return out

    return hook_fn


def generate(model: HookedTransformer, prompt: str, args: argparse.Namespace) -> str:
    return model.generate(prompt, **generation_kwargs(args))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device or default_device()

    labels = load_labels(args.labels_csv)
    feature_labels = {
        feature_id: labels.get(feature_id, {}).get("label", "")
        for feature_id in args.feature_id
    }

    print(f"Loading SAE: {args.sae_dir}")
    sae = SAE.load_from_disk(args.sae_dir, device=device)
    sae.eval()

    print(f"Loading model: {MODEL_NAME} on {device}")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
        dtype=DTYPES[args.dtype],
        trust_remote_code=True,
    )
    model.eval()

    results: list[dict[str, Any]] = []
    for prompt in args.prompts:
        baseline = generate(model, prompt, args)
        prompt_stats = selected_feature_stats(model, sae, prompt, args.feature_id, device)

        steered_runs = []
        for value in args.steering_values:
            hook_fn = make_feature_clamp_hook(sae, args.feature_id, value)
            with model.hooks(fwd_hooks=[(HOOK_NAME, hook_fn)]):
                text = generate(model, prompt, args)
            steered_runs.append({"steering_value": float(value), "generation": text})

        row = {
            "prompt": prompt,
            "baseline_generation": baseline,
            "feature_stats_before_steering": prompt_stats,
            "steered_generations": steered_runs,
        }
        results.append(row)

        print("\nPROMPT")
        print(prompt)
        print("\nBASELINE")
        print(baseline)
        for run in steered_runs:
            print(f"\nSTEERED @ {run['steering_value']:g}")
            print(run["generation"])

    output = {
        "model_name": MODEL_NAME,
        "hook_name": HOOK_NAME,
        "sae_dir": str(args.sae_dir),
        "feature_ids": args.feature_id,
        "feature_labels": feature_labels,
        "steering_method": "clamp selected SAE feature activations at the final token position",
        "steering_values": args.steering_values,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote: {args.output_json}")


if __name__ == "__main__":
    main()
