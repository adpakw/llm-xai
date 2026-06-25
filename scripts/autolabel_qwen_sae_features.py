import argparse
import csv
import json
import random
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI, OpenAIError
import torch
from sae_lens import SAE
from tqdm.auto import tqdm
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-1.7B"
HOOK_NAME = "blocks.14.hook_resid_post"
SAE_DIR = Path("artifacts/qwen3_1_7b_sae_layer14/sae")
ACTIVATION_FILE = Path("artifacts/qwen3_1_7b_activations_layer14/activation_shards/train_activations_00000.pt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Autolabel SAE features with an OpenAI-compatible LLM endpoint.")
    p.add_argument("--sae-dir", type=Path, default=SAE_DIR)
    p.add_argument("--activation-file", type=Path, default=ACTIVATION_FILE)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen3_1_7b_sae_layer14_autolabels"))
    p.add_argument("--max-features", type=int, default=500)
    p.add_argument("--top-contexts", type=int, default=15)
    p.add_argument("--judge-contexts", type=int, default=4)
    p.add_argument("--judge-negative-contexts", type=int, default=4)
    p.add_argument("--context-window", type=int, default=32)
    p.add_argument("--scan-batch-size", type=int, default=2048)
    p.add_argument("--feature-batch-size", type=int, default=64)
    p.add_argument("--min-activation", type=float, default=0.0)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--llm-model", default=None, help="If omitted, the first /v1/models id is used.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-label-tokens", type=int, default=512)
    p.add_argument("--max-judge-tokens", type=int, default=64)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty. Use --overwrite or choose another directory.")
    path.mkdir(parents=True, exist_ok=True)


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", mmap=True)
    if not isinstance(payload, dict) or "acts" not in payload:
        raise ValueError(f"{path} must be a torch-saved dict with an 'acts' tensor.")
    return payload


def batched_ranges(n: int, batch_size: int):
    for start in range(0, n, batch_size):
        yield start, min(start + batch_size, n)


@torch.inference_mode()
def compute_feature_stats(
    sae: SAE,
    acts: torch.Tensor,
    batch_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
    d_sae = sae.cfg.d_sae
    max_activation = torch.zeros(d_sae)
    activation_sum = torch.zeros(d_sae)
    fire_count = torch.zeros(d_sae)

    for start, end in tqdm(list(batched_ranges(len(acts), batch_size)), desc="scan features"):
        x = acts[start:end].to(device, non_blocking=True).float()
        feature_acts = sae.encode(x).detach().float().cpu()
        max_activation = torch.maximum(max_activation, feature_acts.max(dim=0).values)
        activation_sum += feature_acts.sum(dim=0)
        fire_count += (feature_acts > 0).sum(dim=0)

    return {
        "max_activation": max_activation,
        "mean_activation": activation_sum / len(acts),
        "fire_rate": fire_count / len(acts),
    }


def choose_features(stats: dict[str, torch.Tensor], max_features: int, min_activation: float) -> list[int]:
    scores = stats["max_activation"]
    eligible = torch.nonzero(scores >= min_activation, as_tuple=False).flatten()
    if len(eligible) == 0:
        return []
    chosen_scores = scores[eligible]
    _values, order = torch.topk(chosen_scores, k=min(max_features, len(chosen_scores)))
    return [int(eligible[i]) for i in order.tolist()]


def feature_summary_rows(feature_ids: list[int], stats: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows = []
    for feature_id in feature_ids:
        rows.append(
            {
                "feature_id": feature_id,
                "max_activation": float(stats["max_activation"][feature_id]),
                "mean_activation": float(stats["mean_activation"][feature_id]),
                "fire_rate": float(stats["fire_rate"][feature_id]),
            }
        )
    return rows


def add_top_context(
    store: dict[int, list[tuple[float, int]]],
    feature_id: int,
    activation: float,
    row_index: int,
    top_k: int,
) -> None:
    rows = store.setdefault(feature_id, [])
    rows.append((activation, row_index))
    rows.sort(key=lambda item: item[0], reverse=True)
    del rows[top_k:]


@torch.inference_mode()
def collect_top_context_indices(
    sae: SAE,
    acts: torch.Tensor,
    feature_ids: list[int],
    batch_size: int,
    feature_batch_size: int,
    top_k: int,
    device: str,
) -> dict[int, list[tuple[float, int]]]:
    top_contexts = {feature_id: [] for feature_id in feature_ids}

    for start, end in tqdm(list(batched_ranges(len(acts), batch_size)), desc="top contexts"):
        x = acts[start:end].to(device, non_blocking=True).float()
        feature_acts = sae.encode(x).detach().float().cpu()

        for f_start in range(0, len(feature_ids), feature_batch_size):
            batch_feature_ids = feature_ids[f_start : f_start + feature_batch_size]
            selected = feature_acts[:, batch_feature_ids].T
            values, positions = torch.topk(selected, k=min(top_k, selected.shape[1]), dim=1)
            for local_feature, feature_id in enumerate(batch_feature_ids):
                for value, position in zip(values[local_feature].tolist(), positions[local_feature].tolist()):
                    if value > 0:
                        add_top_context(top_contexts, feature_id, float(value), start + int(position), top_k)

    return top_contexts


def decode_context(
    tokenizer: Any,
    tokens: torch.Tensor,
    text_indices: torch.Tensor,
    token_positions: torch.Tensor,
    row_index: int,
    context_window: int,
) -> dict[str, Any]:
    text_index = int(text_indices[row_index])
    token_pos = int(token_positions[row_index])

    start = row_index
    while start > 0 and int(text_indices[start - 1]) == text_index and token_pos - int(token_positions[start - 1]) <= context_window:
        start -= 1

    end = row_index + 1
    while end < len(tokens) and int(text_indices[end]) == text_index and int(token_positions[end]) - token_pos <= context_window:
        end += 1

    left_ids = tokens[start:row_index].tolist()
    focus_id = int(tokens[row_index])
    right_ids = tokens[row_index + 1 : end].tolist()
    left = tokenizer.decode(left_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    focus = tokenizer.decode([focus_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    right = tokenizer.decode(right_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return {
        "global_token_index": row_index,
        "text_index": text_index,
        "token_pos": token_pos,
        "token_id": focus_id,
        "token_text": focus,
        "left": left,
        "focus": focus,
        "right": right,
        "context": f"{left}[[{focus}]]{right}",
    }


def context_rows(
    tokenizer: Any,
    payload: dict[str, Any],
    top_contexts: dict[int, list[tuple[float, int]]],
    context_window: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tokens = payload["tokens"]
    text_indices = payload["text_indices"]
    token_positions = payload["token_positions"]

    for feature_id, examples in top_contexts.items():
        for rank, (activation, row_index) in enumerate(examples, start=1):
            rows.append(
                {
                    "feature_id": feature_id,
                    "rank": rank,
                    "activation": activation,
                    **decode_context(tokenizer, tokens, text_indices, token_positions, row_index, context_window),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def discover_model(client: OpenAI) -> str:
    models = client.models.list().data
    if not models:
        raise RuntimeError("No models returned by OpenAI-compatible endpoint.")
    return models[0].id


def label_prompt(feature_id: int, examples: list[dict[str, Any]]) -> str:
    lines = [
        "You are labeling a sparse autoencoder feature from a language model.",
        "Look for what the highlighted token contexts have in common.",
        "Return strict JSON with keys: label, explanation, confidence.",
        "Use a short noun phrase for label. Confidence is a number from 0 to 1.",
        "",
        f"Feature id: {feature_id}",
        "Top activating contexts:",
    ]
    for row in examples:
        lines.append(f"- activation={row['activation']:.4f}: {row['context']}")
    return "\n".join(lines)


def judge_prompt(label_row: dict[str, Any], context: str) -> str:
    return "\n".join(
        [
            "Decide whether this context matches a sparse autoencoder feature label.",
            "Return strict JSON with keys: active, reason.",
            "active must be true if the highlighted token likely activates the feature, otherwise false.",
            "",
            f"Feature label: {label_row.get('label', '')}",
            f"Feature explanation: {label_row.get('explanation', '')}",
            f"Context: {context}",
        ]
    )


def extract_json(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"LLM response did not contain text content: {text!r}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], str]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "include_reasoning": False,
            "reasoning_effort": "none",
        },
    )
    choice = response.choices[0]
    content = choice.message.content
    if content is None:
        raise ValueError(f"LLM message.content is None. Full choice: {choice.model_dump_json()}")
    return extract_json(content), content


def rows_by_feature(rows: list[dict[str, Any]], feature_ids: list[int]) -> dict[int, list[dict[str, Any]]]:
    grouped = {feature_id: [] for feature_id in feature_ids}
    for row in rows:
        grouped[int(row["feature_id"])].append(row)
    return grouped


def random_negative_contexts(
    tokenizer: Any,
    payload: dict[str, Any],
    feature_ids: list[int],
    excluded_indices: dict[int, set[int]],
    contexts_per_feature: int,
    context_window: int,
    seed: int,
) -> dict[int, list[dict[str, Any]]]:
    rng = random.Random(seed)
    n_tokens = len(payload["tokens"])
    negatives: dict[int, list[dict[str, Any]]] = {feature_id: [] for feature_id in feature_ids}

    for feature_id in feature_ids:
        excluded = excluded_indices.get(feature_id, set())
        attempts = 0
        while len(negatives[feature_id]) < contexts_per_feature and attempts < contexts_per_feature * 100:
            attempts += 1
            row_index = rng.randrange(n_tokens)
            if row_index in excluded:
                continue
            row = decode_context(
                tokenizer,
                payload["tokens"],
                payload["text_indices"],
                payload["token_positions"],
                row_index,
                context_window,
            )
            row.update(
                {
                    "feature_id": feature_id,
                    "rank": "",
                    "activation": 0.0,
                    "expected_active": False,
                    "source": "negative_random",
                }
            )
            negatives[feature_id].append(row)
    return negatives


def autolabel(
    args: argparse.Namespace,
    feature_ids: list[int],
    label_rows_by_feature: dict[int, list[dict[str, Any]]],
    summary_by_feature: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    client = make_client(args.base_url, args.api_key)
    model = args.llm_model or discover_model(client)
    labels = []
    for feature_id in tqdm(feature_ids, desc="label features"):
        examples = label_rows_by_feature[feature_id]
        prompt = label_prompt(feature_id, examples)
        try:
            parsed, raw = call_llm(client, model, prompt, args.temperature, args.max_label_tokens)
            label = {
                "feature_id": feature_id,
                "label": parsed.get("label", ""),
                "explanation": parsed.get("explanation", ""),
                "confidence": parsed.get("confidence", None),
                "raw_response": raw,
                **summary_by_feature[feature_id],
            }
        except (OpenAIError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            label = {
                "feature_id": feature_id,
                "label": "",
                "explanation": f"autolabel_failed: {exc!r}",
                "confidence": None,
                "raw_response": "",
                **summary_by_feature[feature_id],
            }
        labels.append(label)
        time.sleep(0.05)
    return labels


def bool_from_llm(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "active"}:
            return True
        if lowered in {"false", "no", "0", "inactive"}:
            return False
    raise ValueError(f"Expected a boolean-like active value, got {value!r}")


def score_judge_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for row in rows if row["expected_active"] and row["predicted_active"])
    tn = sum(1 for row in rows if not row["expected_active"] and not row["predicted_active"])
    fp = sum(1 for row in rows if not row["expected_active"] and row["predicted_active"])
    fn = sum(1 for row in rows if row["expected_active"] and not row["predicted_active"])
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "judge_accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "num_judged_contexts": total,
    }


def judge_labels(
    args: argparse.Namespace,
    labels: list[dict[str, Any]],
    positive_rows_by_feature: dict[int, list[dict[str, Any]]],
    negative_rows_by_feature: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    client = make_client(args.base_url, args.api_key)
    model = args.llm_model or discover_model(client)
    judge_rows: list[dict[str, Any]] = []

    for label_row in tqdm(labels, desc="judge labels"):
        feature_id = int(label_row["feature_id"])
        contexts = []
        for row in positive_rows_by_feature.get(feature_id, [])[: args.judge_contexts]:
            contexts.append(row | {"expected_active": True, "source": "heldout_active"})
        contexts.extend(negative_rows_by_feature.get(feature_id, [])[: args.judge_negative_contexts])

        for context_row in contexts:
            try:
                parsed, raw = call_llm(
                    client,
                    model,
                    judge_prompt(label_row, context_row["context"]),
                    args.temperature,
                    args.max_judge_tokens,
                )
                predicted_active = bool_from_llm(parsed.get("active"))
                error = ""
            except (OpenAIError, TimeoutError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raw = ""
                predicted_active = False
                error = f"judge_failed: {exc!r}"

            expected_active = bool(context_row["expected_active"])
            judge_rows.append(
                {
                    "feature_id": feature_id,
                    "label": label_row.get("label", ""),
                    "source": context_row.get("source", ""),
                    "expected_active": expected_active,
                    "predicted_active": predicted_active,
                    "correct": expected_active == predicted_active,
                    "activation": context_row.get("activation", ""),
                    "context": context_row["context"],
                    "raw_response": raw,
                    "error": error,
                }
            )
            time.sleep(0.05)

    return judge_rows, score_judge_rows(judge_rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    prepare_output_dir(args.output_dir, args.overwrite)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SAE: {args.sae_dir}")
    sae = SAE.load_from_disk(args.sae_dir, device=device)
    sae.eval()

    print(f"Loading activations: {args.activation_file}")
    payload = load_payload(args.activation_file)
    acts = payload["acts"]

    stats = compute_feature_stats(sae, acts, args.scan_batch_size, device)
    feature_ids = choose_features(stats, args.max_features, args.min_activation)
    if not feature_ids:
        raise ValueError("No features selected.")

    summaries = feature_summary_rows(feature_ids, stats)
    write_csv(
        args.output_dir / "feature_summary.csv",
        summaries,
        ["feature_id", "max_activation", "mean_activation", "fire_rate"],
    )

    top_context_indices = collect_top_context_indices(
        sae=sae,
        acts=acts,
        feature_ids=feature_ids,
        batch_size=args.scan_batch_size,
        feature_batch_size=args.feature_batch_size,
        top_k=args.top_contexts + args.judge_contexts,
        device=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    all_context_rows = context_rows(tokenizer, payload, top_context_indices, args.context_window)
    top_rows = [row for row in all_context_rows if int(row["rank"]) <= args.top_contexts]
    heldout_rows = [
        row | {"expected_active": True, "source": "heldout_active"}
        for row in all_context_rows
        if int(row["rank"]) > args.top_contexts
    ]
    write_csv(
        args.output_dir / "top_feature_contexts.csv",
        top_rows,
        [
            "feature_id",
            "rank",
            "activation",
            "global_token_index",
            "text_index",
            "token_pos",
            "token_id",
            "token_text",
            "left",
            "focus",
            "right",
            "context",
        ],
    )

    config = vars(args) | {
        "model_name": MODEL_NAME,
        "hook_name": HOOK_NAME,
        "num_activation_tokens": len(acts),
        "num_features_selected": len(feature_ids),
        "selected_feature_ids": feature_ids,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    summary_by_feature = {int(row["feature_id"]): row for row in summaries}
    label_rows_by_feature = rows_by_feature(top_rows, feature_ids)
    labels = autolabel(args, feature_ids, label_rows_by_feature, summary_by_feature)

    with (args.output_dir / "feature_labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in labels:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_csv(
        args.output_dir / "feature_labels.csv",
        labels,
        ["feature_id", "label", "explanation", "confidence", "max_activation", "mean_activation", "fire_rate", "raw_response"],
    )

    excluded_indices = {
        feature_id: {row_index for _activation, row_index in top_context_indices.get(feature_id, [])}
        for feature_id in feature_ids
    }
    negative_rows_by_feature = random_negative_contexts(
        tokenizer,
        payload,
        feature_ids,
        excluded_indices,
        args.judge_negative_contexts,
        args.context_window,
        args.seed + 1,
    )
    judge_rows, judge_metrics = judge_labels(
        args,
        labels,
        rows_by_feature(heldout_rows, feature_ids),
        negative_rows_by_feature,
    )
    write_csv(
        args.output_dir / "interpretability_judge.csv",
        judge_rows,
        [
            "feature_id",
            "label",
            "source",
            "expected_active",
            "predicted_active",
            "correct",
            "activation",
            "context",
            "raw_response",
            "error",
        ],
    )
    (args.output_dir / "interpretability_metrics.json").write_text(
        json.dumps(judge_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote labels: {args.output_dir}")


if __name__ == "__main__":
    main()
