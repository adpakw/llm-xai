import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer


MODEL_NAME = "Qwen/Qwen3-1.7B"
HOOK_NAME = "blocks.14.hook_resid_post"
DATASET_NAME = "Skylion007/openwebtext"
DATASET_SPLIT = "train"
TEXT_FIELD = "text"
SPLIT_NAME = "train"
MODEL_DTYPE = torch.bfloat16
STORAGE_DTYPE = torch.float32
ADD_SPECIAL_TOKENS = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--text-limit", type=int, default=5000)

    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=None)

    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/qwen3_1_7b_activations_layer14"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any) -> str | None:
    text = value if isinstance(value, str) else ""
    text = text.strip()
    return text or None


def read_dataset(args: argparse.Namespace) -> Iterator[str]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)

    for row in dataset:
        text = clean_text(row[TEXT_FIELD])
        if text:
            yield text


def load_texts(args: argparse.Namespace) -> list[dict[str, Any]]:
    texts = []
    for text in read_dataset(args):
        texts.append({"text_index": len(texts), "text": text})
        if len(texts) >= args.text_limit:
            break

    if not texts:
        raise ValueError("No non-empty texts found.")
    return texts


def batches(items: list[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} is not empty. Use --overwrite or choose another directory.")
    path.mkdir(parents=True, exist_ok=True)
    (path / "activation_shards").mkdir(exist_ok=True)


def json_dump(path: Path, data: Any) -> None:
    def default(value: Any) -> str:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.dtype):
            return str(value).replace("torch.", "")
        raise TypeError(type(value).__name__)

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=default), encoding="utf-8")


def decode_token(tokenizer: Any, token_id: int, cache: dict[int, str]) -> str:
    if token_id not in cache:
        cache[token_id] = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    return cache[token_id]


@torch.inference_mode()
def activations_for_batch(
    model: HookedTransformer,
    texts: list[str],
    text_indices: list[int],
    context_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = model.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=context_size,
        add_special_tokens=ADD_SPECIAL_TOKENS,
    )
    tokens = encoded["input_ids"].to(device)
    mask = encoded["attention_mask"].bool()

    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name == HOOK_NAME, return_type=None)
    acts = cache[HOOK_NAME].detach()

    flat_acts = []
    flat_tokens = []
    flat_text_indices = []
    flat_positions = []
    for row in range(tokens.shape[0]):
        positions = torch.nonzero(mask[row], as_tuple=False).flatten().to(device)
        row_tokens = tokens[row, positions].to("cpu", dtype=torch.long)
        flat_acts.append(acts[row, positions].to("cpu", dtype=STORAGE_DTYPE))
        flat_tokens.append(row_tokens)
        flat_text_indices.append(torch.full_like(row_tokens, text_indices[row]))
        flat_positions.append(positions.to("cpu", dtype=torch.long))

    return torch.cat(flat_acts), torch.cat(flat_tokens), torch.cat(flat_text_indices), torch.cat(flat_positions)


def save_shard(
    artifact_dir: Path,
    acts_parts: list[torch.Tensor],
    token_parts: list[torch.Tensor],
    text_index_parts: list[torch.Tensor],
    position_parts: list[torch.Tensor],
) -> dict[str, Any]:
    acts = torch.cat(acts_parts).contiguous()
    tokens = torch.cat(token_parts).contiguous()
    text_indices = torch.cat(text_index_parts).contiguous()
    token_positions = torch.cat(position_parts).contiguous()

    path = artifact_dir / "activation_shards" / f"{SPLIT_NAME}_activations_00000.pt"
    torch.save(
        {
            "acts": acts,
            "tokens": tokens,
            "text_indices": text_indices,
            "token_positions": token_positions,
            "global_token_start": 0,
            "global_token_stop": len(tokens),
            "split_name": SPLIT_NAME,
        },
        path,
    )
    return {
        "path": str(path.relative_to(artifact_dir)),
        "num_tokens": len(tokens),
        "d_model": acts.shape[1],
        "dtype": str(acts.dtype).replace("torch.", ""),
        "global_token_start": 0,
        "global_token_stop": len(tokens),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device or ("cuda:2" if torch.cuda.is_available() else "cpu")
    prepare_output_dir(args.artifact_dir, args.overwrite)

    print(f"Loading texts into {args.artifact_dir} ...")
    texts = load_texts(args)
    json_dump(args.artifact_dir / "texts.json", texts)
    json_dump(
        args.artifact_dir / "config.json",
        vars(args)
        | {
            "created_at": now(),
            "device": device,
            "model_name": MODEL_NAME,
            "hook_name": HOOK_NAME,
            "dataset_name": DATASET_NAME,
            "dataset_split": DATASET_SPLIT,
            "text_field": TEXT_FIELD,
            "streaming": True,
            "split_name": SPLIT_NAME,
            "model_dtype": str(MODEL_DTYPE).replace("torch.", ""),
            "storage_dtype": str(STORAGE_DTYPE).replace("torch.", ""),
            "add_special_tokens": ADD_SPECIAL_TOKENS,
        },
    )

    print(f"Loading {MODEL_NAME} on {device} ...")
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
        dtype=MODEL_DTYPE,
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = model.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    metadata_path = args.artifact_dir / "token_metadata.csv"
    metadata_fields = [
        "global_token_index",
        "shard_id",
        "shard_token_index",
        "split_name",
        "text_index",
        "token_pos",
        "token_id",
        "token_text",
    ]

    shards: list[dict[str, Any]] = []
    decode_cache: dict[int, str] = {}
    global_index = 0
    acts_parts: list[torch.Tensor] = []
    token_parts: list[torch.Tensor] = []
    text_index_parts: list[torch.Tensor] = []
    position_parts: list[torch.Tensor] = []

    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_fields)
        writer.writeheader()

        total_batches = (len(texts) + args.batch_size - 1) // args.batch_size
        for batch in tqdm(batches(texts, args.batch_size), total=total_batches):
            batch_texts = [item["text"] for item in batch]
            batch_text_indices = [item["text_index"] for item in batch]
            acts, tokens, text_indices_tensor, positions = activations_for_batch(
                model=model,
                texts=batch_texts,
                text_indices=batch_text_indices,
                context_size=args.context_size,
                device=device,
            )

            if args.max_tokens is not None:
                remaining = args.max_tokens - global_index
                if remaining <= 0:
                    break
                acts = acts[:remaining]
                tokens = tokens[:remaining]
                positions = positions[:remaining]
                text_indices_tensor = text_indices_tensor[:remaining]

            for local_i, (token_id, text_index, token_pos) in enumerate(
                zip(tokens.tolist(), text_indices_tensor.tolist(), positions.tolist())
            ):
                writer.writerow(
                    {
                        "global_token_index": global_index + local_i,
                        "shard_id": 0,
                        "shard_token_index": global_index + local_i,
                        "split_name": SPLIT_NAME,
                        "text_index": text_index,
                        "token_pos": token_pos,
                        "token_id": token_id,
                        "token_text": decode_token(tokenizer, token_id, decode_cache),
                    }
                )

            acts_parts.append(acts)
            token_parts.append(tokens)
            text_index_parts.append(text_indices_tensor)
            position_parts.append(positions)
            global_index += len(tokens)

    if global_index:
        shards.append(
            save_shard(
                args.artifact_dir,
                acts_parts,
                token_parts,
                text_index_parts,
                position_parts,
            )
        )

    manifest = {
        "created_at": now(),
        "model_name": MODEL_NAME,
        "hook_name": HOOK_NAME,
        "dataset_name": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "d_model": model.cfg.d_model,
        "device": device,
        "model_dtype": str(MODEL_DTYPE).replace("torch.", ""),
        "storage_dtype": str(STORAGE_DTYPE).replace("torch.", ""),
        "num_texts": len(texts),
        "num_activation_tokens": global_index,
        "num_shards": len(shards),
        "files": {
            "config": "config.json",
            "texts": "texts.json",
            "token_metadata": "token_metadata.csv",
            "activation_shards": shards,
        },
    }
    json_dump(args.artifact_dir / "manifest.json", manifest)
    print(f"Wrote {global_index} activation rows into {len(shards)} shard(s): {args.artifact_dir}")


if __name__ == "__main__":
    main()
