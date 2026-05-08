#!/usr/bin/env python3
"""FineWeb-Edu fixed-slice perplexity evaluation for OSLA checkpoints."""

import argparse
import json
import math
import os
import time

import torch

import fla  # noqa: F401 - registers FLA model types

try:
    import fla.models.os_delta_net  # noqa: F401
except ImportError:
    pass

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def flush_blocks(model, blocks, device, total_nll, total_tokens):
    if not blocks:
        return total_nll, total_tokens

    input_ids = torch.tensor(blocks, dtype=torch.long, device=device)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = input_ids[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)).float(),
        labels.view(-1),
        reduction="sum",
    )
    total_nll += float(loss.item())
    total_tokens += labels.numel()
    return total_nll, total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_config", default="sample-10BT")
    parser.add_argument("--split", default="train[-10000:]")
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_eval_tokens", type=int, default=10_000_000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = torch.device(args.device)
    start = time.time()

    metadata = {
        "model_path": args.model_path,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "note": "Fixed in-domain slice from cached sample-10BT; FineWeb-Edu cache has train split only.",
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "max_eval_tokens": args.max_eval_tokens,
        "device": str(device),
        "implementation": "batched fixed-block next-token CE, labels shifted by one token",
    }
    print(json.dumps(metadata, indent=2), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    model.config.use_cache = False
    print(f"Loaded model: {type(model).__name__}", flush=True)

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    print(dataset, flush=True)

    eos = tokenizer.eos_token_id
    max_blocks = math.ceil(args.max_eval_tokens / (args.block_size - 1))
    buffer = []
    offset = 0
    pending = []
    total_nll = 0.0
    total_tokens = 0
    total_blocks = 0
    total_docs_read = 0

    pbar = tqdm(total=len(dataset), desc="FW-Edu val")
    with torch.inference_mode():
        for example in dataset:
            ids = tokenizer(example["text"], add_special_tokens=False).input_ids
            if eos is not None:
                ids.append(eos)
            buffer.extend(ids)
            total_docs_read += 1

            while len(buffer) - offset >= args.block_size and total_blocks < max_blocks:
                pending.append(buffer[offset : offset + args.block_size])
                offset += args.block_size
                total_blocks += 1
                if len(pending) >= args.batch_size or total_blocks >= max_blocks:
                    total_nll, total_tokens = flush_blocks(
                        model, pending, device, total_nll, total_tokens
                    )
                    pending.clear()
                    ppl = math.exp(total_nll / max(total_tokens, 1))
                    pbar.set_postfix(
                        blocks=total_blocks,
                        tokens=total_tokens,
                        ppl=f"{ppl:.3f}",
                    )
                if total_blocks >= max_blocks:
                    break

            if offset > args.block_size * 8:
                buffer = buffer[offset:]
                offset = 0

            pbar.update(1)
            if total_blocks >= max_blocks:
                break

        if pending and total_blocks < max_blocks:
            total_nll, total_tokens = flush_blocks(
                model, pending, device, total_nll, total_tokens
            )
            pending.clear()

    pbar.close()

    mean_nll = total_nll / total_tokens
    result = {
        **metadata,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "total_tokens": total_tokens,
        "total_blocks": total_blocks,
        "max_blocks": max_blocks,
        "total_docs_read": total_docs_read,
        "elapsed_sec": time.time() - start,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print("RESULT " + json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
