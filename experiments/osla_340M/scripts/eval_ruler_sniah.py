#!/usr/bin/env python3
"""Run RULER S-NIAH-1/2/3 at 2K/4K/8K for an OSLA checkpoint."""

import argparse
import json
import os
import time

import torch

import fla  # noqa: F401 - registers FLA model types

try:
    import fla.models.os_delta_net  # noqa: F401
except ImportError:
    pass

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tasks", default="niah_single_1,niah_single_2,niah_single_3"
    )
    parser.add_argument("--lengths", default="2048,4096,8192")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=float, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    metadata = {
        "max_seq_lengths": lengths,
        "tokenizer": args.model_path,
        "pretrained": args.model_path,
    }

    print(
        json.dumps(
            {
                "model_path": args.model_path,
                "tasks": tasks,
                "lengths": lengths,
                "batch_size": args.batch_size,
                "device": args.device,
                "limit": args.limit,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            indent=2,
        ),
        flush=True,
    )

    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(
        f"Loaded model: {type(model).__name__}, params={sum(p.numel() for p in model.parameters())}",
        flush=True,
    )

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=max(lengths),
        device=args.device,
    )
    task_manager = TaskManager(include_defaults=True, metadata=metadata)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=False,
        confirm_run_unsafe_code=True,
        task_manager=task_manager,
        metadata=metadata,
    )

    clean = {
        "model_path": args.model_path,
        "tasks": tasks,
        "lengths": lengths,
        "elapsed_sec": time.time() - start,
        "results": results["results"],
        "config": {k: str(v) for k, v in results.get("config", {}).items()},
    }
    with open(args.output, "w") as f:
        json.dump(clean, f, indent=2, default=str)
        f.write("\n")

    print("SUMMARY", flush=True)
    for task, metrics in clean["results"].items():
        vals = {
            str(length): metrics.get(f"{length},none")
            for length in lengths
            if f"{length},none" in metrics
        }
        print(f"{task}: {vals}", flush=True)
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
