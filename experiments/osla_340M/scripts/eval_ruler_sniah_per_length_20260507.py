#!/usr/bin/env python3
"""Run RULER S-NIAH by evaluating each length with its own 50-example limit.

The lm-eval RULER tasks generate 500 examples per requested length and then
concatenate lengths. A global --limit 50 only evaluates the first 50 examples of
the first length. This wrapper runs each task/length pair separately so "n=50"
really means 50 prompts per setting.
"""

from __future__ import annotations

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


def save_json(path: str, data: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tasks", default="niah_single_1,niah_single_2,niah_single_3"
    )
    parser.add_argument("--lengths", default="2048,4096,8192")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit_per_length", type=float, default=50.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]

    header = {
        "model_path": args.model_path,
        "tasks": tasks,
        "lengths": lengths,
        "batch_size": args.batch_size,
        "device": args.device,
        "limit_per_length": args.limit_per_length,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    print(json.dumps(header, indent=2), flush=True)

    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
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

    clean = {
        **header,
        "elapsed_sec": None,
        "results": {task: {"alias": task} for task in tasks},
        "raw_results": {task: {} for task in tasks},
        "config": {},
    }
    save_json(args.output, clean)

    for task in tasks:
        for length in lengths:
            metadata = {
                "max_seq_lengths": [length],
                "tokenizer": args.model_path,
                "pretrained": args.model_path,
            }
            print(f"START {task} length={length} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
            task_manager = TaskManager(include_defaults=True, metadata=metadata)
            result = evaluator.simple_evaluate(
                model=lm,
                tasks=[task],
                batch_size=args.batch_size,
                limit=args.limit_per_length,
                log_samples=False,
                confirm_run_unsafe_code=True,
                task_manager=task_manager,
                metadata=metadata,
            )
            metrics = result["results"][task]
            clean["raw_results"][task][str(length)] = metrics
            key = f"{length},none"
            stderr_key = f"{length}_stderr,none"
            clean["results"][task][key] = metrics.get(key)
            if stderr_key in metrics:
                clean["results"][task][stderr_key] = metrics[stderr_key]
            if result.get("config"):
                clean["config"] = {k: str(v) for k, v in result["config"].items()}
            clean["elapsed_sec"] = time.time() - start
            save_json(args.output, clean)
            print(f"DONE {task} length={length}: {clean['results'][task].get(key)}", flush=True)

    print("SUMMARY", flush=True)
    for task in tasks:
        vals = {
            str(length): clean["results"][task].get(f"{length},none")
            for length in lengths
        }
        print(f"{task}: {vals}", flush=True)
    print(f"Saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
