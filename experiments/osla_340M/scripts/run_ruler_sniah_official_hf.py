#!/usr/bin/env python3
"""Run RULER S-NIAH with official data generation and local HF inference."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

import fla  # noqa: F401

try:
    import fla.models.os_delta_net  # noqa: F401
except ImportError:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def save_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")
    tmp.replace(path)


def ensure_data(
    ruler_dir: Path,
    data_root: Path,
    task: str,
    length: int,
    model_path: str,
    n: int,
) -> Path:
    data_dir = data_root / str(length)
    out = data_dir / task / "validation.jsonl"
    if out.exists() and len(read_jsonl(out)) == n:
        return out

    cmd = [
        sys.executable,
        str(ruler_dir / "scripts/data/prepare.py"),
        "--save_dir",
        str(data_dir),
        "--benchmark",
        "synthetic",
        "--task",
        task,
        "--tokenizer_path",
        model_path,
        "--tokenizer_type",
        "hf",
        "--max_seq_length",
        str(length),
        "--model_template_type",
        "base",
        "--num_samples",
        str(n),
    ]
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = (
        str(ruler_dir / "scripts")
        + os.pathsep
        + str(ruler_dir / "scripts/data")
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )
    subprocess.run(cmd, cwd=str(ruler_dir / "scripts"), env=env, check=True)
    if not out.exists():
        raise FileNotFoundError(out)
    return out


def generate_predictions(
    model,
    tokenizer,
    samples: list[dict],
    batch_size: int,
    max_new_tokens: int,
) -> list[dict]:
    preds: list[dict] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        prompts = [x["input"] + x.get("answer_prefix", "") for x in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for sample, text in zip(batch, texts):
            preds.append(
                {
                    "index": sample["index"],
                    "input": sample["input"],
                    "answer_prefix": sample.get("answer_prefix", ""),
                    "outputs": sample["outputs"],
                    "pred": text,
                    "others": {"id": sample["index"]},
                    "length": sample.get("length"),
                }
            )
    return preds


def score_all(preds: list[dict]) -> tuple[float, float]:
    vals = []
    for row in preds:
        pred = row["pred"].lower()
        refs = row["outputs"]
        vals.append(sum(1.0 if ref.lower() in pred else 0.0 for ref in refs) / len(refs))
    mean = sum(vals) / len(vals)
    if len(vals) <= 1:
        return mean, float("nan")
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
    return mean, math.sqrt(var / len(vals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ruler_dir", default="/DATA/disk1/cyzhou/RULER")
    parser.add_argument("--tasks", default="niah_single_1,niah_single_2,niah_single_3")
    parser.add_argument("--lengths", default="2048,4096,8192")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output = Path(args.output)
    outdir = output.parent
    pred_root = outdir / "ruler_sniah_official_preds_2k4k8k_n50"
    data_root = outdir / "ruler_sniah_official_data_2k4k8k_n50"
    pred_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]

    started = time.time()
    clean = {
        "model_path": args.model_path,
        "ruler_dir": args.ruler_dir,
        "tasks": tasks,
        "lengths": lengths,
        "batch_size": args.batch_size,
        "device": args.device,
        "num_samples_per_length": args.num_samples,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "elapsed_sec": None,
        "results": {task: {"alias": task} for task in tasks},
        "raw_results": {task: {} for task in tasks},
        "config": {},
    }
    save_json(output, clean)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True

    clean["config"] = {
        "model": args.model_path,
        "model_num_parameters": str(sum(p.numel() for p in model.parameters())),
        "model_dtype": "torch.bfloat16",
        "batch_size": str(args.batch_size),
        "max_new_tokens": "128",
        "metric": "RULER string_match_all fraction",
    }
    save_json(output, clean)

    for task in tasks:
        for length in lengths:
            print(f"START {task} length={length} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
            data_file = ensure_data(
                Path(args.ruler_dir),
                data_root,
                task,
                length,
                args.model_path,
                args.num_samples,
            )
            samples = read_jsonl(data_file)
            preds = generate_predictions(model, tokenizer, samples, args.batch_size, 128)
            pred_file = pred_root / f"{task}_{length}.jsonl"
            with pred_file.open("w") as f:
                for row in preds:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            mean, stderr = score_all(preds)
            metrics = {
                f"{length},none": mean,
                f"{length}_stderr,none": stderr if not math.isnan(stderr) else "N/A",
                "num_samples": len(preds),
                "prediction_file": str(pred_file),
                "data_file": str(data_file),
            }
            clean["raw_results"][task][str(length)] = metrics
            clean["results"][task][f"{length},none"] = mean
            clean["results"][task][f"{length}_stderr,none"] = metrics[f"{length}_stderr,none"]
            clean["elapsed_sec"] = time.time() - started
            save_json(output, clean)
            print(f"DONE {task} length={length}: {mean}", flush=True)

    print("SUMMARY", flush=True)
    for task in tasks:
        vals = {str(length): clean["results"][task].get(f"{length},none") for length in lengths}
        print(f"{task}: {vals}", flush=True)
    print(f"Saved to {output}", flush=True)


if __name__ == "__main__":
    main()
