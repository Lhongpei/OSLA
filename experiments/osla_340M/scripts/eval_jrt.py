#!/usr/bin/env python3
"""
Evaluate FDA/SWDE/SQuAD using the JRT (Just Read Twice) protocol from Arora et al.
Matches the Gated DeltaNet paper's evaluation setup:
  - context_length=1000, answer_length=50, cutting_context=True
  - Cloze-style prompt formatting from HazyResearch/prefix-linear-attention
"""

import argparse
import json
import re
from copy import deepcopy
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import fla  # noqa — registers FLA model types
try:
    import fla.models.osla  # noqa
except (ImportError, ModuleNotFoundError):
    pass


def contains_score(prediction: str, labels: List[str]) -> int:
    return max(
        int(bool(re.search(re.compile(re.escape(label), re.IGNORECASE), prediction)))
        for label in labels
    )


def truncate_context(text, answer, tokenizer, context_length=1000, answer_length=50):
    """JRT-style context truncation: center a window around the answer position."""
    desired_length = context_length - answer_length
    tokens = tokenizer.encode(text)

    # Find answer position in text, then convert to token position
    answer_pattern = re.compile(re.escape(answer), re.IGNORECASE)
    match = answer_pattern.search(text)
    if not match:
        # answer not in text, take the last desired_length tokens
        subset = tokens[-desired_length:]
        return tokenizer.decode(subset, skip_special_tokens=True)

    text_before_answer = text[:match.start()]
    answer_tok_pos = len(tokenizer.encode(text_before_answer))

    half = desired_length // 2
    start = max(0, answer_tok_pos - half)
    completed = answer_tok_pos - start
    remaining = desired_length - completed
    end = min(len(tokens), answer_tok_pos + remaining)

    return tokenizer.decode(tokens[start:end], skip_special_tokens=True)


def format_fda_prompt(doc, cut_text):
    """FDA cloze prompt from HazyResearch based_fda task."""
    key = doc["key"]
    upper_key = key[0].upper() + key[1:]
    question = upper_key + ":"
    # Clean up trailing question if present
    t = cut_text.strip("\n").strip(".")
    if not t.endswith("."):
        t += "."
    return t + " " + question


def format_fda_twice_prompt(doc, cut_text):
    """FDA-twice: repeat context with intro."""
    key = doc["key"]
    upper_key = key[0].upper() + key[1:]
    question = upper_key + ":"
    intro_q = question.strip(":")
    t = cut_text.strip("\n").strip(".")
    if not t.endswith("."):
        t += "."
    return f"Information about {intro_q}. " + t + "\n" + t + " " + question


def format_swde_prompt(doc, cut_text):
    """SWDE cloze prompt."""
    key = doc["key"]
    upper_key = key[0].upper() + key[1:]
    question = upper_key + ":"
    t = cut_text.strip("\n").strip(".")
    if not t.endswith("."):
        t += "."
    return t + " " + question


def format_squad_prompt(doc, cut_text):
    """SQuAD completion prompt."""
    return cut_text


TASK_CONFIGS = {
    "based_fda": {
        "dataset": "hazyresearch/based-fda",
        "format_fn": format_fda_prompt,
        "context_key": "text",
    },
    "based_fda_twice": {
        "dataset": "hazyresearch/based-fda",
        "format_fn": format_fda_twice_prompt,
        "context_key": "text",
    },
    "based_swde": {
        "dataset": "hazyresearch/based-swde-v2",
        "format_fn": format_swde_prompt,
        "context_key": "text",
    },
    "based_swde_twice": {
        "dataset": "hazyresearch/based-swde-v2",
        "format_fn": format_swde_prompt,  # same format, context is doubled in truncation
        "context_key": "text",
    },
    "based_squad": {
        "dataset": "hazyresearch/based-squad",
        "format_fn": format_squad_prompt,
        "context_key": "text",
    },
    "based_squad_twice": {
        "dataset": "hazyresearch/based-squad",
        "format_fn": format_squad_prompt,
        "context_key": "text",
    },
}


@torch.no_grad()
def evaluate_task(model, tokenizer, task_name, device, batch_size=32,
                  context_length=1000, answer_length=50):
    cfg = TASK_CONFIGS[task_name]
    ds = load_dataset(cfg["dataset"], split="validation")
    is_twice = task_name.endswith("_twice")
    format_fn = cfg["format_fn"]
    ctx_key = cfg["context_key"]

    scores = []
    skipped = 0

    prompts = []
    targets = []

    for doc in ds:
        raw_text = doc[ctx_key]
        answer = doc["value"]
        if not answer or len(answer) <= 1:
            skipped += 1
            continue

        # Twice variant: duplicate context
        if is_twice and task_name != "based_fda_twice":
            raw_text = raw_text + "\n" + raw_text

        cut_text = truncate_context(raw_text, answer, tokenizer, context_length, answer_length)
        prompt = format_fn(doc, cut_text)
        prompts.append(prompt)
        targets.append(answer)

    print(f"  {task_name}: {len(prompts)} samples (skipped {skipped})")

    # Batch generation
    for i in tqdm(range(0, len(prompts), batch_size), desc=task_name):
        batch_prompts = prompts[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                          max_length=context_length + 64).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        for j, (out, target) in enumerate(zip(outputs, batch_targets)):
            input_len = inputs["input_ids"][j].shape[0]
            gen_tokens = out[input_len:]
            pred = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            # Stop at newline (matching harness behavior)
            pred = pred.split("\n")[0].strip()
            score = contains_score(pred, [target])
            scores.append(score)

    acc = np.mean(scores) if scores else 0.0
    return {"contains": float(acc), "n_samples": len(scores), "n_skipped": skipped}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", required=True, help="comma-separated task names")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context_length", type=int, default=1000)
    parser.add_argument("--answer_length", type=int, default=50)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map=args.device
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model: {type(model).__name__}, params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    tasks = [t.strip() for t in args.tasks.split(",")]
    results = {}

    for task in tasks:
        print(f"\nEvaluating {task}...")
        r = evaluate_task(
            model, tokenizer, task, args.device,
            batch_size=args.batch_size,
            context_length=args.context_length,
            answer_length=args.answer_length,
        )
        results[task] = r
        print(f"  => contains: {r['contains']:.4f} ({r['n_samples']} samples)")

    print("\n" + "=" * 60)
    for task, r in results.items():
        print(f"  {task:30s}: contains={r['contains']:.4f}")
    print("=" * 60)

    with open(args.output, "w") as f:
        json.dump({"model_path": args.model_path, "results": results}, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
