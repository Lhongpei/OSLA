#!/usr/bin/env python3
"""Evaluate a model with lm_eval by loading it manually to avoid dtype issues."""

import argparse
import json
import sys
import torch
import fla  # noqa — registers FLA model types
import fla.models.os_delta_net  # noqa — registers OSLA model type

from transformers import AutoModelForCausalLM, AutoTokenizer
import lm_eval
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tasks', default='wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} on {args.device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"Model loaded: {type(model).__name__}, params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)

    tasks = [t.strip() for t in args.tasks.split(',')]
    print(f"Running tasks: {tasks}")

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        batch_size=args.batch_size,
        log_samples=False,
    )

    # Print summary table
    print("\n" + "="*80)
    print(f"Results for: {args.model_path}")
    print("="*80)
    for task, metrics in results['results'].items():
        relevant = {k: v for k, v in metrics.items() if not k.endswith('_stderr') and k != 'alias'}
        print(f"  {task:30s}: {relevant}")
    print("="*80)

    # Save results (strip non-serializable objects)
    clean_results = {
        'model_path': args.model_path,
        'results': results['results'],
        'config': {k: str(v) for k, v in results.get('config', {}).items()},
    }
    with open(args.output, 'w') as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
