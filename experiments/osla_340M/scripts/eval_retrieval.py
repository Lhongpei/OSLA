#!/usr/bin/env python3
"""Evaluate retrieval tasks (NIAH, FDA, SWDE, SQuAD) aligned with Gated DeltaNet paper."""

import argparse
import json
import torch
import fla  # noqa
import fla.models.osla  # noqa

from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_length', type=int, default=None)
    parser.add_argument('--metadata', type=str, default=None)
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} on {args.device}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"Model loaded: {type(model).__name__}, params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    lm_kwargs = dict(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    if args.max_length is not None:
        lm_kwargs['max_length'] = args.max_length
    lm = HFLM(**lm_kwargs)

    tasks = [t.strip() for t in args.tasks.split(',')]
    print(f"Running tasks: {tasks}")

    eval_kwargs = dict(
        model=lm,
        tasks=tasks,
        batch_size=args.batch_size,
        log_samples=False,
    )
    if args.metadata:
        eval_kwargs['metadata'] = json.loads(args.metadata)

    results = evaluator.simple_evaluate(**eval_kwargs)

    print("\n" + "="*80)
    print(f"Results for: {args.model_path}")
    print("="*80)
    for task, metrics in results['results'].items():
        relevant = {k: v for k, v in metrics.items() if not k.endswith('_stderr') and k != 'alias'}
        print(f"  {task:30s}: {relevant}")
    print("="*80)

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
