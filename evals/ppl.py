# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
from functools import partial
from typing import Any, Dict, Iterator, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


class PerplexityEvaluator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        block_size: int = 32768,
        bucket_size: int = 2048,
        batch_size: int = 1
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = block_size
        self.bucket_size = bucket_size
        self.batch_size = batch_size

    @staticmethod
    def preprocess(
        examples: Dict[str, List[Any]],
        tokenizer: PreTrainedTokenizer,
        column_name: str = 'text'
    ) -> Dict[str, List[List[int]]]:
        tokenized = tokenizer(examples[column_name])
        return {
            'input_ids': tokenized['input_ids'],
            'length': [len(ids) for ids in tokenized['input_ids']]
        }

    def batchify(self, dataset: Dataset, tokens_per_batch: int) -> Iterator[List[torch.Tensor]]:
        current_tokens = []

        for sentence in dataset:
            tokens = sentence['input_ids'].tolist() if torch.is_tensor(sentence['input_ids']) else list(sentence['input_ids'])
            if not tokens:
                continue
            current_tokens.extend(tokens)

            while len(current_tokens) >= self.block_size * self.batch_size:
                batch = []
                for _ in range(self.batch_size):
                    batch.append(torch.tensor(current_tokens[:self.block_size], dtype=torch.long))
                    current_tokens = current_tokens[self.block_size:]
                yield batch

        if len(current_tokens) >= self.block_size:
            remaining_batches = len(current_tokens) // self.block_size
            remaining_batches = min(remaining_batches, self.batch_size)
            if remaining_batches > 0:
                batch = []
                for _ in range(remaining_batches):
                    batch.append(torch.tensor(current_tokens[:self.block_size], dtype=torch.long))
                    current_tokens = current_tokens[self.block_size:]
                yield batch

    def process_batch(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack(batch).to(self.device)

        # Forward pass without labels to avoid redundant loss computation
        outputs = self.model(input_ids)
        # logits: (B, T, V), shift to get next-token predictions
        # shift logits and labels for next-token prediction
        logits = outputs['logits'][:, :-1]  # (B, T-1, V)
        targets = input_ids[:, 1:]          # (B, T-1)

        # Per-token NLL via cross_entropy (single softmax, no redundancy)
        nlls = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction='none'
        ).reshape(targets.shape)  # (B, T-1)

        return {
            'nlls': nlls,
            'num_tokens': targets.shape[0] * targets.shape[1],
        }

    def evaluate(self, dataset: Dataset, rank: int = 0, world_size: int = 1) -> Dict[str, Any]:
        total_tokens = torch.tensor(0, dtype=torch.long, device=self.device)
        total_sentences = torch.tensor(0, dtype=torch.long, device=self.device)

        num_blocks = (self.block_size - 1) // self.bucket_size + 1
        block_loss = [torch.tensor(0., dtype=torch.float64, device=self.device) for _ in range(num_blocks)]
        block_tokens = [torch.tensor(0, dtype=torch.long, device=self.device) for _ in range(num_blocks)]

        if world_size > 1:
            indices = list(range(rank, len(dataset), world_size))
            dataset = dataset.select(indices)

        bar = tqdm(self.batchify(dataset, self.block_size), disable=(rank != 0))

        for batch in bar:
            batch_outputs = self.process_batch(batch)
            nlls = batch_outputs['nlls']  # (B, T-1)
            B = nlls.shape[0]

            total_tokens += batch_outputs['num_tokens']
            total_sentences += B

            # Accumulate block-level NLL
            # nlls is (B, T-1) where T = block_size, so T-1 tokens per sequence
            for i, j in enumerate(range(0, self.block_size - 1, self.bucket_size)):
                end = min(j + self.bucket_size, nlls.shape[1])
                block_loss[i] += nlls[:, j:end].sum().double()
                block_tokens[i] += B * (end - j)

            if rank == 0:
                ppls = [f"{math.exp(min(bl.item() / max(bt.item(), 1), 20)):6.2f}" for bl, bt in zip(block_loss, block_tokens)]
                bar.set_description_str(f"[{total_tokens.item():10} tokens, {total_sentences.item():8} sentences] " + ' '.join(ppls))

        if world_size > 1:
            for t in [total_tokens, total_sentences] + block_loss + block_tokens:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

        total_nll = sum(bl.item() for bl in block_loss)
        final_ppl = math.exp(total_nll / total_tokens.item())
        block_ppls = [math.exp(bl.item() / max(bt.item(), 1)) for bl, bt in zip(block_loss, block_tokens)]

        return {
            'perplexity': final_ppl,
            'block_perplexities': block_ppls,
            'total_tokens': total_tokens.item(),
            'total_sentences': total_sentences.item()
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity")
    parser.add_argument('-p', '--path', type=str, default='fla-hub/gla-1.3B-100B')
    parser.add_argument('-d', '--data', type=str, default='fla-hub/pg19')
    parser.add_argument('-s', '--split', type=str, default='test')
    parser.add_argument('-n', '--column_name', type=str, default='text')
    parser.add_argument('--block_size', type=int, default=28672)
    parser.add_argument('--bucket_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        device = f'cuda:{local_rank}'
    else:
        if args.device is None:
            from fla.utils import device
        else:
            device = args.device

    torch.manual_seed(0)

    if rank == 0:
        print(f"Loading model {args.path}")
    tokenizer = AutoTokenizer.from_pretrained(args.path)
    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        device_map={"": device}
    ).bfloat16().eval()

    if rank == 0:
        print(f"Loading data {args.data} (split={args.split})")
    dataset = load_dataset(args.data, split=args.split)
    dataset = dataset.map(
        partial(PerplexityEvaluator.preprocess, tokenizer=tokenizer, column_name=args.column_name),
        batched=True,
        num_proc=32
    )

    if rank == 0:
        print(dataset)
        print(f"batch_size={args.batch_size}, block_size={args.block_size}, bucket_size={args.bucket_size}")

    evaluator = PerplexityEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        block_size=args.block_size,
        bucket_size=args.bucket_size,
        batch_size=args.batch_size
    )

    with torch.no_grad():
        results = evaluator.evaluate(dataset, rank=rank, world_size=world_size)

    if rank == 0:
        print("\nEvaluation Results:")
        print(f"Final Perplexity: {results['perplexity']:.2f}")
        print(f"Total Tokens: {results['total_tokens']}")
        print(f"Total Sentences: {results['total_sentences']}")
        print("\nBlock-wise Perplexities:")
        for i, ppl in enumerate(results['block_perplexities']):
            print(f"Block {i}: {ppl:.2f}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
