#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeltaNet Evaluation Toolkit
Unified evaluation entry script
"""

import argparse
import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description='DeltaNet Evaluation Toolkit')
    subparsers = parser.add_subparsers(dest='command', help='Available evaluation commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test model loading and basic functionality')
    test_parser.add_argument('--model', type=str, default='fla-hub/delta_net-1.3B-100B',
                           help='DeltaNet model name')
    
    # Simple evaluation command
    simple_parser = subparsers.add_parser('simple', help='Simple text generation evaluation')
    simple_parser.add_argument('--model', type=str, default='fla-hub/delta_net-1.3B-100B',
                              help='DeltaNet model name')
    
    # Quick evaluation command
    quick_parser = subparsers.add_parser('quick', help='Quick standard evaluation')
    quick_parser.add_argument('--model', type=str, default='fla-hub/delta_net-1.3B-100B',
                             help='DeltaNet model name')
    quick_parser.add_argument('--tasks', type=str, default='wikitext,lambada_openai',
                             help='Evaluation task list')
    quick_parser.add_argument('--batch_size', type=int, default=32,
                             help='Batch size')
    quick_parser.add_argument('--device', type=str, default='cuda',
                             help='Device type')
    
    # Full evaluation command
    full_parser = subparsers.add_parser('full', help='Full standard evaluation')
    full_parser.add_argument('--model', type=str, default='fla-hub/delta_net-1.3B-100B',
                           help='DeltaNet model name')
    full_parser.add_argument('--tasks', type=str, 
                           default='wikitext,lambada_openai,piqa,hellaswag,winogrande,arc_easy,arc_challenge,boolq,sciq,copa,openbookqa',
                           help='Evaluation task list')
    full_parser.add_argument('--batch_size', type=int, default=64,
                           help='Batch size')
    full_parser.add_argument('--device', type=str, default='cuda',
                           help='Device type')
    full_parser.add_argument('--dtype', type=str, default='bfloat16',
                           help='Data type')
    full_parser.add_argument('--output_path', type=str, default='./results',
                           help='Output path for results')
    full_parser.add_argument('--max_length', type=int, default=2048,
                           help='Maximum sequence length')
    full_parser.add_argument('--multi_gpu', action='store_true',
                           help='Whether to use multi-GPU evaluation')
    
    # Perplexity evaluation command
    ppl_parser = subparsers.add_parser('perplexity', help='Perplexity evaluation')
    ppl_parser.add_argument('--model', type=str, default='fla-hub/delta_net-1.3B-100B',
                           help='DeltaNet model name')
    ppl_parser.add_argument('--data', type=str, default='fla-hub/pg19',
                           help='Dataset name')
    ppl_parser.add_argument('--split', type=str, default='train',
                           help='Dataset split')
    ppl_parser.add_argument('--block_size', type=int, default=28672,
                           help='Block size')
    ppl_parser.add_argument('--bucket_size', type=int, default=2048,
                           help='Bucket size')
    ppl_parser.add_argument('--batch_size', type=int, default=32,
                           help='Batch size')
    ppl_parser.add_argument('--device', type=str, default=None,
                           help='Device type')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print("=" * 80)
    print("DeltaNet Evaluation Toolkit")
    print("=" * 80)
    
    # Execute corresponding evaluation based on command
    if args.command == 'test':
        from test_deltanet import main as test_main
        test_main()
    elif args.command == 'simple':
        from simple_eval_deltanet import main as simple_main
        simple_main()
    elif args.command == 'quick':
        from quick_eval_deltanet import main as quick_main
        quick_main()
    elif args.command == 'full':
        from evaluate_deltanet import main as full_main
        full_main()
    elif args.command == 'perplexity':
        from ppl import main as ppl_main
        ppl_main()
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main()
