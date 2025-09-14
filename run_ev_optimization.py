#!/usr/bin/env python3
import argparse
from ev_charging_optimizer import main_optimization_pipeline

def parse_args():
    p = argparse.ArgumentParser(description="Run Cagliari MCF optimization (EUR)")
    p.add_argument("--seed", type=int, default=123,
                   help="Random seed (default: 123)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Optional output root directory (default: results/)")
    return p.parse_args()

def main():
    args = parse_args()
    solution = main_optimization_pipeline(seed=args.seed, output_dir=args.output_dir)
    return solution

if __name__ == "__main__":
    main()