#!/usr/bin/env python3
"""
Сравнение алгоритмов KL и Multilevel на тестовом наборе графов.

Запуск:
    python experiments/compare_kl_vs_multilevel.py --suite data/test_suite --output results.csv
"""

import sys
import time
import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Any

sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from algorithms.kernighan_lin import KernighanLin
from algorithms.multilevel_slow import MultilevelPartitioner


def load_graphs(suite_dir: str) -> List[Tuple[str, Graph]]:
    """Загружает все .txt графы из поддиректорий suite_dir."""
    graphs = []
    suite_path = Path(suite_dir)
    for graph_file in suite_path.rglob("*.txt"):
        try:
            g = Graph.load_from_file(str(graph_file))
            name = str(graph_file.relative_to(suite_path))
            graphs.append((name, g))
        except Exception as e:
            print(f"Error loading {graph_file}: {e}")
    return graphs


def run_algorithm(algo, graph: Graph, balance: float, max_passes: int = 20, seed: int = 42):
    """Запускает алгоритм и возвращает (cut_weight, time_seconds, balance_quality)."""
    start = time.perf_counter()
    if algo == "KL":
        partitioner = KernighanLin(max_passes=max_passes, seed=seed)
    else:
        partitioner = MultilevelPartitioner(coarsen_to=50, max_passes=5, seed=seed)
    partition, metrics = partitioner.partition(graph, balance)
    elapsed = time.perf_counter() - start
    return metrics.cut_weight, elapsed, partition.balance_quality()


def main():
    parser = argparse.ArgumentParser(description="Compare KL and Multilevel")
    parser.add_argument("--suite", type=str, default="data/test_suite", help="Directory with test graphs")
    parser.add_argument("--output", type=str, default="comparison_results.csv", help="Output CSV file")
    parser.add_argument("--balance", type=float, default=0.5, help="Target balance ratio")
    parser.add_argument("--max-passes-kl", type=int, default=20, help="Max passes for KL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("Loading graphs...")
    graphs = load_graphs(args.suite)
    if not graphs:
        print("No graphs found.")
        return
    print(f"Loaded {len(graphs)} graphs.\n")

    results = []
    for name, g in graphs:
        print(f"Processing {name} (V={g.num_vertices}, E={g.num_edges})...")

        cut_kl, time_kl, bal_kl = run_algorithm("KL", g, args.balance, args.max_passes_kl, args.seed)
        cut_ml, time_ml, bal_ml = run_algorithm("ML", g, args.balance, max_passes=5, seed=args.seed)

        improvement = (cut_kl - cut_ml) / max(cut_kl, 1) * 100
        speedup = time_kl / max(time_ml, 1e-9)

        results.append({
            "graph": name,
            "vertices": g.num_vertices,
            "edges": g.num_edges,
            "kl_cut": cut_kl,
            "kl_time": time_kl,
            "kl_balance": bal_kl,
            "ml_cut": cut_ml,
            "ml_time": time_ml,
            "ml_balance": bal_ml,
            "improvement_%": improvement,
            "speedup": speedup,
        })

        print(f"  KL: cut={cut_kl}, time={time_kl:.3f}s, bal={bal_kl:.3f}")
        print(f"  ML: cut={cut_ml}, time={time_ml:.3f}s, bal={bal_ml:.3f}")
        print(f"  Improvement: {improvement:+.1f}%, Speedup: {speedup:.2f}x\n")

    # Write CSV
    with open(args.output, "w", newline="") as f:
        fieldnames = [
            "graph", "vertices", "edges",
            "kl_cut", "kl_time", "kl_balance",
            "ml_cut", "ml_time", "ml_balance",
            "improvement_%", "speedup"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {args.output}")

    # Summary statistics
    avg_improvement = sum(r["improvement_%"] for r in results) / len(results)
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    wins = sum(1 for r in results if r["improvement_%"] > 0)
    print("\n=== SUMMARY ===")
    print(f"Average improvement (ML over KL): {avg_improvement:+.1f}%")
    print(f"Average speedup (KL time / ML time): {avg_speedup:.2f}x")
    print(f"Multilevel wins on {wins}/{len(results)} graphs")


if __name__ == "__main__":
    main()