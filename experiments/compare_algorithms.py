#!/usr/bin/env python3
"""
Сравнение алгоритмов разбиения на тестовом наборе графов

Запуск:
    python experiments/benchmark_partitioners.py --suite data/test_suite --output results.csv
"""

import os
import sys
import time
import json
import argparse
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition
from algorithms.kernighan_lin import KernighanLin
from algorithms.multilevel import FastMultilevelPartitioner
from algorithms.multilevel_slow import MultilevelPartitioner
from metrics.partition_metrics import PartitionMetrics


class Benchmarker:
    """Бенчмарк для сравнения алгоритмов разбиения"""
    
    def __init__(self, timeout: int = 300):
        self.timeout = timeout  # таймаут в секундах
        self.results = []
    
    def load_graphs(self, suite_dir: str) -> List[Tuple[str, Graph]]:
        """Загрузка всех графов из тестового набора"""
        graphs = []
        suite_path = Path(suite_dir)
        
        for graph_type_dir in suite_path.iterdir():
            if not graph_type_dir.is_dir():
                continue
            
            for file_path in graph_type_dir.glob("*.txt"):
                try:
                    graph = Graph.load_from_file(str(file_path))
                    name = f"{graph_type_dir.name}/{file_path.stem}"
                    graphs.append((name, graph))
                    print(f"  Loaded: {name} ({graph.num_vertices} vertices, {graph.num_edges} edges)")
                except Exception as e:
                    print(f"  Failed to load {file_path}: {e}")
        
        return graphs
    
    def run_algorithm(self, name: str, algorithm, graph: Graph, balance: float = 0.5) -> Dict[str, Any]:
        """Запуск одного алгоритма с замером времени"""
        result = {
            'algorithm': name,
            'success': False,
            'cut_weight': None,
            'time': None,
            'memory_mb': None,
            'balance': None,
            'error': None
        }
        
        try:
            start = time.time()
            partition, metrics = algorithm.partition(graph, balance)
            elapsed = time.time() - start
            
            result['success'] = True
            result['cut_weight'] = metrics.cut_weight
            result['time'] = elapsed
            result['memory_mb'] = metrics.memory_mb
            result['balance'] = partition.balance_quality()
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def benchmark_graph(self, graph: Graph, graph_name: str, algorithms: Dict[str, Any]) -> Dict[str, Any]:
        """Запуск всех алгоритмов на одном графе"""
        print(f"\n  Benchmarking {graph_name}...")
        
        result = {
            'graph_name': graph_name,
            'vertices': graph.num_vertices,
            'edges': graph.num_edges,
            'density': 2 * graph.num_edges / (graph.num_vertices * (graph.num_vertices - 1)) if graph.num_vertices > 1 else 0,
            'vertex_weights': [graph.get_vertex_weight(v) for v in range(min(5, graph.num_vertices))],
            'results': {}
        }
        
        for algo_name, algorithm in algorithms.items():
            print(f"    Running {algo_name}...", end=" ", flush=True)
            algo_result = self.run_algorithm(algo_name, algorithm, graph)
            result['results'][algo_name] = algo_result
            
            if algo_result['success']:
                print(f"cut={algo_result['cut_weight']}, time={algo_result['time']:.3f}s")
            else:
                print(f"FAILED: {algo_result['error']}")
        
        return result
    
    def run_benchmark(self, suite_dir: str, algorithms: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Запуск полного бенчмарка"""
        print("=" * 70)
        print("LOADING GRAPHS")
        print("=" * 70)
        
        graphs = self.load_graphs(suite_dir)
        print(f"\nLoaded {len(graphs)} graphs")
        
        print("\n" + "=" * 70)
        print("RUNNING BENCHMARK")
        print("=" * 70)
        
        results = []
        for i, (name, graph) in enumerate(graphs):
            print(f"\n[{i+1}/{len(graphs)}]")
            result = self.benchmark_graph(graph, name, algorithms)
            results.append(result)
        
        return results
    
    def export_csv(self, results: List[Dict[str, Any]], output_file: str):
        """Экспорт результатов в CSV"""
        rows = []
        
        for res in results:
            for algo_name, algo_res in res['results'].items():
                row = {
                    'graph_name': res['graph_name'],
                    'vertices': res['vertices'],
                    'edges': res['edges'],
                    'density': f"{res['density']:.6f}",
                    'algorithm': algo_name,
                    'cut_weight': algo_res['cut_weight'] if algo_res['success'] else 'FAILED',
                    'time_seconds': f"{algo_res['time']:.4f}" if algo_res['success'] else 'FAILED',
                    'memory_mb': f"{algo_res['memory_mb']:.2f}" if algo_res['success'] else 'FAILED',
                    'balance': f"{algo_res['balance']:.4f}" if algo_res['success'] else 'FAILED',
                    'error': algo_res['error'] or ''
                }
                rows.append(row)
        
        with open(output_file, 'w', newline='') as f:
            fieldnames = ['graph_name', 'vertices', 'edges', 'density', 'algorithm', 
                         'cut_weight', 'time_seconds', 'memory_mb', 'balance', 'error']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"\n✓ Results exported to {output_file}")
    
    def export_json(self, results: List[Dict[str, Any]], output_file: str):
        """Экспорт результатов в JSON"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"✓ Results exported to {output_file}")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Вывод сводной таблицы"""
        print("\n" + "=" * 100)
        print("SUMMARY TABLE")
        print("=" * 100)
        
        # Заголовок
        print(f"{'Graph':<35} {'Vertices':<10} {'Edges':<10} {'Algorithm':<15} {'Cut':<10} {'Time(s)':<10} {'Memory(MB)':<12}")
        print("-" * 100)
        
        for res in results:
            first_algo = True
            for algo_name, algo_res in res['results'].items():
                if first_algo:
                    graph_display = f"{res['graph_name']}"
                    vertices_display = f"{res['vertices']}"
                    edges_display = f"{res['edges']}"
                    first_algo = False
                else:
                    graph_display = ""
                    vertices_display = ""
                    edges_display = ""
                
                if algo_res['success']:
                    cut = f"{algo_res['cut_weight']}"
                    time_s = f"{algo_res['time']:.3f}"
                    memory = f"{algo_res['memory_mb']:.1f}"
                else:
                    cut = "FAILED"
                    time_s = "FAILED"
                    memory = "FAILED"
                
                print(f"{graph_display:<35} {vertices_display:<10} {edges_display:<10} {algo_name:<15} {cut:<10} {time_s:<10} {memory:<12}")
            
            print("-" * 100)
    
    def generate_report(self, results: List[Dict[str, Any]], output_dir: str):
        """Генерация HTML отчёта"""
        report_path = Path(output_dir) / "benchmark_report.html"
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Graph Partitioning Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1e1e1e; color: #d4d4d4; }}
        h1, h2 {{ color: #3498db; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #555; padding: 8px; text-align: left; }}
        th {{ background-color: #2c3e50; color: #3498db; }}
        tr:nth-child(even) {{ background-color: #2a2a2a; }}
        .success {{ color: #2ecc71; }}
        .failed {{ color: #e74c3c; }}
        .best {{ background-color: #27ae60; color: white; }}
        .summary {{ background-color: #2c3e50; padding: 10px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 0 20px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
    </style>
</head>
<body>
    <h1>📊 Graph Partitioning Benchmark Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <div class="metric">
            <div>Total Graphs</div>
            <div class="metric-value">{len(results)}</div>
        </div>
        <div class="metric">
            <div>Algorithms</div>
            <div class="metric-value">{len(results[0]['results']) if results else 0}</div>
        </div>
    </div>
    
    <h2>📈 Results by Graph</h2>
    <table>
        <thead>
            <tr><th>Graph</th><th>Vertices</th><th>Edges</th><th>Density</th>"""
        
        # Добавляем колонки для каждого алгоритма
        if results:
            for algo_name in results[0]['results'].keys():
                html += f"<th>{algo_name} (cut)</th><th>{algo_name} (time)</th>"
        
        html += "</tr></thead><tbody>"
        
        for res in results:
            html += f"""
            <tr>
                <td>{res['graph_name']}</td>
                <td>{res['vertices']}</td>
                <td>{res['edges']}</td>
                <td>{res['density']:.6f}</td>"""
            
            # Находим лучший cut
            best_cut = min((v['cut_weight'] for v in res['results'].values() if v['success']), default=None)
            
            for algo_name, algo_res in res['results'].items():
                if algo_res['success']:
                    cut_class = "best" if algo_res['cut_weight'] == best_cut and best_cut is not None else ""
                    html += f"""
                <td class="{cut_class}">{algo_res['cut_weight']}</td>
                <td>{algo_res['time']:.4f}s</td>"""
                else:
                    html += """
                <td class="failed">FAILED</td>
                <td class="failed">FAILED</td>"""
            
            html += "</tr>"
        
        html += """
    </tbody>
</table>

<h2>📊 Best Results by Graph Size</h2>
<table>
    <thead>
        <tr><th>Size Range</th><th>Best Algorithm (avg cut)</th><th>Best Algorithm (avg time)</th></tr>
    </thead>
    <tbody>"""
        
        # Группировка по размеру
        size_groups = {}
        for res in results:
            size_group = "Small (<200)" if res['vertices'] < 200 else "Medium (200-1000)" if res['vertices'] < 1000 else "Large (>1000)"
            if size_group not in size_groups:
                size_groups[size_group] = []
            size_groups[size_group].append(res)
        
        for size_group, group_results in size_groups.items():
            # Усредняем результаты
            algo_cuts = {}
            algo_times = {}
            for res in group_results:
                for algo_name, algo_res in res['results'].items():
                    if algo_res['success']:
                        algo_cuts.setdefault(algo_name, []).append(algo_res['cut_weight'])
                        algo_times.setdefault(algo_name, []).append(algo_res['time'])
            
            best_cut_algo = min(algo_cuts.keys(), key=lambda a: sum(algo_cuts[a])/len(algo_cuts[a]) if algo_cuts[a] else float('inf')) if algo_cuts else "N/A"
            best_time_algo = min(algo_times.keys(), key=lambda a: sum(algo_times[a])/len(algo_times[a]) if algo_times[a] else float('inf')) if algo_times else "N/A"
            
            html += f"""
        <tr>
            <td>{size_group}</td>
            <td>{best_cut_algo}</td>
            <td>{best_time_algo}</td>
        </tr>"""
        
        html += """
    </tbody>
</table>

</body>
</html>"""
        
        with open(report_path, 'w') as f:
            f.write(html)
        
        print(f"✓ HTML report generated: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Benchmark graph partitioning algorithms')
    parser.add_argument('--suite', type=str, default='data/test_suite',
                       help='Path to test suite directory')
    parser.add_argument('--output', type=str, default='benchmark_results',
                       help='Output file prefix (without extension)')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout per algorithm in seconds')
    parser.add_argument('--balance', type=float, default=0.5,
                       help='Balance ratio for partitioning')
    parser.add_argument('--max-vertices', type=int, default=None,
                       help='Maximum vertices to benchmark (for quick test)')
    
    args = parser.parse_args()
    
    # Настройка алгоритмов
    algorithms = {
        'KL': KernighanLin(max_passes=20, seed=42),
        'Multilevel': MultilevelPartitioner(min_coarse_vertices=20, max_levels=10, seed=42),
        'FastMultilevel': FastMultilevelPartitioner(refinement_passes=1, seed=42)
    }
    
    print("=" * 70)
    print("GRAPH PARTITIONING BENCHMARK")
    print("=" * 70)
    print(f"Test suite: {args.suite}")
    print(f"Balance ratio: {args.balance}")
    print(f"Timeout: {args.timeout}s")
    print(f"\nAlgorithms: {', '.join(algorithms.keys())}")
    
    # Загрузка и фильтрация графов
    benchmarker = Benchmarker(timeout=args.timeout)
    all_graphs = benchmarker.load_graphs(args.suite)
    
    # Фильтрация по размеру
    if args.max_vertices:
        all_graphs = [(name, g) for name, g in all_graphs if g.num_vertices <= args.max_vertices]
        print(f"\nFiltered to {len(all_graphs)} graphs (max vertices = {args.max_vertices})")
    
    if not all_graphs:
        print("No graphs found!")
        return
    
    # Запуск бенчмарка
    results = benchmarker.run_benchmark(args.suite, algorithms)
    
    # Вывод результатов
    benchmarker.print_summary(results)
    
    # Экспорт
    benchmarker.export_csv(results, f"{args.output}.csv")
    benchmarker.export_json(results, f"{args.output}.json")
    benchmarker.generate_report(results, ".")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()