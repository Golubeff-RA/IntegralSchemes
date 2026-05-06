#!/usr/bin/env python3
"""
Сравнение алгоритмов разбиения: Kernighan-Lin vs Multilevel

Запуск:
    python experiments/compare_algorithms.py --sizes 100,500,1000,5000 --trials 5

Параметры:
    --sizes: размеры графов через запятую (по умолчанию: 100,200,500,1000)
    --trials: количество повторений для усреднения (по умолчанию: 3)
    --output: выходной CSV файл (по умолчанию: comparison_results.csv)
    --seed: seed для воспроизводимости (по умолчанию: 42)
"""

import sys
import time
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from algorithms.kernighan_lin import KernighanLin
from algorithms.multilevel import FastMultilevelPartitioner, UltraFastMultilevelPartitioner #MultilevelPartitioner, AdaptiveMultilevelPartitioner
from data.generators import FastClusterGenerator, BarabasiAlbertGenerator


class AlgorithmComparator:
    """Сравнение алгоритмов разбиения"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.results = []
    
    # В experiments/compare_algorithms.py измените generate_test_graphs:

    def generate_test_graphs(self, sizes: List[int], graph_type: str = 'cluster') -> List[Tuple[str, Graph]]:
        """Генерация тестовых графов разных размеров"""
        graphs = []
        
        for size in sizes:
            if graph_type in ['cluster', 'both']:
                num_clusters = max(2, size // 30)  # Больше кластеров
                vertices_per_cluster = size // num_clusters
                
                # Количество рёбер: примерно 3-5 на вершину (разреженный граф)
                target_edges = min(size * 5, 100000)  # Не более 5 рёбер на вершину
                
                gen = FastClusterGenerator(
                    num_clusters=num_clusters,
                    vertices_per_cluster=vertices_per_cluster,
                    target_edges=target_edges,  # Уменьшено!
                    intra_ratio=0.5,  # Меньше внутренних связей
                    weight_range=(1, 10),  # Простые веса
                    vertex_weight_range=(1, 20),
                    seed=self.seed
                )
                graph = gen.generate_ultra_fast()  # Используем ультра-быстрый метод
                graphs.append((f"cluster_{size}", graph))
        
        return graphs
    
    def run_comparison(self, graphs: List[Tuple[str, Graph]], 
                   trials: int = 3,
                   balance_ratio: float = 0.5) -> List[Dict[str, Any]]:
        """Запуск сравнения с оптимизированными версиями"""
        
        results = []
        
        for name, graph in graphs:
            print(f"\n{'='*70}")
            print(f"Testing: {name}")
            print(f"  Vertices: {graph.num_vertices}, Edges: {graph.num_edges}")
            print(f"{'='*70}")
            
            # KL алгоритм (медленный, но качественный)
            kl_times = []
            kl_cuts = []
            
            for t in range(min(trials, 2)):  # Меньше повторов для KL
                print(f"  KL: trial {t+1}/{min(trials,2)}...", end=" ", flush=True)
                kl = KernighanLin(max_passes=15, seed=self.seed + t)
                partition, metrics = kl.partition(graph, balance_ratio)
                kl_times.append(metrics.time_seconds)
                kl_cuts.append(metrics.cut_weight)
                print(f"cut={metrics.cut_weight}, time={metrics.time_seconds:.3f}s")
            
            # Fast Multilevel
            ml_times = []
            ml_cuts = []
            
            for t in range(trials):
                print(f"  FastML: trial {t+1}/{trials}...", end=" ", flush=True)
                ml = FastMultilevelPartitioner(
                    min_coarse_vertices=max(50, graph.num_vertices // 20),
                    refinement_passes=1,
                    seed=self.seed + t
                )
                partition, metrics = ml.partition(graph, balance_ratio)
                ml_times.append(metrics.time_seconds)
                ml_cuts.append(metrics.cut_weight)
                print(f"cut={metrics.cut_weight}, time={metrics.time_seconds:.3f}s")
            
            # Ultra Fast Multilevel для больших графов
            if graph.num_vertices > 1000:
                uf_times = []
                uf_cuts = []
                
                for t in range(trials):
                    print(f"  UltraFastML: trial {t+1}/{trials}...", end=" ", flush=True)
                    uf = UltraFastMultilevelPartitioner(seed=self.seed + t)
                    partition, metrics = uf.partition(graph, balance_ratio)
                    uf_times.append(metrics.time_seconds)
                    uf_cuts.append(metrics.cut_weight)
                    print(f"cut={metrics.cut_weight}, time={metrics.time_seconds:.3f}s")
                
                best_ml = min(ml_cuts)
                best_uf = min(uf_cuts)
                best_ml_time = min(ml_times)
                best_uf_time = min(uf_times)
                
                ml_cut = best_uf if best_uf < best_ml else best_ml
                ml_time = best_uf_time if best_uf < best_ml else best_ml_time
            else:
                ml_cut = min(ml_cuts)
                ml_time = min(ml_times)
            
            result = {
                'graph_name': name,
                'num_vertices': graph.num_vertices,
                'num_edges': graph.num_edges,
                'kl_cut': min(kl_cuts),
                'kl_time': min(kl_times),
                'ml_cut': ml_cut,
                'ml_time': ml_time,
                'cut_improvement': (min(kl_cuts) - ml_cut) / max(1, min(kl_cuts)) * 100,
                'speedup': min(kl_times) / max(0.001, ml_time),
            }
            
            results.append(result)
            
            print(f"\n  Summary:")
            print(f"    KL:        cut={result['kl_cut']}, time={result['kl_time']:.4f}s")
            print(f"    Multilevel: cut={result['ml_cut']}, time={result['ml_time']:.4f}s")
            print(f"    Improvement: {result['cut_improvement']:+.2f}%")
            print(f"    Speedup: {result['speedup']:.2f}x")
        
        return results
    
    def _std(self, values: List[float]) -> float:
        """Вычисление стандартного отклонения"""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def print_summary(self):
        """Вывод сводки результатов"""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "=" * 90)
        print("COMPARISON SUMMARY")
        print("=" * 90)
        print(f"{'Graph':<20} {'Vertices':<10} {'Edges':<10} {'KL cut':<10} {'ML cut':<10} {'Improve':<10} {'Speedup':<10} {'Winner':<12}")
        print("-" * 90)
        
        for r in self.results:
            print(f"{r['graph_name']:<20} {r['num_vertices']:<10} {r['num_edges']:<10} "
                  f"{r['kl_best_cut']:<10} {r['ml_best_cut']:<10} "
                  f"{r['cut_improvement']:>+8.2f}%   {r['speedup']:>8.2f}x   {r['winner']:<12}")
        
        print("=" * 90)
        
        # Общая статистика
        improvements = [r['cut_improvement'] for r in self.results]
        speedups = [r['speedup'] for r in self.results]
        
        print(f"\nOverall Statistics:")
        print(f"  Average improvement: {sum(improvements)/len(improvements):+.2f}%")
        print(f"  Average speedup: {sum(speedups)/len(speedups):.2f}x")
        print(f"  Multilevel wins: {sum(1 for r in self.results if r['winner'] == 'Multilevel')}/{len(self.results)}")
    
    def export_to_csv(self, filename: str):
        """Экспорт результатов в CSV"""
        if not self.results:
            print("No results to export")
            return
        
        fieldnames = [
            'graph_name', 'num_vertices', 'num_edges', 'density',
            'kl_time_avg', 'kl_time_std', 'kl_cut_avg', 'kl_cut_std', 'kl_balance_avg', 'kl_best_cut', 'kl_best_time',
            'ml_time_avg', 'ml_time_std', 'ml_cut_avg', 'ml_cut_std', 'ml_balance_avg', 'ml_best_cut', 'ml_best_time', 'ml_levels_avg',
            'cut_improvement', 'speedup', 'winner'
        ]
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow(r)
        
        print(f"\nResults exported to {filename}")
    
    def plot_results(self, output_dir: str = "."):
        """Построение графиков (если есть matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            if not self.results:
                print("No results to plot")
                return
            
            # Подготовка данных
            sizes = [r['num_vertices'] for r in self.results]
            kl_cuts = [r['kl_best_cut'] for r in self.results]
            ml_cuts = [r['ml_best_cut'] for r in self.results]
            improvements = [r['cut_improvement'] for r in self.results]
            speedups = [r['speedup'] for r in self.results]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # График 1: Cut size vs размер графа
            ax1 = axes[0, 0]
            ax1.plot(sizes, kl_cuts, 'o-', label='KL', color='red', linewidth=2, markersize=8)
            ax1.plot(sizes, ml_cuts, 's-', label='Multilevel', color='blue', linewidth=2, markersize=8)
            ax1.set_xlabel('Number of Vertices')
            ax1.set_ylabel('Cut Size')
            ax1.set_title('Cut Size Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # График 2: Улучшение vs размер графа
            ax2 = axes[0, 1]
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            ax2.bar(range(len(sizes)), improvements, color=colors, alpha=0.7)
            ax2.set_xticks(range(len(sizes)))
            ax2.set_xticklabels([str(s) for s in sizes], rotation=45)
            ax2.set_xlabel('Number of Vertices')
            ax2.set_ylabel('Improvement (%)')
            ax2.set_title('Multilevel Improvement over KL')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.grid(True, alpha=0.3)
            
            # График 3: Speedup vs размер графа
            ax3 = axes[1, 0]
            ax3.plot(sizes, speedups, 'd-', color='purple', linewidth=2, markersize=8)
            ax3.axhline(y=1, color='gray', linestyle='--', linewidth=1)
            ax3.set_xlabel('Number of Vertices')
            ax3.set_ylabel('Speedup (KL time / ML time)')
            ax3.set_title('Speedup Comparison')
            ax3.grid(True, alpha=0.3)
            
            # График 4: Распределение времени
            ax4 = axes[1, 1]
            kl_times = [r['kl_best_time'] for r in self.results]
            ml_times = [r['ml_best_time'] for r in self.results]
            x = np.arange(len(sizes))
            width = 0.35
            ax4.bar(x - width/2, kl_times, width, label='KL', color='red', alpha=0.7)
            ax4.bar(x + width/2, ml_times, width, label='Multilevel', color='blue', alpha=0.7)
            ax4.set_xticks(x)
            ax4.set_xticklabels([str(s) for s in sizes], rotation=45)
            ax4.set_xlabel('Number of Vertices')
            ax4.set_ylabel('Time (seconds)')
            ax4.set_title('Runtime Comparison')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/comparison_plot.png", dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"Plot saved to {output_dir}/comparison_plot.png")
            
        except ImportError:
            print("Matplotlib not available, skipping plots")


def main():
    parser = argparse.ArgumentParser(description='Compare KL and Multilevel algorithms')
    parser.add_argument('--sizes', type=str, default='100,200,500,1000',
                        help='Graph sizes to test (comma-separated)')
    parser.add_argument('--trials', type=int, default=3,
                        help='Number of trials for averaging')
    parser.add_argument('--graph-type', type=str, default='cluster',
                        choices=['cluster', 'barabasi', 'both'],
                        help='Type of graphs to generate')
    parser.add_argument('--output', type=str, default='comparison_results.csv',
                        help='Output CSV file')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting')
    
    args = parser.parse_args()
    
    # Парсим размеры
    sizes = [int(s.strip()) for s in args.sizes.split(',')]
    
    print("=" * 70)
    print("ALGORITHM COMPARISON: Kernighan-Lin vs Multilevel")
    print("=" * 70)
    print(f"Sizes: {sizes}")
    print(f"Trials per size: {args.trials}")
    print(f"Graph type: {args.graph_type}")
    print(f"Seed: {args.seed}")
    
    # Создаём компаратор
    comparator = AlgorithmComparator(seed=args.seed)
    
    # Генерируем графы
    print("\nGenerating test graphs...")
    graphs = comparator.generate_test_graphs(sizes, args.graph_type)
    print(f"Generated {len(graphs)} graphs")
    
    # Запускаем сравнение
    comparator.run_comparison(graphs, trials=args.trials)
    
    # Выводим сводку
    comparator.print_summary()
    
    # Экспортируем результаты
    comparator.export_to_csv(args.output)
    
    # Строим графики
    if not args.no_plot:
        comparator.plot_results()
    
    print("\nDone!")


if __name__ == "__main__":
    main()