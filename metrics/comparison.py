"""
Сравнение различных алгоритмов разбиения графов
"""

import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition
from algorithms.base_partitioner import BasePartitioner
from .partition_metrics import PartitionMetrics
from .performance_tracker import PerformanceTracker, PerformanceRecord


class AlgorithmComparator:
    """
    Сравнение нескольких алгоритмов разбиения
    
    Пример использования:
        comparator = AlgorithmComparator()
        comparator.add_algorithm('KL', KernighanLin())
        comparator.add_algorithm('Multilevel', MultilevelPartitioner())
        
        results = comparator.compare_on_graph(graph, num_parts=2)
        comparator.print_comparison()
    """
    
    def __init__(self):
        self.algorithms: Dict[str, BasePartitioner] = {}
        self.results: List[Dict[str, Any]] = []
        self.tracker = PerformanceTracker()
    
    def add_algorithm(self, name: str, algorithm: BasePartitioner) -> None:
        """
        Добавление алгоритма для сравнения
        
        Args:
            name: имя алгоритма
            algorithm: экземпляр алгоритма
        """
        self.algorithms[name] = algorithm
    
    def remove_algorithm(self, name: str) -> None:
        """Удаление алгоритма"""
        if name in self.algorithms:
            del self.algorithms[name]
    
    def compare_on_graph(self, graph: Graph, num_parts: int = 2,
                         balance_ratio: float = 0.5,
                         repeat: int = 1) -> Dict[str, Any]:
        """
        Сравнение алгоритмов на одном графе
        
        Args:
            graph: граф
            num_parts: количество частей
            balance_ratio: коэффициент баланса
            repeat: количество повторений для усреднения
        
        Returns:
            Словарь с результатами сравнения
        """
        results = {}
        
        for name, algorithm in self.algorithms.items():
            print(f"\nRunning {name}...")
            
            times = []
            cuts = []
            balances = []
            partitions = []
            
            for i in range(repeat):
                # Замеряем производительность
                self.tracker.start(f"{name}_run_{i}")
                
                partition = algorithm.partition(graph, num_parts, balance_ratio)
                
                record = self.tracker.end()
                metrics = PartitionMetrics(graph, partition)
                all_metrics = metrics.get_all_metrics()
                
                times.append(record.elapsed_time)
                cuts.append(all_metrics['cut_size'])
                balances.append(all_metrics['balance_quality'])
                partitions.append(partition)
            
            results[name] = {
                'algorithm': algorithm,
                'partition': partitions[np.argmin(cuts)],  # Лучшее разбиение
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'avg_cut': np.mean(cuts),
                'std_cut': np.std(cuts),
                'min_cut': np.min(cuts),
                'max_cut': np.max(cuts),
                'avg_balance': np.mean(balances),
                'std_balance': np.std(balances),
                'all_times': times,
                'all_cuts': cuts,
                'all_balances': balances
            }
        
        # Добавляем сравнение
        results['comparison'] = self._compare_results(results)
        
        self.results.append({
            'graph': graph,
            'num_parts': num_parts,
            'balance_ratio': balance_ratio,
            'repeat': repeat,
            'results': results
        })
        
        return results
    
    def compare_on_multiple_graphs(self, graphs: List[Tuple[str, Graph]],
                                   num_parts: int = 2,
                                   balance_ratio: float = 0.5) -> pd.DataFrame:
        """
        Сравнение алгоритмов на нескольких графах
        
        Args:
            graphs: список (имя_графа, граф)
            num_parts: количество частей
            balance_ratio: коэффициент баланса
        
        Returns:
            DataFrame с результатами
        """
        all_results = []
        
        for graph_name, graph in graphs:
            print(f"\n{'='*60}")
            print(f"Testing on graph: {graph_name}")
            print(f"  Vertices: {graph.num_vertices}, Edges: {graph.num_edges}")
            print(f"{'='*60}")
            
            results = self.compare_on_graph(graph, num_parts, balance_ratio, repeat=1)
            
            for algo_name, algo_results in results.items():
                if algo_name != 'comparison':
                    all_results.append({
                        'graph_name': graph_name,
                        'num_vertices': graph.num_vertices,
                        'num_edges': graph.num_edges,
                        'algorithm': algo_name,
                        'cut_size': algo_results['min_cut'],
                        'time_seconds': algo_results['min_time'],
                        'balance_quality': algo_results['avg_balance'],
                        'part_sizes': algo_results['partition'].part_sizes.tolist()
                    })
        
        df = pd.DataFrame(all_results)
        return df
    
    def _compare_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Сравнение результатов алгоритмов
        """
        comparison = {}
        
        algo_names = [name for name in results if name != 'comparison']
        
        if len(algo_names) >= 2:
            baseline = algo_names[0]
            baseline_cut = results[baseline]['min_cut']
            baseline_time = results[baseline]['min_time']
            
            for name in algo_names[1:]:
                cut_improvement = (baseline_cut - results[name]['min_cut']) / baseline_cut * 100
                time_speedup = baseline_time / results[name]['min_time']
                
                comparison[f'{name}_vs_{baseline}'] = {
                    'cut_improvement_percent': cut_improvement,
                    'time_speedup': time_speedup,
                    'better_cut': results[name]['min_cut'] < baseline_cut,
                    'faster': results[name]['min_time'] < baseline_time
                }
        
        # Находим лучший алгоритм по cut size
        best_cut_algo = min(algo_names, key=lambda x: results[x]['min_cut'])
        best_time_algo = min(algo_names, key=lambda x: results[x]['min_time'])
        best_balance_algo = max(algo_names, key=lambda x: results[x]['avg_balance'])
        
        comparison['best_cut_algorithm'] = best_cut_algo
        comparison['best_time_algorithm'] = best_time_algo
        comparison['best_balance_algorithm'] = best_balance_algo
        
        return comparison
    
    def print_comparison(self, detailed: bool = False):
        """Вывод сравнения"""
        if not self.results:
            print("No results to display")
            return
        
        last_result = self.results[-1]
        results = last_result['results']
        
        print("\n" + "=" * 80)
        print("ALGORITHM COMPARISON")
        print("=" * 80)
        print(f"Graph: {last_result['graph'].num_vertices} vertices, "
              f"{last_result['graph'].num_edges} edges")
        print(f"Number of parts: {last_result['num_parts']}")
        print(f"Balance ratio: {last_result['balance_ratio']}")
        
        print("\n" + "-" * 80)
        print(f"{'Algorithm':<20} {'Cut Size':<12} {'Time (s)':<12} {'Balance':<12} {'Speedup':<10}")
        print("-" * 80)
        
        baseline_time = None
        baseline_cut = None
        
        for name, algo_results in results.items():
            if name == 'comparison':
                continue
            
            if baseline_time is None:
                baseline_time = algo_results['min_time']
                baseline_cut = algo_results['min_cut']
            
            speedup = baseline_time / algo_results['min_time']
            
            print(f"{name:<20} {algo_results['min_cut']:<12} "
                  f"{algo_results['min_time']:<12.4f} "
                  f"{algo_results['avg_balance']:<12.4f} "
                  f"{speedup:<10.2f}x")
        
        print("-" * 80)
        
        if 'comparison' in results:
            comp = results['comparison']
            print(f"\nBest cut: {comp.get('best_cut_algorithm', 'N/A')}")
            print(f"Best time: {comp.get('best_time_algorithm', 'N/A')}")
            print(f"Best balance: {comp.get('best_balance_algorithm', 'N/A')}")
        
        print("=" * 80)
        
        if detailed:
            self._print_detailed_comparison(results)
    
    def _print_detailed_comparison(self, results: Dict[str, Any]):
        """Детальный вывод сравнения"""
        print("\n" + "=" * 80)
        print("DETAILED COMPARISON")
        print("=" * 80)
        
        for name, algo_results in results.items():
            if name == 'comparison':
                continue
            
            print(f"\n{name}:")
            print(f"  Cut size: {algo_results['min_cut']} "
                  f"(avg: {algo_results['avg_cut']:.1f} ± {algo_results['std_cut']:.1f})")
            print(f"  Time: {algo_results['min_time']:.4f}s "
                  f"(avg: {algo_results['avg_time']:.4f} ± {algo_results['std_time']:.4f})")
            print(f"  Balance: {algo_results['avg_balance']:.4f}")
            
            # Выводим разбиение
            partition = algo_results['partition']
            print(f"  Part sizes: {partition.part_sizes}")
        
        print("=" * 80)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Получение результатов в виде DataFrame"""
        rows = []
        
        for result in self.results:
            graph = result['graph']
            for algo_name, algo_results in result['results'].items():
                if algo_name == 'comparison':
                    continue
                
                rows.append({
                    'num_vertices': graph.num_vertices,
                    'num_edges': graph.num_edges,
                    'num_parts': result['num_parts'],
                    'algorithm': algo_name,
                    'cut_size': algo_results['min_cut'],
                    'time_seconds': algo_results['min_time'],
                    'balance_quality': algo_results['avg_balance'],
                    'part_sizes': str(algo_results['partition'].part_sizes)
                })
        
        return pd.DataFrame(rows)
    
    def export_results(self, filename: str):
        """Экспорт результатов в CSV"""
        df = self.get_results_dataframe()
        df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")
    
    def reset(self):
        """Сброс всех результатов"""
        self.results.clear()
        self.tracker.reset()


class ComparisonReport:
    """
    Генерация отчётов по сравнению алгоритмов
    """
    
    @staticmethod
    def generate_html_report(comparator: AlgorithmComparator, output_file: str):
        """
        Генерация HTML отчёта
        
        Args:
            comparator: объект AlgorithmComparator с результатами
            output_file: путь к выходному HTML файлу
        """
        df = comparator.get_results_dataframe()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Algorithm Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .best {{ background-color: #d4edda; }}
            </style>
        </head>
        <body>
            <h1>Graph Partitioning Algorithm Comparison</h1>
            
            <h2>Summary</h2>
            <p>Total comparisons: {len(comparator.results)}</p>
            <p>Algorithms tested: {', '.join(comparator.algorithms.keys())}</p>
            
            <h2>Results</h2>
            {df.to_html(index=False, classes='dataframe')}
            
            <h2>Best Results</h2>
        """
        
        # Добавляем информацию о лучших результатах
        if len(df) > 0:
            best_cut = df.loc[df.groupby('num_vertices')['cut_size'].idxmin()]
            html += best_cut.to_html(index=False)
        
        html += """
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"HTML report generated: {output_file}")