"""
Метрики для оценки качества разбиения графа
Добавлена метрика числа межсоединений (cut size)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition


class PartitionMetrics:
    """
    Класс для вычисления различных метрик качества разбиения
    
    Метрики:
    - Cut size / число межсоединений: количество рёбер между частями
    - Balance: насколько равномерно распределены вершины
    - Edge cut ratio: доля разрезанных рёбер
    - Communication volume: объём коммуникаций (для взвешенных графов)
    - Maximum part weight: максимальный вес части
    - Standard deviation of part sizes: отклонение размеров частей
    """
    
    def __init__(self, graph: Graph, partition: Partition):
        """
        Инициализация метрик
        
        Args:
            graph: граф
            partition: разбиение графа
        """
        self.graph = graph
        self.partition = partition
        self._cache = {}
    
    def cut_size(self) -> int:
        """
        Число межсоединений (количество рёбер между разными частями)
        Это основная метрика качества разбиения
        """
        if 'cut_size' not in self._cache:
            self._cache['cut_size'] = self.partition.cut_edges(self.graph)
        return self._cache['cut_size']
    
    def cut_size_detailed(self) -> Dict[Tuple[int, int], int]:
        """
        Детальная информация о межсоединениях между парами частей
        
        Returns:
            Словарь {(часть1, часть2): количество рёбер}
        """
        if 'cut_size_detailed' not in self._cache:
            cuts = defaultdict(int)
            
            for u, v, w in self.graph.edges():
                pu = self.partition.get_part(u)
                pv = self.partition.get_part(v)
                
                if pu != pv:
                    key = tuple(sorted((pu, pv)))
                    cuts[key] += w
            
            self._cache['cut_size_detailed'] = dict(cuts)
        
        return self._cache['cut_size_detailed']
    
    def cut_ratio(self) -> float:
        """
        Доля разрезанных рёбер
        Returns: отношение разрезанных рёбер к общему числу рёбер
        """
        if self.graph.num_edges == 0:
            return 0.0
        return self.cut_size() / self.graph.num_edges
    
    def cut_edges_per_vertex(self) -> float:
        """
        Среднее количество разрезов на вершину
        """
        if self.graph.num_vertices == 0:
            return 0.0
        return self.cut_size() / self.graph.num_vertices
    
    def balance_ratio(self) -> float:
        """
        Коэффициент балансировки (чем ближе к 1, тем лучше)
        Returns: отношение минимальной части к максимальной
        """
        sizes = self.partition.part_sizes
        if len(sizes) == 0 or max(sizes) == 0:
            return 0.0
        return min(sizes) / max(sizes)
    
    def balance_quality(self) -> float:
        """
        Качество балансировки (1 - идеально, 0 - плохо)
        Учитывает количество частей
        """
        sizes = self.partition.part_sizes
        target = self.graph.num_vertices / self.partition.num_parts
        
        if target == 0:
            return 0.0
        
        deviations = [abs(size - target) / target for size in sizes]
        avg_deviation = np.mean(deviations)
        
        return max(0.0, 1.0 - avg_deviation)
    
    def part_size_std(self) -> float:
        """Стандартное отклонение размеров частей"""
        return float(np.std(self.partition.part_sizes))
    
    def part_size_range(self) -> Tuple[int, int]:
        """Диапазон размеров частей (мин, макс)"""
        sizes = self.partition.part_sizes
        return (int(min(sizes)), int(max(sizes)))
    
    def max_part_weight(self) -> int:
        """Максимальный вес части"""
        if 'max_weight' not in self._cache:
            weights = self.partition.compute_part_weights(self.graph)
            self._cache['max_weight'] = int(max(weights))
        return self._cache['max_weight']
    
    def communication_volume(self) -> int:
        """
        Объём коммуникаций (сумма весов рёбер между частями)
        Важно для параллельных вычислений
        """
        if 'comm_vol' not in self._cache:
            vol = 0
            for u, v, w in self.graph.edges():
                if self.partition.get_part(u) != self.partition.get_part(v):
                    vol += w
            self._cache['comm_vol'] = vol
        return self._cache['comm_vol']
    
    def edge_cut_distribution(self) -> Dict[int, int]:
        """
        Распределение разрезов по частям
        Returns: {часть: количество разрезанных рёбер, инцидентных части}
        """
        cut_counts = {i: 0 for i in range(self.partition.num_parts)}
        
        for u, v, w in self.graph.edges():
            pu = self.partition.get_part(u)
            pv = self.partition.get_part(v)
            
            if pu != pv:
                cut_counts[pu] += w
                cut_counts[pv] += w
        
        return cut_counts
    
    def internal_edges_ratio(self) -> float:
        """
        Доля внутренних рёбер (не разрезанных)
        """
        internal = self.graph.num_edges - self.cut_size()
        return internal / self.graph.num_edges if self.graph.num_edges > 0 else 1.0
    
    def part_density(self) -> List[float]:
        """
        Плотность каждой части (отношение внутренних рёбер к максимально возможным)
        """
        densities = []
        
        for part in range(self.partition.num_parts):
            vertices = self.partition.get_vertices_in_part(part)
            n = len(vertices)
            
            if n < 2:
                densities.append(0.0)
                continue
            
            # Считаем внутренние рёбра
            internal_edges = 0
            vertex_set = set(vertices)
            
            for v in vertices:
                for u in self.graph.get_neighbors(v):
                    if u in vertex_set and v < u:
                        internal_edges += 1
            
            max_edges = n * (n - 1) / 2
            densities.append(internal_edges / max_edges if max_edges > 0 else 0.0)
        
        return densities
    
    def cut_edges_list(self) -> List[Tuple[int, int, int]]:
        """
        Список всех разрезанных рёбер
        Returns: список (u, v, вес)
        """
        cut_edges = []
        
        for u, v, w in self.graph.edges():
            if self.partition.get_part(u) != self.partition.get_part(v):
                cut_edges.append((u, v, w))
        
        return cut_edges
    
    def part_connectivity(self) -> Dict[int, List[int]]:
        """
        Связность между частями
        Returns: {часть: [список частей, с которыми соединена]}
        """
        connectivity = {i: set() for i in range(self.partition.num_parts)}
        
        for u, v, _ in self.graph.edges():
            pu = self.partition.get_part(u)
            pv = self.partition.get_part(v)
            
            if pu != pv:
                connectivity[pu].add(pv)
                connectivity[pv].add(pu)
        
        return {k: list(v) for k, v in connectivity.items()}
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Получение всех метрик"""
        
        # Принудительно исправляем разбиение
        if not self.partition.is_complete():
            self.partition.fix_unassigned(self.graph)
        
        metrics = {
            'cut_size': self.cut_size(),
            'cut_ratio': self.cut_ratio(),
            'balance_quality': self.balance_quality(),
            'part_sizes': self.partition.part_sizes.tolist(),
            'num_parts': self.partition.num_parts,
            'total_vertices': self.graph.num_vertices,
            'total_edges': self.graph.num_edges
        }
        
        return metrics

    def cut_size(self) -> int:
        return self.partition.cut_edges(self.graph)

def edge_cut_distribution(self) -> Dict[int, int]:
    """
    Распределение разрезов по частям
    """
    cut_counts = {i: 0 for i in range(self.partition.num_parts)}
    
    for u, v, w in self.graph.edges():
        pu = self.partition.get_part(u)
        pv = self.partition.get_part(v)
        
        # Если всё ещё есть -1, пропускаем (но после fix_unassigned их быть не должно)
        if pu == -1 or pv == -1:
            continue
            
        if pu != pv:
            cut_counts[pu] += w
            cut_counts[pv] += w
    
    return cut_counts
    
    def print_summary(self):
        """Вывод сводки метрик с акцентом на число межсоединений"""
        metrics = self.get_all_metrics()
        
        print("\n" + "=" * 60)
        print("PARTITION METRICS SUMMARY")
        print("=" * 60)
        print(f"Graph: {metrics['total_vertices']} vertices, {metrics['total_edges']} edges")
        print(f"Parts: {metrics['num_parts']}")
        
        print(f"\n📊 CUT METRICS (Межсоединения):")
        print(f"  ✂️  Cut size (число межсоединений): {metrics['cut_size']}")
        print(f"  📈 Cut ratio: {metrics['cut_ratio']:.4f} ({metrics['cut_ratio']*100:.2f}%)")
        print(f"  🔗 Cut edges per vertex: {metrics['cut_edges_per_vertex']:.2f}")
        print(f"  📋 Number of cut edges: {metrics['num_cut_edges']}")
        
        if metrics['cut_detailed']:
            print(f"\n  Detailed cuts between parts:")
            for (p1, p2), count in sorted(metrics['cut_detailed'].items()):
                print(f"    Part {p1} <-> Part {p2}: {count} edges")
        
        print(f"\n⚖️  BALANCE METRICS:")
        print(f"  Balance ratio: {metrics['balance_ratio']:.4f}")
        print(f"  Balance quality: {metrics['balance_quality']:.4f}")
        print(f"  Part sizes: {metrics['part_sizes']}")
        print(f"  Part size std: {metrics['part_size_std']:.2f}")
        print(f"  Size range: [{metrics['part_size_min']}, {metrics['part_size_max']}]")
        
        print(f"\n📦 OTHER METRICS:")
        print(f"  Communication volume: {metrics['communication_volume']}")
        print(f"  Internal edges ratio: {metrics['internal_edges_ratio']:.4f}")
        
        print("=" * 60)
        
        return metrics['cut_size']  # Возвращаем число межсоединений
    
    @staticmethod
    def compare_partitions(graph: Graph, partitions: List[Tuple[str, Partition]]) -> Dict[str, Any]:
        """
        Сравнение нескольких разбиений
        
        Args:
            graph: граф
            partitions: список (имя, разбиение)
        
        Returns:
            Словарь со сравнением
        """
        results = {}
        
        for name, partition in partitions:
            metrics = PartitionMetrics(graph, partition)
            results[name] = metrics.get_all_metrics()
        
        # Добавляем сравнение
        comparison = {}
        
        if len(partitions) >= 2:
            names = [n for n, _ in partitions]
            best_cut = min(results[n]['cut_size'] for n in names)
            best_balance = max(results[n]['balance_quality'] for n in names)
            
            comparison['best_cut_size'] = best_cut
            comparison['best_balance_quality'] = best_balance
            
            baseline = names[0]
            baseline_cut = results[baseline]['cut_size']
            
            for name in names[1:]:
                cut_improvement = (baseline_cut - results[name]['cut_size']) / baseline_cut * 100
                comparison[f'{name}_cut_improvement_percent'] = cut_improvement
        
        results['comparison'] = comparison
        return results