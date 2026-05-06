"""
Алгоритм Кернигана-Лина (KL) для бисекции графов

Классический алгоритм локальной оптимизации:
1. Вычисляем gain для каждой вершины (экономия от перемещения)
2. Выбираем лучшую пару вершин для обмена
3. Повторяем до улучшения
"""

import random
import heapq
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition
from .base_partitioner import PartitionerWithStats, PerformanceMetrics


@dataclass
class VertexGain:
    """Вершина с её gain для приоритетной очереди"""
    vertex: int
    gain: int
    
    def __lt__(self, other):
        return self.gain > other.gain  # Чем больше gain, тем выше приоритет


class KernighanLin(PartitionerWithStats):
    """
    Алгоритм Кернигана-Лина для бисекции графов
    
    Сложность: O(V² log V) с оптимизациями
    """
    
    def __init__(self, max_passes: int = 10, max_iterations: int = 100, seed: int = 42):
        """
        Args:
            max_passes: максимальное количество проходов
            max_iterations: максимальное количество итераций в проходе
            seed: seed для воспроизводимости
        """
        super().__init__(name="Kernighan-Lin")
        self.max_passes = max_passes
        self.max_iterations = max_iterations
        self.seed = seed
        random.seed(seed)
    
    def _partition_impl(self, graph: Graph, balance_ratio: float = 0.5) -> Partition:
        """
        Реализация KL алгоритма
        
        Этапы:
        1. Создание начального сбалансированного разбиения
        2. Основной цикл улучшения
        3. Применение лучшей последовательности обменов
        """
        n = graph.num_vertices
        
        # 1. Начальное разбиение
        partition = self._initial_partition(graph, balance_ratio)
        best_partition = partition.copy()
        best_cut = partition.cut_weight(graph)
        
        self._record_iteration(best_cut)
        
        # 2. Основной цикл
        for _ in range(self.max_passes):
            partition, improved = self._kl_pass(graph, partition, balance_ratio)
            current_cut = partition.cut_weight(graph)
            
            if current_cut < best_cut:
                best_cut = current_cut
                best_partition = partition.copy()
                self._record_iteration(best_cut)
            
            if not improved:
                break
        
        return best_partition
    
    def _initial_partition(self, graph: Graph, balance_ratio: float) -> Partition:
        """
        Создание начального сбалансированного разбиения
        
        Стратегия: сортируем вершины по степени, распределяем хабы равномерно
        """
        n = graph.num_vertices
        partition = Partition(n)
        
        # Целевой размер первой части
        target_size = int(n * balance_ratio)
        
        # Сортируем вершины по степени (убывание)
        vertices = list(range(n))
        vertices.sort(key=lambda v: graph.get_degree(v), reverse=True)
        
        # Распределяем
        part0_size = 0
        for v in vertices:
            if part0_size < target_size:
                partition.assign(v, 0)
                part0_size += 1
            else:
                partition.assign(v, 1)
        
        # Обновляем веса частей
        partition.update_weights(graph)
        
        return partition
    
    def _kl_pass(self, graph: Graph, partition: Partition, balance_ratio: float) -> Tuple[Partition, bool]:
        """
        Один проход KL алгоритма
        """
        n = graph.num_vertices
        
        # Запоминаем начальные размеры частей
        size0 = partition.size0
        size1 = partition.size1
        
        # Вычисляем начальные gain
        gains = self._compute_all_gains(graph, partition)
        
        # Заблокированные вершины
        locked = [False] * n
        
        # История обменов
        swaps = []  # (v0, v1, delta)
        
        max_swaps = min(size0, size1, self.max_iterations)
        
        for _ in range(max_swaps):
            # Находим лучшую пару для обмена
            best_pair = self._find_best_swap(graph, partition, gains, locked, size0, size1, balance_ratio)
            
            if best_pair is None:
                break
            
            v0, v1, delta = best_pair
            
            # Убеждаемся, что v0 в part0, v1 в part1
            if partition.get_part(v0) != 0 or partition.get_part(v1) != 1:
                # Пробуем поменять местами
                if partition.get_part(v0) == 1 and partition.get_part(v1) == 0:
                    v0, v1 = v1, v0
                else:
                    # Неправильные части - пропускаем
                    continue
            
            # Запоминаем обмен
            swaps.append((v0, v1, delta))
            
            # Блокируем вершины
            locked[v0] = True
            locked[v1] = True
            
            # Выполняем обмен
            partition.swap_vertices(v0, v1, graph)
            
            # Обновляем gain для соседей
            self._update_gains(graph, partition, gains, locked, v0, v1)
        
        # Находим лучшую последовательность обменов
        if not swaps:
            return partition, False
        
        # Вычисляем накопленные delta
        cumulative = []
        current = 0
        for _, _, delta in swaps:
            current += delta
            cumulative.append(current)
        
        # Индекс с максимальным улучшением
        best_idx = max(range(len(cumulative)), key=lambda i: cumulative[i])
        best_delta = cumulative[best_idx]
        
        if best_delta <= 0:
            return partition, False
        
        # Откатываем все обмены
        for v0, v1, _ in reversed(swaps):
            # Убеждаемся, что v0 в part1, v1 в part0 (после обменов)
            if partition.get_part(v0) == 1 and partition.get_part(v1) == 0:
                partition.swap_vertices(v0, v1, graph)
            else:
                partition.swap_vertices(v1, v0, graph)
        
        # Применяем лучшую последовательность
        for i in range(best_idx + 1):
            v0, v1, _ = swaps[i]
            # Убеждаемся, что v0 в part0, v1 в part1
            if partition.get_part(v0) == 0 and partition.get_part(v1) == 1:
                partition.swap_vertices(v0, v1, graph)
            else:
                partition.swap_vertices(v1, v0, graph)
        
        return partition, best_delta > 0

    def _find_best_swap(self, graph: Graph, partition: Partition, gains: List[int],
                        locked: List[bool], size0: int, size1: int,
                        balance_ratio: float) -> Optional[Tuple[int, int, int]]:
        """
        Поиск лучшей пары вершин для обмена
        """
        n = graph.num_vertices
        
        # Собираем кандидатов (убеждаемся, что в правильных частях)
        candidates0 = []
        candidates1 = []
        
        for v in range(n):
            if locked[v]:
                continue
            part = partition.get_part(v)
            if part == 0:
                candidates0.append(v)
            elif part == 1:
                candidates1.append(v)
        
        if not candidates0 or not candidates1:
            return None
        
        # Проверка баланса (при обмене размеры не меняются)
        min_size0 = int(n * (1 - balance_ratio))
        max_size0 = int(n * balance_ratio)
        
        if not (min_size0 <= size0 <= max_size0):
            return None
        
        # Ищем лучшую пару
        best_gain = -float('inf')
        best_pair = None
        
        # Оптимизация: рассматриваем только топ-кандидатов
        top_k = min(50, len(candidates0), len(candidates1))
        
        # Сортируем по gain
        candidates0.sort(key=lambda v: gains[v], reverse=True)
        candidates1.sort(key=lambda v: gains[v], reverse=True)
        
        for i in range(min(top_k, len(candidates0))):
            for j in range(min(top_k, len(candidates1))):
                v0 = candidates0[i]
                v1 = candidates1[j]
                
                # Gain от обмена
                edge_w = graph.get_edge_weight(v0, v1)
                delta = gains[v0] + gains[v1] - 2 * edge_w
                
                if delta > best_gain:
                    best_gain = delta
                    best_pair = (v0, v1, delta)
        
        return best_pair
    
    def _compute_all_gains(self, graph: Graph, partition: Partition) -> List[int]:
        """
        Вычисление gain для всех вершин
        
        Gain = (внешний вес) - (внутренний вес)
        Положительный gain означает выгоду от перемещения
        """
        n = graph.num_vertices
        gains = [0] * n
        
        for v in range(n):
            gains[v] = self._compute_gain(graph, partition, v)
        
        return gains
    
    def _compute_gain(self, graph: Graph, partition: Partition, vertex: int) -> int:
        """Вычисление gain для одной вершины"""
        part = partition.get_part(vertex)
        if part == -1:
            return 0
        
        internal = 0  # Вес рёбер внутри части
        external = 0  # Вес рёбер к другой части
        
        for neighbor, weight in graph.get_neighbors(vertex):
            neighbor_part = partition.get_part(neighbor)
            if neighbor_part == part:
                internal += weight
            elif neighbor_part != -1:
                external += weight
        
        return external - internal
    
    def _find_best_swap(self, graph: Graph, partition: Partition, gains: List[int],
                        locked: List[bool], size0: int, size1: int,
                        balance_ratio: float) -> Optional[Tuple[int, int, int]]:
        """
        Поиск лучшей пары вершин для обмена
        
        Returns:
            Tuple[int, int, int]: (вершина_из_0, вершина_из_1, delta) или None
        """
        n = graph.num_vertices
        
        # Собираем кандидатов
        candidates0 = [v for v in range(n) if not locked[v] and partition.get_part(v) == 0]
        candidates1 = [v for v in range(n) if not locked[v] and partition.get_part(v) == 1]
        
        if not candidates0 or not candidates1:
            return None
        
        # Проверка баланса (при обмене размеры не меняются)
        min_size0 = int(n * (1 - balance_ratio))
        max_size0 = int(n * balance_ratio)
        
        if not (min_size0 <= size0 <= max_size0):
            return None
        
        # Ищем лучшую пару
        best_gain = -float('inf')
        best_pair = None
        
        # Оптимизация: рассматриваем только топ-кандидатов
        top_k = min(50, len(candidates0), len(candidates1))
        
        # Сортируем по gain
        candidates0.sort(key=lambda v: gains[v], reverse=True)
        candidates1.sort(key=lambda v: gains[v], reverse=True)
        
        for i in range(min(top_k, len(candidates0))):
            for j in range(min(top_k, len(candidates1))):
                v0 = candidates0[i]
                v1 = candidates1[j]
                
                # Gain от обмена
                edge_w = graph.get_edge_weight(v0, v1)
                delta = gains[v0] + gains[v1] - 2 * edge_w
                
                if delta > best_gain:
                    best_gain = delta
                    best_pair = (v0, v1, delta)
        
        return best_pair
    
    def _update_gains(self, graph: Graph, partition: Partition, gains: List[int],
                      locked: List[bool], v0: int, v1: int) -> None:
        """
        Обновление gain после обмена вершин
        """
        # Обновляем для соседей v0
        for neighbor, _ in graph.get_neighbors(v0):
            if not locked[neighbor]:
                gains[neighbor] = self._compute_gain(graph, partition, neighbor)
        
        # Обновляем для соседей v1
        for neighbor, _ in graph.get_neighbors(v1):
            if not locked[neighbor]:
                gains[neighbor] = self._compute_gain(graph, partition, neighbor)
        
        # Обновляем для самих вершин
        gains[v0] = self._compute_gain(graph, partition, v0)
        gains[v1] = self._compute_gain(graph, partition, v1)


class FastKernighanLin(KernighanLin):
    """
    Ускоренная версия KL с использованием приоритетных очередей
    
    Сложность: O(V log V) на итерацию
    """
    
    def __init__(self, max_passes: int = 10, max_iterations: int = 100, seed: int = 42):
        super().__init__(max_passes, max_iterations, seed)
        self.name = "FastKernighanLin"
    
    def _find_best_swap(self, graph: Graph, partition: Partition, gains: List[int],
                        locked: List[bool], size0: int, size1: int,
                        balance_ratio: float) -> Optional[Tuple[int, int, int]]:
        """
        Ускоренный поиск с использованием приоритетной очереди
        """
        n = graph.num_vertices
        
        # Создаём приоритетные очереди для каждой части
        heap0 = []
        heap1 = []
        
        for v in range(n):
            if locked[v]:
                continue
            part = partition.get_part(v)
            if part == 0:
                heapq.heappush(heap0, VertexGain(v, gains[v]))
            elif part == 1:
                heapq.heappush(heap1, VertexGain(v, gains[v]))
        
        if not heap0 or not heap1:
            return None
        
        # Проверка баланса
        min_size0 = int(n * (1 - balance_ratio))
        max_size0 = int(n * balance_ratio)
        
        if not (min_size0 <= size0 <= max_size0):
            return None
        
        # Берём лучшие вершины
        best0 = heapq.heappop(heap0)
        best1 = heapq.heappop(heap1)
        
        # Проверяем, актуальны ли значения gain
        if gains[best0.vertex] != best0.gain:
            return self._find_best_swap(graph, partition, gains, locked, size0, size1, balance_ratio)
        
        if gains[best1.vertex] != best1.gain:
            return self._find_best_swap(graph, partition, gains, locked, size0, size1, balance_ratio)
        
        edge_w = graph.get_edge_weight(best0.vertex, best1.vertex)
        delta = best0.gain + best1.gain - 2 * edge_w
        
        return (best0.vertex, best1.vertex, delta)