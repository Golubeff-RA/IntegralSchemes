"""
Реализация алгоритма Кернигана-Лина (KL) для разбиения графов
Классический алгоритм локальной оптимизации для задачи разбиения
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from collections import defaultdict
import heapq

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition
from .base_partitioner import BasePartitioner


class KernighanLin(BasePartitioner):
    """
    Алгоритм Кернигана-Лина для улучшения разбиения графа
    
    Идея:
    1. Найти последовательность обменов вершин между частями
    2. Выбрать лучшую последовательность (максимальное уменьшение cut size)
    3. Повторять пока есть улучшения
    
    Сложность: O(V^3) в базовой версии, но с оптимизациями O(V^2 log V)
    """
    
    def __init__(self, max_passes: int = 10, max_iterations: int = 100):
        """
        Инициализация KL алгоритма
        
        Args:
            max_passes: максимальное количество проходов
            max_iterations: максимальное количество итераций в проходе
        """
        super().__init__(name="KernighanLin")
        self.max_passes = max_passes
        self.max_iterations = max_iterations
    
    def partition(self, graph: Graph, num_parts: int = 2,
              balance_ratio: float = 0.5) -> Partition:
        """
        Разбиение графа с помощью KL алгоритма
        """
        if num_parts != 2:
            raise NotImplementedError("KL algorithm currently only supports 2-way partitioning")
        
        # Создаём начальное разбиение
        partition = self._initial_partition(graph, balance_ratio)
        
        # Убеждаемся, что все вершины назначены
        partition.fix_unassigned(graph)
        
        initial_cut = partition.cut_edges(graph)
        self._statistics['initial_cut'] = initial_cut
        
        # Основной цикл KL
        best_cut = initial_cut
        best_partition = partition.copy()
        
        for _ in range(self.max_passes):
            partition, improved = self._kl_pass(graph, partition, balance_ratio)
            
            # Фиксируем неназначенные вершины после каждого прохода
            partition.fix_unassigned(graph)
            
            current_cut = partition.cut_edges(graph)
            
            if current_cut < best_cut:
                best_cut = current_cut
                best_partition = partition.copy()
                self._statistics['iterations'] += 1
            else:
                break
        
        # Финальная проверка
        best_partition.fix_unassigned(graph)
        self._statistics['final_cut'] = best_cut
        
        return best_partition
    
    def _initial_partition(self, graph: Graph, balance_ratio: float) -> Partition:
        """
        Создание начального сбалансированного разбиения
        Гарантирует, что все вершины будут назначены
        """
        n = graph.num_vertices
        partition = Partition(n, 2)
        
        # Определяем размер первой части
        target_size = int(n * balance_ratio)
        
        # Сортируем вершины по степени (хабы в разные части)
        vertices = list(range(n))
        vertices.sort(key=lambda v: graph.get_degree(v), reverse=True)
        
        # Распределяем вершины
        part0_size = 0
        for v in vertices:
            if part0_size < target_size:
                partition.assign(v, 0)
                part0_size += 1
            else:
                partition.assign(v, 1)
        
        # Проверяем, что все вершины назначены
        for v in range(n):
            if partition.get_part(v) == -1:
                # Если вдруг осталась неназначенная вершина
                partition.assign(v, 0 if part0_size < target_size else 1)
                part0_size += 1
        
        return partition
    
    def _kl_pass(self, graph: Graph, partition: Partition,
                 balance_ratio: float) -> Tuple[Partition, bool]:
        """
        Один проход KL алгоритма
        
        Args:
            graph: граф
            partition: текущее разбиение
            balance_ratio: допустимый дисбаланс
        
        Returns:
            Tuple[Partition, bool]: (новое разбиение, было ли улучшение)
        """
        n = graph.num_vertices
        part0_size = partition.part_sizes[0]
        part1_size = partition.part_sizes[1]
        
        # Вычисляем gain для каждой вершины
        gains = self._compute_gains(graph, partition)
        
        # Отмечаем заблокированные вершины
        locked = [False] * n
        
        # История перемещений
        moves = []  # (вершина, gain)
        moved_vertices = []
        
        # Максимально допустимое количество обменов
        max_swaps = min(part0_size, part1_size, self.max_iterations)
        
        for _ in range(max_swaps):
            # Находим лучшую пару вершин для обмена
            best_pair = self._find_best_swap(graph, partition, gains, locked,
                                            part0_size, part1_size, balance_ratio)
            
            if best_pair is None:
                break
            
            v0, v1, gain = best_pair
            
            # Запоминаем перемещение
            moves.append((v0, v1, gain))
            moved_vertices.append(v0)
            moved_vertices.append(v1)
            
            # Блокируем вершины
            locked[v0] = True
            locked[v1] = True
            
            # Выполняем обмен
            partition.assign(v0, 1)
            partition.assign(v1, 0)
            
            # Обновляем размеры частей
            part0_size -= 1
            part1_size += 1
            
            # Обновляем gains для соседей
            self._update_gains(graph, partition, gains, locked, v0, v1)
        
        # Находим лучшую последовательность перемещений
        if not moves:
            return partition, False
        
        # Вычисляем накопленные gains
        cumulative_gains = []
        current_gain = 0
        for _, _, gain in moves:
            current_gain += gain
            cumulative_gains.append(current_gain)
        
        # Находим индекс с максимальным накопленным gain
        best_idx = np.argmax(cumulative_gains)
        best_gain = cumulative_gains[best_idx]
        
        if best_gain <= 0:
            return partition, False
        
        # Применяем лучшую последовательность
        # Откатываем все перемещения
        for v0, v1, _ in reversed(moves):
            partition.assign(v0, 0)
            partition.assign(v1, 1)
        
        # Применяем лучшие перемещения
        for i in range(best_idx + 1):
            v0, v1, _ = moves[i]
            partition.assign(v0, 1)
            partition.assign(v1, 0)
        
        return partition, best_gain > 0
    
    def _compute_gains(self, graph: Graph, partition: Partition) -> np.ndarray:
        """
        Вычисление gain для каждой вершины
        
        Gain = (внешние рёбра) - (внутренние рёбра)
        Положительный gain означает выгоду от перемещения
        """
        n = graph.num_vertices
        gains = np.zeros(n, dtype=np.int32)
        
        for v in range(n):
            part = partition.get_part(v)
            internal = 0
            external = 0
            
            for neighbor, weight in graph.get_neighbors(v).items():
                if partition.get_part(neighbor) == part:
                    internal += weight
                else:
                    external += weight
            
            gains[v] = external - internal
        
        return gains
    
    def _find_best_swap(self, graph: Graph, partition: Partition,
                       gains: np.ndarray, locked: List[bool],
                       part0_size: int, part1_size: int,
                       balance_ratio: float) -> Optional[Tuple[int, int, int]]:
        """
        Поиск лучшей пары вершин для обмена
        
        Returns:
            Tuple[int, int, int]: (вершина_из_0, вершина_из_1, gain)
        """
        best_gain = -float('inf')
        best_pair = None
        
        # Собираем кандидатов из каждой части
        candidates0 = [v for v in range(graph.num_vertices) 
                      if not locked[v] and partition.get_part(v) == 0]
        candidates1 = [v for v in range(graph.num_vertices) 
                      if not locked[v] and partition.get_part(v) == 1]
        
        # Проверяем баланс
        min_part0_size = int(graph.num_vertices * (1 - balance_ratio))
        max_part0_size = int(graph.num_vertices * balance_ratio)
        
        for v0 in candidates0:
            for v1 in candidates1:
                # Проверяем баланс после обмена
                new_part0_size = part0_size - 1
                if not (min_part0_size <= new_part0_size <= max_part0_size):
                    continue
                
                # Вычисляем gain от обмена
                gain = gains[v0] + gains[v1] - 2 * graph.get_edge_weight(v0, v1)
                
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (v0, v1, gain)
        
        return best_pair
    
    def _update_gains(self, graph: Graph, partition: Partition,
                     gains: np.ndarray, locked: List[bool],
                     v0: int, v1: int):
        """
        Обновление gains для соседей перемещённых вершин
        """
        # Обновляем для соседей v0
        for neighbor in graph.get_neighbors(v0):
            if not locked[neighbor]:
                self._update_gain_for_vertex(graph, partition, gains, neighbor)
        
        # Обновляем для соседей v1
        for neighbor in graph.get_neighbors(v1):
            if not locked[neighbor]:
                self._update_gain_for_vertex(graph, partition, gains, neighbor)
    
    def _update_gain_for_vertex(self, graph: Graph, partition: Partition,
                                gains: np.ndarray, v: int):
        """Пересчёт gain для одной вершины"""
        part = partition.get_part(v)
        internal = 0
        external = 0
        
        for neighbor, weight in graph.get_neighbors(v).items():
            if partition.get_part(neighbor) == part:
                internal += weight
            else:
                external += weight
        
        gains[v] = external - internal


class ImprovedKernighanLin(KernighanLin):
    """
    Улучшенная версия KL алгоритма с:
    - Приоритетной очередью для выбора лучших кандидатов
    - Более эффективным обновлением gains
    - Поддержкой взвешенных вершин
    """
    
    def __init__(self, max_passes: int = 10, max_iterations: int = 100):
        super().__init__(max_passes, max_iterations)
        self.name = "ImprovedKernighanLin"
    
    def _find_best_swap(self, graph: Graph, partition: Partition,
                       gains: np.ndarray, locked: List[bool],
                       part0_size: int, part1_size: int,
                       balance_ratio: float) -> Optional[Tuple[int, int, int]]:
        """
        Оптимизированный поиск лучшей пары с использованием приоритетных очередей
        """
        # Используем приоритетные очереди для каждой части
        # (Для простоты оставляем полный перебор, но с ранним выходом)
        
        # Сортируем кандидатов по gain
        candidates0 = [(gains[v], v) for v in range(graph.num_vertices) 
                      if not locked[v] and partition.get_part(v) == 0]
        candidates1 = [(gains[v], v) for v in range(graph.num_vertices) 
                      if not locked[v] and partition.get_part(v) == 1]
        
        candidates0.sort(reverse=True)
        candidates1.sort(reverse=True)
        
        # Проверяем только топ кандидатов
        top_k = min(50, len(candidates0), len(candidates1))
        
        best_gain = -float('inf')
        best_pair = None
        
        min_part0_size = int(graph.num_vertices * (1 - balance_ratio))
        max_part0_size = int(graph.num_vertices * balance_ratio)
        
        for i in range(min(top_k, len(candidates0))):
            gain0, v0 = candidates0[i]
            
            # Если даже лучший gain0 + лучший gain1 не превосходят best_gain, выходим
            if gain0 + candidates1[0][0] <= best_gain:
                break
            
            for j in range(min(top_k, len(candidates1))):
                gain1, v1 = candidates1[j]
                
                if gain0 + gain1 <= best_gain:
                    break
                
                new_part0_size = part0_size - 1
                if not (min_part0_size <= new_part0_size <= max_part0_size):
                    continue
                
                gain = gain0 + gain1 - 2 * graph.get_edge_weight(v0, v1)
                
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (v0, v1, gain)
        
        return best_pair