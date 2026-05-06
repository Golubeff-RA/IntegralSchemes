"""
Многоуровневое разбиение графа - ОПТИМИЗИРОВАННАЯ ВЕРСИЯ
"""

import random
import time
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition
from core.coarse_graph import CoarseGraph
from .base_partitioner import PartitionerWithStats
from .kernighan_lin import KernighanLin


@dataclass
class CoarseningLevel:
    level: int
    graph: Graph
    coarse_graph: CoarseGraph
    vertex_map: Dict[int, int]
    reverse_map: Dict[int, List[int]]
    compression_ratio: float = 1.0


class FastMultilevelPartitioner(PartitionerWithStats):
    """
    Быстрый многоуровневый алгоритм разбиения
    
    Оптимизации:
    1. Уменьшено количество проходов улучшения
    2. Используется быстрый KL только на маленьких графах
    3. Оптимизирована проекция через прямой маппинг
    """
    
    def __init__(self, 
                 min_coarse_vertices: int = 100,
                 max_levels: int = 8,
                 refinement_passes: int = 1,  # Уменьшено с 3 до 1
                 seed: int = 42):
        super().__init__(name="FastMultilevel")
        self.min_coarse_vertices = min_coarse_vertices
        self.max_levels = max_levels
        self.refinement_passes = refinement_passes
        self.seed = seed
        random.seed(seed)
        self.levels: List[CoarseningLevel] = []
    
    def _partition_impl(self, graph: Graph, balance_ratio: float = 0.5) -> Partition:
        """Быстрое многоуровневое разбиение"""
        
        # 1. Coarsening
        self._coarsen_fast(graph)
        
        if not self.levels:
            # Если стягивание не удалось, используем KL
            kl = KernighanLin(seed=self.seed)
            partition, _ = kl.partition(graph, balance_ratio)
            return partition
        
        # 2. Initial partitioning на самом грубом графе
        partition = self._initial_partition_fast(balance_ratio)
        
        # 3. Uncoarsening - прямая проекция без лишних улучшений
        partition = self._uncoarsen_fast(partition)
        
        return partition
    
    def _coarsen_fast(self, graph: Graph) -> None:
        """Быстрое стягивание - один проход"""
        self.levels = []
        current = graph
        
        for level_num in range(1, self.max_levels + 1):
            if current.num_vertices <= self.min_coarse_vertices:
                break
            
            # Находим паросочетание (только лучшие рёбра)
            matching = self._fast_matching(current)
            
            if not matching or len(matching) < current.num_vertices // 20:
                break
            
            # Стягиваем граф
            coarse_graph = CoarseGraph.from_matching(current, matching)
            compression = coarse_graph.num_vertices / current.num_vertices
            
            self.levels.append(CoarseningLevel(
                level=level_num,
                graph=current,
                coarse_graph=coarse_graph,
                vertex_map={v: cv for v, cv in coarse_graph._original_to_coarse.items()},
                reverse_map=coarse_graph._coarse_to_original.copy(),
                compression_ratio=compression
            ))
            
            current = coarse_graph.to_graph()
            
            if compression > 0.85:
                break
    
    def _fast_matching(self, graph: Graph) -> List[Tuple[int, int]]:
        """
        Очень быстрое паросочетание - O(E)
        Используем жадный алгоритм без сортировки
        """
        used = set()
        matching = []
        
        # Проходим по всем вершинам
        for v in range(graph.num_vertices):
            if v in used:
                continue
            
            # Берём первого попавшегося соседа
            neighbors = list(graph.get_neighbors(v))
            if neighbors:
                # Выбираем соседа с максимальным весом (без полной сортировки)
                best = max(neighbors, key=lambda n: graph.get_edge_weight(v, n))
                if best not in used:
                    matching.append((v, best))
                    used.add(v)
                    used.add(best)
        
        return matching
    
    def _initial_partition_fast(self, balance_ratio: float) -> Partition:
        """Быстрое начальное разбиение - одна попытка"""
        coarse_graph = self.levels[-1].coarse_graph.to_graph()
        partition = self._random_partition_fast(coarse_graph, balance_ratio)
        
        # Только один проход KL на грубом графе
        kl = KernighanLin(max_passes=5, seed=self.seed)
        partition, _ = kl.partition(coarse_graph, balance_ratio)
        
        return partition
    
    def _random_partition_fast(self, graph: Graph, balance_ratio: float) -> Partition:
        """Очень быстрое случайное разбиение"""
        n = graph.num_vertices
        partition = Partition(n)
        
        target = int(n * balance_ratio)
        
        # Простое чередование для скорости
        for i in range(n):
            if i < target:
                partition.assign(i, 0)
            else:
                partition.assign(i, 1)
        
        partition.update_weights(graph)
        return partition
    
    def _uncoarsen_fast(self, partition: Partition) -> Partition:
        """
        Быстрая проекция - прямая, без улучшений на каждом уровне
        """
        current_partition = partition
        
        for level in reversed(self.levels):
            # Прямая проекция
            current_partition = self._project_fast(current_partition, level)
        
        return current_partition
    
    def _project_fast(self, partition: Partition, level: CoarseningLevel) -> Partition:
        """
        Быстрая проекция через маппинг
        """
        coarse_graph = level.coarse_graph
        num_fine = level.graph.num_vertices
        
        fine_partition = Partition(num_fine)
        
        # Прямой маппинг для скорости
        for fine_v in range(num_fine):
            coarse_v = coarse_graph.get_coarse_vertex(fine_v)
            if coarse_v != -1 and coarse_v < partition.num_vertices:
                part = partition.get_part(coarse_v)
                if part != -1:
                    fine_partition.assign(fine_v, part)
        
        # Быстрое назначение оставшихся
        for v in range(num_fine):
            if fine_partition.get_part(v) == -1:
                # Назначаем в часть с меньшим размером
                if fine_partition.size0 <= fine_partition.size1:
                    fine_partition.assign(v, 0)
                else:
                    fine_partition.assign(v, 1)
        
        return fine_partition


class UltraFastMultilevelPartitioner(FastMultilevelPartitioner):
    """
    Ультра-быстрый многоуровневый алгоритм для больших графов
    """
    
    def __init__(self, seed: int = 42):
        super().__init__(
            min_coarse_vertices=200,
            max_levels=5,
            refinement_passes=0,  # Без улучшений
            seed=seed
        )
        self.name = "UltraFastMultilevel"
    
    def _fast_matching(self, graph: Graph) -> List[Tuple[int, int]]:
        """Максимально быстрое паросочетание"""
        used = set()
        matching = []
        
        # Берём каждую вторую вершину
        for v in range(0, graph.num_vertices, 2):
            if v in used:
                continue
            
            neighbors = graph.get_neighbors(v)
            if neighbors:
                # Берём первого соседа
                for n in neighbors:
                    if n not in used:
                        matching.append((v, n))
                        used.add(v)
                        used.add(n)
                        break
        
        return matching
    
    def _initial_partition_fast(self, balance_ratio: float) -> Partition:
        """Максимально быстрое начальное разбиение без KL"""
        coarse_graph = self.levels[-1].coarse_graph.to_graph()
        n = coarse_graph.num_vertices
        partition = Partition(n)
        
        # Простое разбиение пополам
        half = n // 2
        for v in range(n):
            partition.assign(v, 0 if v < half else 1)
        
        partition.update_weights(coarse_graph)
        return partition


class AdaptiveFastMultilevelPartitioner:
    """
    Адаптивный выбор стратегии в зависимости от размера графа
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
    
    def partition(self, graph: Graph, balance_ratio: float = 0.5) -> Tuple[Partition, Any]:
        """Автоматически выбирает оптимальную стратегию"""
        n = graph.num_vertices
        
        if n < 500:
            # Маленький граф - используем KL
            kl = KernighanLin(max_passes=20, seed=self.seed)
            return kl.partition(graph, balance_ratio)
        elif n < 5000:
            # Средний граф - быстрый многоуровневый с 1 проходом
            ml = FastMultilevelPartitioner(refinement_passes=1, seed=self.seed)
            return ml.partition(graph, balance_ratio)
        else:
            # Большой граф - ультра-быстрый многоуровневый
            ml = UltraFastMultilevelPartitioner(seed=self.seed)
            return ml.partition(graph, balance_ratio)