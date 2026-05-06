"""
Многоуровневое разбиение графа (Multilevel Partitioning)

Трёхэтапный алгоритм:
1. Coarsening (стягивание) - уменьшение графа путём объединения вершин
2. Initial Partitioning - разбиение на самом грубом уровне
3. Uncoarsening (проекция) + Refinement - обратная проекция с локальным улучшением
"""

import random
import time
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition
from core.coarse_graph import CoarseGraph
from .base_partitioner import PartitionerWithStats, PerformanceMetrics
from .kernighan_lin import KernighanLin


@dataclass
class CoarseningLevel:
    """Уровень стягивания"""

    level: int
    graph: Graph
    coarse_graph: CoarseGraph
    vertex_map: Dict[int, int]  # исходная -> грубая
    reverse_map: Dict[int, List[int]]  # грубая -> [исходные]
    compression_ratio: float = 1.0


class MultilevelPartitioner(PartitionerWithStats):
    """
    Многоуровневый алгоритм разбиения графа

    Этапы:
    1. Coarsening: многократное стягивание графа до < 100 вершин
    2. Initial partitioning: разбиение грубого графа (KL algorithm)
    3. Uncoarsening: обратная проекция с улучшением на каждом уровне
    """

    def __init__(
        self,
        min_coarse_vertices: int = 50,
        max_levels: int = 10,
        coarsening_ratio: float = 0.5,
        refinement_passes: int = 3,
        seed: int = 42,
    ):
        """
        Args:
            min_coarse_vertices: минимальное количество вершин в грубом графе
            max_levels: максимальное количество уровней стягивания
            coarsening_ratio: целевое соотношение размеров соседних уровней
            refinement_passes: количество проходов улучшения на каждом уровне
            seed: seed для воспроизводимости
        """
        super().__init__(name="Multilevel")
        self.min_coarse_vertices = min_coarse_vertices
        self.max_levels = max_levels
        self.coarsening_ratio = coarsening_ratio
        self.refinement_passes = refinement_passes
        self.seed = seed

        random.seed(seed)

        # История уровней для визуализации
        self.levels: List[CoarseningLevel] = []

    def _partition_impl(self, graph: Graph, balance_ratio: float = 0.5) -> Partition:
        """
        Многоуровневое разбиение графа
        """
        # 1. Coarsening - стягивание графа
        self._reset_stats()
        with self._stage("coarsening"):
            self._coarsen(graph)

        if not self.levels:
            # Если стягивание не удалось, используем KL на исходном графе
            with self._stage("initial_partition"):
                kl = KernighanLin(seed=self.seed)
                partition, _ = kl.partition(graph, balance_ratio)
            return partition

        # 2. Initial partitioning - разбиение грубого графа
        with self._stage("initial_partition"):
            partition = self._initial_partition(balance_ratio)

        # 3. Uncoarsening + refinement
        with self._stage("uncoarsening"):
            partition = self._uncoarsen_and_refine(partition, balance_ratio)

        return partition

    def _coarsen(self, graph: Graph) -> None:
        """
        Этап стягивания графа
        """
        self.levels = []
        current = graph
        level_num = 0
        
        print(f"\n  📉 Coarsening phase:")
        print(f"     Level 0: {current.num_vertices} vertices, {current.num_edges} edges")
        
        for level_num in range(1, self.max_levels + 1):
            if current.num_vertices <= self.min_coarse_vertices:
                print(f"     ✓ Stopped: reached minimum vertices ({current.num_vertices} <= {self.min_coarse_vertices})")
                break
            
            # Находим паросочетание для стягивания
            matching = self._find_matching(current)
            
            if not matching:
                print(f"     ✗ No matching found at level {level_num}")
                break
            
            # Стягиваем граф
            coarse_graph = CoarseGraph.from_matching(current, matching)
            compression = coarse_graph.num_vertices / current.num_vertices
            
            # Сохраняем уровень (сохраняем исходный граф, а не to_graph)
            level = CoarseningLevel(
                level=level_num,
                graph=current,  # Сохраняем исходный граф для этого уровня
                coarse_graph=coarse_graph,
                vertex_map={v: cv for v, cv in coarse_graph._original_to_coarse.items()},
                reverse_map=coarse_graph._coarse_to_original.copy(),
                compression_ratio=compression
            )
            self.levels.append(level)
            
            print(f"     Level {level_num}: {coarse_graph.num_vertices} vertices (compression: {compression:.3f})")
            
            # Создаём граф для следующего уровня - используем to_graph() только здесь
            current = coarse_graph.to_graph()
            
            # Если сжатие слишком слабое, останавливаемся
            if compression > 0.9:
                print(f"     ✓ Stopped: weak compression ({compression:.3f} > 0.9)")
                break
        
        # Запоминаем самый грубый граф (последний current)
        self.coarsest_graph = current

    def _find_matching(self, graph: Graph) -> List[Tuple[int, int]]:
        """
        Нахождение паросочетания для стягивания

        Стратегия: Heavy Edge Matching - предпочитаем рёбра с большим весом
        """
        used = set()
        matching = []

        # Получаем все рёбра и сортируем по весу (убывание)
        edges = list(graph.edges())
        edges.sort(key=lambda x: x[2], reverse=True)

        for u, v, w in edges[: min(len(edges), graph.num_vertices * 3)]:
            if u not in used and v not in used:
                matching.append((u, v))
                used.add(u)
                used.add(v)

        return matching

    def _initial_partition(self, balance_ratio: float) -> Partition:
        """
        Разбиение самого грубого графа
        """
        coarse_graph = self.coarsest_graph
        best_partition = None
        best_cut = float("inf")

        # Пробуем несколько различных начальных разбиений
        for trial in range(5):
            # Случайное сбалансированное разбиение
            partition = self._random_partition(coarse_graph, balance_ratio, trial)

            # Улучшаем KL
            kl = KernighanLin(max_passes=10, seed=self.seed + trial)
            partition, _ = kl.partition(coarse_graph, balance_ratio)

            cut = partition.cut_weight(coarse_graph)

            if cut < best_cut:
                best_cut = cut
                best_partition = partition

        print(f"     Initial cut on coarse graph: {best_cut}")
        self._record_iteration(best_cut)

        return best_partition

    def _random_partition(self, graph: Graph, balance_ratio: float, seed: int) -> Partition:
        """Создание случайного сбалансированного разбиения"""
        random.seed(seed)
        n = graph.num_vertices
        partition = Partition(n)

        target_size = int(n * balance_ratio)

        # Создаём перемешанный список вершин
        vertices = list(range(n))
        random.shuffle(vertices)

        # Назначаем первые target_size вершин в часть 0
        for i, v in enumerate(vertices):
            if i < target_size:
                partition.assign(v, 0)
            else:
                partition.assign(v, 1)

        partition.update_weights(graph)
        return partition

    def _uncoarsen_and_refine(self, partition: Partition, balance_ratio: float) -> Partition:
        """
        Обратная проекция с улучшением на каждом уровне
        """
        current_partition = partition
        total_levels = len(self.levels)
        
        for i, level in enumerate(reversed(self.levels)):
            level_num = total_levels - i
            
            print(f"     Level {level_num}: projecting from {level.coarse_graph.num_vertices} to {level.graph.num_vertices} vertices")
            
            # 1. Проекция разбиения
            current_partition = level.coarse_graph.expand_partition(current_partition)
            
            # Обновляем веса частей на основе текущего графа
            current_partition.update_weights(level.graph)
            
            # 2. Улучшаем разбиение на текущем уровне
            for attempt in range(self.refinement_passes):
                kl = KernighanLin(max_passes=5, seed=self.seed + attempt)
                current_partition, _ = kl.partition(level.graph, balance_ratio)
                
                cut = current_partition.cut_weight(level.graph)
                self._record_iteration(cut)
            
            cut = current_partition.cut_weight(level.graph)
            print(f"       Cut after refinement: {cut}")
        
        return current_partition
    def get_coarsening_history(self) -> List[Dict[str, Any]]:
        """
        Получение истории стягивания для визуализации

        Returns:
            Список словарей с информацией о каждом уровне
        """
        history = []
        for level in self.levels:
            history.append(
                {
                    "level": level.level,
                    "vertices": level.graph.num_vertices,
                    "edges": level.graph.num_edges,
                    "compression_ratio": level.compression_ratio,
                    "graph": level.graph,
                    "coarse_graph": level.coarse_graph,
                    "reverse_map": level.reverse_map,
                }
            )
        return history

    def print_levels(self) -> None:
        """Вывод информации об уровнях стягивания"""
        print("\n" + "=" * 50)
        print("COARSENING LEVELS")
        print("=" * 50)
        print(f"{'Level':<6} {'Vertices':<10} {'Edges':<10} {'Compression':<12}")
        print("-" * 50)

        for level in self.levels:
            print(
                f"{level.level:<6} {level.graph.num_vertices:<10} {level.graph.num_edges:<10} {level.compression_ratio:<12.4f}"
            )

        if self.levels:
            final = self.levels[-1]
            print(
                f"\nFinal compression: {final.coarse_graph.num_vertices} / {self.levels[0].graph.num_vertices} = {final.compression_ratio:.4f}"
            )
        print("=" * 50)


class AdaptiveMultilevelPartitioner(MultilevelPartitioner):
    """
    Адаптивный многоуровневый алгоритм

    Автоматически подбирает параметры в зависимости от размера графа
    """

    def __init__(self, total_vertices: int = 0, seed: int = 42):
        """
        Args:
            total_vertices: ожидаемое количество вершин (для подбора параметров)
            seed: seed для воспроизводимости
        """
        # Адаптивный подбор параметров
        if total_vertices > 0:
            min_coarse = max(20, total_vertices // 100)
            max_levels = max(3, min(10, int(total_vertices**0.3)))
            refinement = max(1, min(5, total_vertices // 1000))
        else:
            min_coarse = 50
            max_levels = 10
            refinement = 3

        super().__init__(
            min_coarse_vertices=min_coarse,
            max_levels=max_levels,
            coarsening_ratio=0.5,
            refinement_passes=refinement,
            seed=seed,
        )
        self.name = "AdaptiveMultilevel"


class FastMultilevelPartitioner(MultilevelPartitioner):
    """
    Быстрый многоуровневый алгоритм для больших графов

    Использует упрощённое стягивание и меньше проходов улучшения
    """

    def __init__(self, seed: int = 42):
        super().__init__(min_coarse_vertices=100, max_levels=8, coarsening_ratio=0.6, refinement_passes=1, seed=seed)
        self.name = "FastMultilevel"

    def _find_matching(self, graph: Graph) -> List[Tuple[int, int]]:
        """
        Упрощённый поиск паросочетания (быстрее для больших графов)
        """
        used = set()
        matching = []

        # Простой жадный алгоритм (O(V))
        for v in range(graph.num_vertices):
            if v in used:
                continue

            # Ищем любого непомеченного соседа
            for neighbor in graph.get_neighbors(v):
                if neighbor not in used:
                    matching.append((v, neighbor))
                    used.add(v)
                    used.add(neighbor)
                    break

        return matching
