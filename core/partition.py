"""
Разбиение графа на 2 части (бисекция) с учётом весов вершин и рёбер
"""

import numpy as np
from typing import List, Tuple, Optional, Dict

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph


class Partition:
    """
    Бисекция графа (разбиение на 2 части)

    Каждая вершина может быть:
    - в части 0
    - в части 1
    - не назначена (-1)
    """

    def __init__(self, num_vertices: int):
        """
        Инициализация пустого разбиения

        Args:
            num_vertices: общее количество вершин
        """
        self._num_vertices = num_vertices

        # Принадлежность вершин: -1 = не назначена, 0 или 1 = часть
        self._part = np.full(num_vertices, -1, dtype=np.int8)

        # Количество вершин в каждой части
        self._size = np.zeros(2, dtype=np.int32)

        # Суммарный вес вершин в каждой части
        self._weight = np.zeros(2, dtype=np.int64)

        # Кэш для cut weight
        self._cut_weight_cache = None
        self._cut_edges_cache = None

    @property
    def num_vertices(self) -> int:
        """Количество вершин в разбиении"""
        return self._num_vertices

    def assign(self, vertex: int, part: int) -> None:
        """
        Назначение вершины в часть

        Args:
            vertex: индекс вершины
            part: 0 или 1
        """
        if vertex < 0 or vertex >= self._num_vertices:
            raise IndexError(f"Vertex {vertex} out of range [0, {self._num_vertices})")

        if part not in (0, 1):
            raise ValueError(f"Part must be 0 or 1, got {part}")

        old_part = self._part[vertex]

        if old_part == part:
            return

        # Обновляем счётчики при перемещении
        if old_part != -1:
            self._size[old_part] -= 1

        self._part[vertex] = part
        self._size[part] += 1

        # Инвалидируем кэши
        self._cut_weight_cache = None
        self._cut_edges_cache = None

    def set_vertex_weight(self, vertex: int, weight: int, graph: Graph) -> None:
        """
        Установка веса вершины и обновление веса части

        Args:
            vertex: индекс вершины
            weight: новый вес
            graph: граф (для получения текущего веса)
        """
        old_weight = graph.get_vertex_weight(vertex)
        part = self._part[vertex]

        if part != -1:
            self._weight[part] -= old_weight
            self._weight[part] += weight

    def update_weights(self, graph: Graph) -> None:
        """Пересчёт весов всех частей на основе весов вершин в графе"""
        self._weight.fill(0)
        for v in range(self._num_vertices):
            part = self._part[v]
            if part != -1:
                self._weight[part] += graph.get_vertex_weight(v)

    def get_part(self, vertex: int) -> int:
        """Получение части вершины (-1, 0 или 1)"""
        if vertex < 0 or vertex >= self._num_vertices:
            return -1
        return self._part[vertex]

    def get_vertices(self, part: int) -> List[int]:
        """Получение всех вершин в указанной части"""
        return [v for v in range(self._num_vertices) if self._part[v] == part]

    @property
    def size(self) -> np.ndarray:
        """Количество вершин в каждой части"""
        return self._size.copy()

    @property
    def weight(self) -> np.ndarray:
        """Суммарный вес вершин в каждой части"""
        return self._weight.copy()

    @property
    def size0(self) -> int:
        """Количество вершин в части 0"""
        return self._size[0]

    @property
    def size1(self) -> int:
        """Количество вершин в части 1"""
        return self._size[1]

    @property
    def weight0(self) -> int:
        """Суммарный вес вершин в части 0"""
        return int(self._weight[0])

    @property
    def weight1(self) -> int:
        """Суммарный вес вершин в части 1"""
        return int(self._weight[1])

    @property
    def total_weight(self) -> int:
        """Суммарный вес всех вершин"""
        return int(self._weight[0] + self._weight[1])

    def cut_weight(self, graph: Graph) -> int:
        """Суммарный вес разрезанных рёбер (cut weight)"""
        if self._cut_weight_cache is not None:
            return self._cut_weight_cache

        cut = 0
        for u, v, w in graph.edges():
            pu = self._part[u]
            pv = self._part[v]

            if pu != -1 and pv != -1 and pu != pv:
                cut += w

        self._cut_weight_cache = cut
        return cut

    def cut_edges(self, graph: Graph) -> List[Tuple[int, int, int]]:
        """Список разрезанных рёбер с весами"""
        if self._cut_edges_cache is not None:
            return self._cut_edges_cache

        edges = []
        for u, v, w in graph.edges():
            pu = self._part[u]
            pv = self._part[v]

            if pu != -1 and pv != -1 and pu != pv:
                edges.append((u, v, w))

        self._cut_edges_cache = edges
        return edges

    def balance_quality(self) -> float:
        """Качество балансировки по количеству вершин (чем ближе к 1, тем лучше)"""
        if self._size[0] == 0 and self._size[1] == 0:
            return 0.0

        min_size = min(self._size[0], self._size[1])
        max_size = max(self._size[0], self._size[1])

        return min_size / max_size if max_size > 0 else 0.0

    def weight_balance_quality(self) -> float:
        """Качество балансировки по весам (чем ближе к 1, тем лучше)"""
        if self._weight[0] == 0 and self._weight[1] == 0:
            return 0.0

        min_weight = min(self._weight[0], self._weight[1])
        max_weight = max(self._weight[0], self._weight[1])

        return min_weight / max_weight if max_weight > 0 else 0.0

    def move_vertex(self, vertex: int, graph: Graph) -> int:
        """
        Перемещение вершины в противоположную часть
        """
        old_part = self._part[vertex]
        
        if old_part == -1:
            return 0
        
        new_part = 1 - old_part
        
        # Вычисляем изменение cut weight
        delta = 0
        for neighbor, weight in graph.get_neighbors(vertex):
            neighbor_part = self._part[neighbor]
            
            if neighbor_part == old_part:
                delta += weight
            elif neighbor_part == new_part:
                delta -= weight
        
        # Обновляем разбиение
        self._part[vertex] = new_part
        self._size[old_part] -= 1
        self._size[new_part] += 1
        
        # Обновляем веса частей
        vertex_weight = graph.get_vertex_weight(vertex)
        self._weight[old_part] -= vertex_weight
        self._weight[new_part] += vertex_weight
        
        # Инвалидируем кэши
        self._cut_weight_cache = None
        self._cut_edges_cache = None
        
        return delta

    def move_vertex_to(self, vertex: int, new_part: int, graph: Graph) -> int:
        """
        Перемещение вершины в конкретную часть
        """
        old_part = self._part[vertex]
        
        if old_part == new_part:
            return 0
        
        if old_part == -1:
            # Вершина не назначена - просто назначаем
            self.assign(vertex, new_part)
            self._weight[new_part] += graph.get_vertex_weight(vertex)
            return 0
        
        # Вычисляем изменение cut weight
        delta = 0
        for neighbor, weight in graph.get_neighbors(vertex):
            neighbor_part = self._part[neighbor]
            
            if neighbor_part == old_part:
                delta += weight
            elif neighbor_part == new_part:
                delta -= weight
        
        # Обновляем разбиение
        self._part[vertex] = new_part
        self._size[old_part] -= 1
        self._size[new_part] += 1
        
        # Обновляем веса частей
        vertex_weight = graph.get_vertex_weight(vertex)
        self._weight[old_part] -= vertex_weight
        self._weight[new_part] += vertex_weight
        
        # Инвалидируем кэши
        self._cut_weight_cache = None
        self._cut_edges_cache = None
        
        return delta

    def swap_vertices(self, v0: int, v1: int, graph: Graph) -> int:
        """
        Обмен местами двух вершин
        """
        part0 = self._part[v0]
        part1 = self._part[v1]
        
        # Если вершины не назначены, назначаем их
        if part0 == -1 and part1 == -1:
            self.assign(v0, 0)
            self.assign(v1, 1)
            return 0
        
        if part0 == -1:
            # v0 не назначена - перемещаем v1 в часть 0
            return self.move_vertex_to(v1, 0, graph)
        
        if part1 == -1:
            # v1 не назначена - перемещаем v0 в часть 1
            return self.move_vertex_to(v0, 1, graph)
        
        # Если вершины в одной части
        if part0 == part1:
            # Перемещаем одну вершину в другую часть
            if part0 == 0:
                return self.move_vertex_to(v1, 1, graph)
            else:
                return self.move_vertex_to(v0, 0, graph)
        
        # Нормальный обмен (разные части)
        if part0 == 0 and part1 == 1:
            # Вычисляем изменение cut weight
            delta = 0
            
            # Вклад вершины v0
            for neighbor, weight in graph.get_neighbors(v0):
                if neighbor != v1:
                    neighbor_part = self._part[neighbor]
                    if neighbor_part == 0:
                        delta += weight  # теряем внутреннее ребро
                    elif neighbor_part == 1:
                        delta -= weight  # приобретаем внутреннее ребро
            
            # Вклад вершины v1
            for neighbor, weight in graph.get_neighbors(v1):
                if neighbor != v0:
                    neighbor_part = self._part[neighbor]
                    if neighbor_part == 1:
                        delta += weight
                    elif neighbor_part == 0:
                        delta -= weight
            
            # Учёт ребра между v0 и v1
            edge_weight = graph.get_edge_weight(v0, v1)
            delta = delta - 2 * edge_weight
            
            # Выполняем обмен
            self._part[v0] = 1
            self._part[v1] = 0
            
            # Обновляем веса частей
            w0 = graph.get_vertex_weight(v0)
            w1 = graph.get_vertex_weight(v1)
            self._weight[0] = self._weight[0] - w0 + w1
            self._weight[1] = self._weight[1] - w1 + w0
            
            # Инвалидируем кэши
            self._cut_weight_cache = None
            self._cut_edges_cache = None
            
            return delta
        else:
            # v0 в 1, v1 в 0 - меняем местами
            return self.swap_vertices(v1, v0, graph)

    def _compute_gain(self, vertex: int, graph: Graph) -> int:
        """Вычисление gain для вершины"""
        part = self._part[vertex]
        if part == -1:
            return 0

        internal = 0
        external = 0

        for neighbor, weight in graph.get_neighbors(vertex):
            neighbor_part = self._part[neighbor]
            if neighbor_part == part:
                internal += weight
            elif neighbor_part != -1:
                external += weight

        return external - internal

    def to_array(self) -> np.ndarray:
        """Преобразование в массив принадлежности"""
        return self._part.copy()

    @staticmethod
    def from_array(assignment: np.ndarray) -> "Partition":
        """
        Создание разбиения из массива принадлежности

        Args:
            assignment: массив, где каждый элемент 0 или 1
        """
        num_vertices = len(assignment)
        partition = Partition(num_vertices)

        for v, part in enumerate(assignment):
            if part in (0, 1):
                partition.assign(v, part)

        return partition

    def fix_unassigned(self, graph: Graph) -> None:
        """Назначает все неназначенные вершины в часть с меньшим весом"""
        for v in range(self._num_vertices):
            if self._part[v] == -1:
                target = 0 if self._weight[0] <= self._weight[1] else 1
                self.assign(v, target)
                self._weight[target] += graph.get_vertex_weight(v)

        self._cut_weight_cache = None
        self._cut_edges_cache = None

    def is_complete(self) -> bool:
        """Проверяет, что все вершины назначены"""
        return np.all(self._part != -1)

    def copy(self) -> "Partition":
        """Создание копии разбиения"""
        new = Partition(self._num_vertices)
        new._part = self._part.copy()
        new._size = self._size.copy()
        new._weight = self._weight.copy()
        new._cut_weight_cache = self._cut_weight_cache
        new._cut_edges_cache = self._cut_edges_cache
        return new

    def __len__(self) -> int:
        return self._num_vertices

    def __repr__(self) -> str:
        return (
            f"Partition(n={self._num_vertices}, size0={self.size0}, size1={self.size1}, cut={self._cut_weight_cache})"
        )
