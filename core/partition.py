"""
Класс для представления разбиения графа на части
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np
from collections import defaultdict

from .graph import Graph


class Partition:
    """
    Разбиение графа на заданное количество частей
    """
    
    def __init__(self, num_vertices: int, num_parts: int = 2):
        self.num_vertices = num_vertices
        self.num_parts = num_parts
        self._assignment = np.full(num_vertices, -1, dtype=np.int32)
        self._part_sizes = np.zeros(num_parts, dtype=np.int32)
        self._part_weights = np.zeros(num_parts, dtype=np.int64)
        self._cut_size_cache = None
    
    def assign(self, vertex: int, part: int) -> None:
        """Назначение вершины в часть"""
        if vertex < 0 or vertex >= self.num_vertices:
            raise IndexError(f"Vertex {vertex} out of range [0, {self.num_vertices})")
        
        if part < 0 or part >= self.num_parts:
            raise IndexError(f"Part {part} out of range [0, {self.num_parts})")
        
        old_part = self._assignment[vertex]
        
        if old_part == part:
            return
        
        if old_part != -1:
            self._part_sizes[old_part] -= 1
        
        self._assignment[vertex] = part
        self._part_sizes[part] += 1
        self._cut_size_cache = None
    
    def get_part(self, vertex: int) -> int:
        """Получение части для вершины"""
        if vertex < 0 or vertex >= self.num_vertices:
            return -1
        return self._assignment[vertex]
    
    def get_vertices_in_part(self, part: int) -> List[int]:
        return [v for v in range(self.num_vertices) if self._assignment[v] == part]
    
    @property
    def part_sizes(self) -> np.ndarray:
        return self._part_sizes.copy()
    
    def compute_part_weights(self, graph) -> np.ndarray:
        weights = np.zeros(self.num_parts, dtype=np.int64)
        for v in range(self.num_vertices):
            part = self._assignment[v]
            if part != -1:
                weights[part] += graph.get_vertex_weight(v)
        self._part_weights = weights
        return weights.copy()
    
    def cut_edges(self, graph) -> int:
        """Вычисление количества разрезов"""
        if self._cut_size_cache is not None:
            return self._cut_size_cache
        
        # Проверяем соответствие размеров
        if graph.num_vertices != self.num_vertices:
            print(f"    ERROR: graph has {graph.num_vertices} vertices but partition has {self.num_vertices}")
            return 0
        
        cut = 0
        for u, v, w in graph.edges():
            if u >= self.num_vertices or v >= self.num_vertices:
                continue
            
            pu = self._assignment[u]
            pv = self._assignment[v]
            
            if pu != -1 and pv != -1 and pu != pv:
                cut += w
        
        self._cut_size_cache = cut
        return cut
    
    def is_balanced(self, balance_ratio: float = 0.5) -> bool:
        if self.num_parts == 0:
            return True
        
        target_size = self.num_vertices / self.num_parts
        max_allowed = target_size * (1 + (1 - balance_ratio))
        
        return np.all(self._part_sizes <= max_allowed)
    
    def balance_quality(self) -> float:
        if self.num_parts == 0:
            return 1.0
        
        sizes = self._part_sizes[self._part_sizes > 0]
        if len(sizes) == 0:
            return 0.0
        
        return np.min(sizes) / np.max(sizes)
    
    def move_vertex(self, vertex: int, new_part: int, graph) -> int:
        old_part = self._assignment[vertex]
        
        if old_part == new_part or old_part == -1:
            return 0
        
        delta = 0
        for neighbor, weight in graph.get_neighbors(vertex).items():
            if neighbor < self.num_vertices:
                neighbor_part = self._assignment[neighbor]
                if neighbor_part == old_part:
                    delta += weight
                elif neighbor_part == new_part:
                    delta -= weight
        
        self._assignment[vertex] = new_part
        self._part_sizes[old_part] -= 1
        self._part_sizes[new_part] += 1
        self._cut_size_cache = None
        
        return delta
    
    def fix_unassigned(self, graph: Graph = None) -> None:
        """Назначает все неназначенные вершины"""
        for v in range(self.num_vertices):
            if self._assignment[v] == -1:
                min_part = np.argmin(self._part_sizes)
                self.assign(v, min_part)
        self._cut_size_cache = None
    
    def is_complete(self) -> bool:
        return np.all(self._assignment != -1)
    
    def copy(self) -> 'Partition':
        new_part = Partition(self.num_vertices, self.num_parts)
        new_part._assignment = self._assignment.copy()
        new_part._part_sizes = self._part_sizes.copy()
        new_part._part_weights = self._part_weights.copy()
        return new_part
    
    def to_array(self) -> np.ndarray:
        return self._assignment.copy()
    
    @staticmethod
    def from_array(assignment: np.ndarray, num_parts: int = None) -> 'Partition':
        if num_parts is None:
            num_parts = int(np.max(assignment)) + 1
        
        num_vertices = len(assignment)
        partition = Partition(num_vertices, num_parts)
        
        for v, part in enumerate(assignment):
            if part >= 0:
                partition.assign(v, part)
        
        return partition
    
    def __repr__(self) -> str:
        return f"Partition({self.num_vertices} vertices, {self.num_parts} parts)"