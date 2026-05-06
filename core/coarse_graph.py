"""
Грубый граф для многоуровневого разбиения
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition


class CoarseGraph:
    """Грубый граф, полученный стягиванием нескольких вершин"""
    
    def __init__(self, num_vertices: int = 0):
        self.num_vertices = num_vertices
        self.num_edges = 0
        self._adjacency: List[Dict[int, int]] = [{} for _ in range(num_vertices)]
        self._vertex_weights = np.zeros(num_vertices, dtype=np.int32)
        self._coarse_to_original: List[List[int]] = [[] for _ in range(num_vertices)]
        self._original_to_coarse: Dict[int, int] = {}
    
    def add_edge(self, u: int, v: int, weight: int = 1) -> None:
        if u == v:
            return
        if u >= self.num_vertices or v >= self.num_vertices:
            return
        
        if v in self._adjacency[u]:
            self._adjacency[u][v] += weight
            self._adjacency[v][u] += weight
        else:
            self._adjacency[u][v] = weight
            self._adjacency[v][u] = weight
            self.num_edges += 1
    
    def get_coarse_vertex(self, original_vertex: int) -> int:
        return self._original_to_coarse.get(original_vertex, -1)
    
    def get_original_vertices(self, coarse_vertex: int) -> List[int]:
        if coarse_vertex < len(self._coarse_to_original):
            return self._coarse_to_original[coarse_vertex].copy()
        return []
    
    def get_original_vertex_count(self) -> int:
        if not self._original_to_coarse:
            return 0
        return max(self._original_to_coarse.keys()) + 1
    
    def expand_partition(self, partition: Partition) -> Partition:
        if not self._original_to_coarse:
            return Partition(0)
        
        num_original = self.get_original_vertex_count()
        original_partition = Partition(num_original)
        
        for original_v, coarse_v in self._original_to_coarse.items():
            if coarse_v < partition.num_vertices:
                part = partition.get_part(coarse_v)
                if part != -1:
                    original_partition.assign(original_v, part)
        
        for v in range(num_original):
            if original_partition.get_part(v) == -1:
                if original_partition.size0 <= original_partition.size1:
                    original_partition.assign(v, 0)
                else:
                    original_partition.assign(v, 1)
        
        return original_partition
    
    def to_graph(self) -> Graph:
        g = Graph(self.num_vertices)
        for v in range(self.num_vertices):
            g.set_vertex_weight(v, self._vertex_weights[v])
        for u in range(self.num_vertices):
            for v, w in self._adjacency[u].items():
                if u < v:
                    g.add_edge(u, v, w)
        return g
    
    @staticmethod
    def from_matching(graph: Graph, matching: List[Tuple[int, int]]) -> 'CoarseGraph':
        n = graph.num_vertices
        
        vertex_map = {}
        next_id = 0
        used = set()
        
        for u, v in matching:
            if u in used or v in used:
                continue
            vertex_map[u] = next_id
            vertex_map[v] = next_id
            used.add(u)
            used.add(v)
            next_id += 1
        
        for v in range(n):
            if v not in vertex_map:
                vertex_map[v] = next_id
                next_id += 1
        
        coarse = CoarseGraph(next_id)
        
        # Заполняем маппинги и веса - ПРАВИЛЬНАЯ ИТЕРАЦИЯ
        for orig_v, coarse_v in vertex_map.items():
            # orig_v - это int, coarse_v - это int
            coarse._coarse_to_original[coarse_v].append(orig_v)
            coarse._original_to_coarse[orig_v] = coarse_v
            # Получаем вес вершины
            weight = graph.get_vertex_weight(orig_v)
            coarse._vertex_weights[coarse_v] += int(weight)
        
        # Добавляем рёбра
        edge_weights = defaultdict(int)
        for u, v, w in graph.edges():
            cu = vertex_map[u]
            cv = vertex_map[v]
            if cu != cv:
                key = (min(cu, cv), max(cu, cv))
                edge_weights[key] += w
        
        for (cu, cv), w in edge_weights.items():
            coarse._adjacency[cu][cv] = w
            coarse._adjacency[cv][cu] = w
            coarse.num_edges += 1
        
        return coarse
    def __repr__(self) -> str:
        return f"CoarseGraph(n={self.num_vertices}, edges={self.num_edges})"