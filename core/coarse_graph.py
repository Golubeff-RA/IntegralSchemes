"""
Специальный класс для представления грубых графов на этапе стягивания
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition


class CoarseGraph:
    """
    Грубый граф, полученный стягиванием исходного
    """
    
    def __init__(self, num_coarse_vertices: int):
        self.num_vertices = num_coarse_vertices
        self.num_edges = 0
        self._adjacency: List[Dict[int, int]] = [{} for _ in range(num_coarse_vertices)]
        self._vertex_weights = np.zeros(num_coarse_vertices, dtype=np.int32)
        self._coarse_to_original: List[List[int]] = [[] for _ in range(num_coarse_vertices)]
        self._original_to_coarse: Dict[int, int] = {}
    
    def add_edge(self, u: int, v: int, weight: int = 1) -> None:
        if u == v:
            return
        
        if v in self._adjacency[u]:
            self._adjacency[u][v] += weight
            self._adjacency[v][u] += weight
        else:
            self._adjacency[u][v] = weight
            self._adjacency[v][u] = weight
            self.num_edges += 1
    
    def get_neighbors(self, v: int) -> Dict[int, int]:
        if v < len(self._adjacency):
            return self._adjacency[v].copy()
        return {}
    
    def get_vertex_weight(self, v: int) -> int:
        if v < len(self._vertex_weights):
            return self._vertex_weights[v]
        return 0
    
    def get_original_vertices(self, coarse_vertex: int) -> List[int]:
        if coarse_vertex < len(self._coarse_to_original):
            return self._coarse_to_original[coarse_vertex].copy()
        return []
    
    def get_coarse_vertex(self, original_vertex: int) -> int:
        return self._original_to_coarse.get(original_vertex, -1)
    
    def get_original_vertex_count(self) -> int:
        if self._original_to_coarse:
            return max(self._original_to_coarse.keys()) + 1
        return 0
    
    def expand_partition(self, partition: Partition) -> Partition:
        """
        Проекция разбиения с грубого графа на исходный
        """
        # Находим количество исходных вершин
        if not self._original_to_coarse:
            return Partition(0, partition.num_parts)
        
        num_original = self.get_original_vertex_count()
        
        # Создаём новое разбиение правильного размера
        original_partition = Partition(num_original, partition.num_parts)
        
        # Проецируем разбиение
        for original_v, coarse_v in self._original_to_coarse.items():
            if coarse_v < partition.num_vertices:
                part = partition.get_part(coarse_v)
                if part != -1 and part < partition.num_parts:
                    original_partition.assign(original_v, part)
        
        # Назначаем оставшиеся вершины (если есть)
        for v in range(num_original):
            if original_partition.get_part(v) == -1:
                # Назначаем в часть с минимальным размером
                sizes = original_partition.part_sizes
                min_part = np.argmin(sizes)
                original_partition.assign(v, min_part)
        
        return original_partition
    
    @staticmethod
    def from_matching(graph: Graph, matching: List[Tuple[int, int]]) -> 'CoarseGraph':
        """Создание грубого графа на основе паросочетания"""
        vertex_map = {}
        next_coarse_id = 0
        used = set()
        
        # Обрабатываем пары
        for u, v in matching:
            if u in used or v in used:
                continue
            
            vertex_map[u] = next_coarse_id
            vertex_map[v] = next_coarse_id
            used.add(u)
            used.add(v)
            next_coarse_id += 1
        
        # Оставшиеся вершины
        for v in range(graph.num_vertices):
            if v not in vertex_map:
                vertex_map[v] = next_coarse_id
                next_coarse_id += 1
        
        # Создаём грубый граф
        coarse = CoarseGraph(next_coarse_id)
        
        # Заполняем маппинги
        for v, cv in vertex_map.items():
            coarse._coarse_to_original[cv].append(v)
            coarse._original_to_coarse[v] = cv
            coarse._vertex_weights[cv] += graph.get_vertex_weight(v)
        
        # Добавляем рёбра
        edge_weights = defaultdict(int)
        for u, v, w in graph.edges():
            cu = vertex_map[u]
            cv = vertex_map[v]
            if cu != cv:
                if cu < cv:
                    edge_weights[(cu, cv)] += w
                else:
                    edge_weights[(cv, cu)] += w
        
        for (cu, cv), w in edge_weights.items():
            coarse._adjacency[cu][cv] = w
            coarse._adjacency[cv][cu] = w
            coarse.num_edges += 1
        
        return coarse
    
    def to_graph(self) -> Graph:
        """Преобразование в обычный граф"""
        g = Graph(self.num_vertices)
        
        for v in range(self.num_vertices):
            g.set_vertex_weight(v, self._vertex_weights[v])
        
        for u in range(self.num_vertices):
            for v, w in self._adjacency[u].items():
                if u < v:
                    g.add_edge(u, v, w)
        
        return g
    
    def __repr__(self) -> str:
        return f"CoarseGraph(n={self.num_vertices}, m={self.num_edges})"