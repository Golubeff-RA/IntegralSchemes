"""
Эффективная реализация графа для 10^5 вершин и 5×10^5 рёбер
"""

import numpy as np
from typing import List, Tuple, Iterator, Dict, Optional
from collections import defaultdict


class Graph:
    """Неориентированный граф с весами вершин и рёбер"""
    
    def __init__(self, num_vertices: int = 0):
        self._num_vertices = num_vertices
        self._num_edges = 0
        self._adj = [[] for _ in range(num_vertices)]
        self._adj_weights = [[] for _ in range(num_vertices)]
        self._vertex_weights = np.ones(num_vertices, dtype=np.int32)
        self._csr_built = False
    
    def add_edge(self, u: int, v: int, weight: int = 1) -> None:
        if u < 0 or u >= self._num_vertices or v < 0 or v >= self._num_vertices:
            raise IndexError(f"Vertex out of range: {u}, {v}")
        if u == v:
            return
        
        # Проверка на существующее ребро
        for i, nb in enumerate(self._adj[u]):
            if nb == v:
                self._adj_weights[u][i] += weight
                for j, nb2 in enumerate(self._adj[v]):
                    if nb2 == u:
                        self._adj_weights[v][j] += weight
                        return
        
        self._adj[u].append(v)
        self._adj_weights[u].append(weight)
        self._adj[v].append(u)
        self._adj_weights[v].append(weight)
        self._num_edges += 1
        self._csr_built = False
    
    def set_vertex_weight(self, v: int, weight: int) -> None:
        if 0 <= v < self._num_vertices:
            self._vertex_weights[v] = int(weight)
    
    def get_vertex_weight(self, v: int) -> int:
        """Получение веса вершины"""
        if v is None:
            return 1
        try:
            if 0 <= v < self._num_vertices:
                w = self._vertex_weights[v]
                if isinstance(w, (list, np.ndarray)):
                    return int(w[0]) if len(w) > 0 else 1
                return int(w)
        except (TypeError, ValueError):
            return 1
        return 1
    
    def get_edge_weight(self, u: int, v: int) -> int:
        for i, nb in enumerate(self._adj[u]):
            if nb == v:
                return self._adj_weights[u][i]
        return 0
    
    def get_neighbors(self, v: int) -> List[Tuple[int, int]]:
        return list(zip(self._adj[v], self._adj_weights[v]))
    
    def get_degree(self, v: int) -> int:
        return len(self._adj[v])
    
    @property
    def num_vertices(self) -> int:
        return self._num_vertices
    
    @property
    def num_edges(self) -> int:
        return self._num_edges
    
    def edges(self) -> Iterator[Tuple[int, int, int]]:
        seen = set()
        for u in range(self._num_vertices):
            for i, v in enumerate(self._adj[u]):
                if (u, v) not in seen and (v, u) not in seen:
                    seen.add((u, v))
                    yield (u, v, self._adj_weights[u][i])
    
    def has_edge(self, u: int, v: int) -> bool:
        return v in self._adj[u]
    
    def save_to_file(self, filename: str) -> None:
        """Сохранение графа в файл с весами вершин"""
        with open(filename, 'w') as f:
            # Заголовок: количество вершин, количество рёбер, флаг наличия весов вершин
            f.write(f"{self._num_vertices} {self._num_edges} 1\n")  # 1 = есть веса вершин
            
            # Строка с весами вершин
            vertex_weights = [str(self.get_vertex_weight(v)) for v in range(self._num_vertices)]
            f.write(" ".join(vertex_weights) + "\n")
            
            # Рёбра
            seen = set()
            for u in range(self._num_vertices):
                for i, v in enumerate(self._adj[u]):
                    if u < v and (u, v) not in seen:
                        seen.add((u, v))
                        w = self._adj_weights[u][i]
                        f.write(f"{u+1} {v+1} {w}\n")
    
    @staticmethod
    def load_from_file(filename: str) -> 'Graph':
        """Загрузка графа из файла с поддержкой весов вершин"""
        with open(filename, 'r') as f:
            # Читаем заголовок
            parts = f.readline().split()
            n = int(parts[0])
            m = int(parts[1])
            has_vertex_weights = len(parts) > 2 and parts[2] == '1'
            
            g = Graph(n)
            
            # Читаем веса вершин (если есть)
            if has_vertex_weights:
                weights_line = f.readline().strip()
                while weights_line == '':
                    weights_line = f.readline().strip()
                weights = list(map(int, weights_line.split()))
                for v, w in enumerate(weights[:n]):
                    g.set_vertex_weight(v, w)
            
            # Читаем рёбра
            for _ in range(m):
                line = f.readline().strip()
                while line == '':
                    line = f.readline().strip()
                
                parts = line.split()
                u = int(parts[0]) - 1
                v = int(parts[1]) - 1
                w = int(parts[2]) if len(parts) > 2 else 1
                g.add_edge(u, v, w)
        
        return g
    
    def __repr__(self) -> str:
        return f"Graph(n={self._num_vertices}, m={self._num_edges})"