"""
Генератор графов Барабаши-Альберт (preferential attachment)
Эффективен для больших графов O(n log n)
"""

import random
import numpy as np
from typing import List, Optional, Tuple
import heapq

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.graph import Graph
from .base_generator import BaseGraphGenerator


class BarabasiAlbertGenerator(BaseGraphGenerator):
    """
    Генератор масштабно-инвариантных графов
    
    Параметры:
    - n: количество вершин
    - m0: начальное количество вершин (образуют полный граф)
    - m: количество рёбер для каждой новой вершины
    - weight_range: диапазон весов рёбер
    - vertex_weight_range: диапазон весов вершин
    """
    
    def __init__(self,
                 n: int = 1000,
                 m0: int = 5,
                 m: int = 2,
                 weight_range: Tuple[int, int] = (1, 1),
                 vertex_weight_range: Tuple[int, int] = (1, 1),
                 seed: Optional[int] = None):
        
        super().__init__(seed)
        self.n = n
        self.m0 = m0
        self.m = m
        self.weight_range = weight_range
        self.vertex_weight_range = vertex_weight_range
        
        if m0 > n:
            raise ValueError("m0 must be <= n")
        if m > m0:
            raise ValueError("m must be <= m0")
    
    def generate(self) -> Graph:
        """Генерация графа Барабаши-Альберт"""
        graph = Graph(self.n)
        
        # 1. Начальный полный граф из m0 вершин
        for i in range(self.m0):
            for j in range(i + 1, self.m0):
                w = random.randint(*self.weight_range)
                graph.add_edge(i, j, w)
        
        # 2. Preferential attachment для остальных вершин
        degrees = [graph.get_degree(i) for i in range(self.m0)]
        total_degree = sum(degrees)
        
        for new_v in range(self.m0, self.n):
            # Выбираем m вершин с вероятностью пропорциональной степени
            targets = self._weighted_choice(degrees, self.m)
            
            for target in targets:
                w = random.randint(*self.weight_range)
                graph.add_edge(new_v, target, w)
            
            # Обновляем степени
            degrees.append(self.m)
            for target in targets:
                degrees[target] += 1
            total_degree += 2 * self.m
        
        # 3. Добавляем веса вершин
        for v in range(self.n):
            w = random.randint(*self.vertex_weight_range)
            graph.set_vertex_weight(v, w)
        
        return graph
    
    def _weighted_choice(self, weights: List[int], k: int) -> List[int]:
        """Выбор k элементов с вероятностью пропорциональной весу"""
        if sum(weights) == 0:
            return random.sample(range(len(weights)), min(k, len(weights)))
        
        # Используем heap для эффективного выбора
        heap = [(-random.random() ** (1.0 / weights[i]), i) for i in range(len(weights))]
        heapq.heapify(heap)
        
        chosen = set()
        while len(chosen) < k and heap:
            _, idx = heapq.heappop(heap)
            chosen.add(idx)
        
        return list(chosen)