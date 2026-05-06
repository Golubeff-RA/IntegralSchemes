"""Базовый класс для генераторов графов"""

from abc import ABC, abstractmethod
import random
import numpy as np
from typing import Optional, Tuple, Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.graph import Graph


class BaseGraphGenerator(ABC):
    """Абстрактный базовый класс для генерации графов"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    @abstractmethod
    def generate(self) -> Graph:
        """Генерация графа"""
        pass
    
    def generate_with_weights(self, 
                              vertex_weight_range: Tuple[int, int] = (1, 10),
                              edge_weight_range: Tuple[int, int] = (1, 10)) -> Graph:
        """Генерация графа со случайными весами"""
        graph = self.generate()
        
        # Добавляем случайные веса вершин
        for v in range(graph.num_vertices):
            weight = random.randint(*vertex_weight_range)
            graph.set_vertex_weight(v, weight)
        
        # Веса рёбер уже могут быть заданы генератором
        # Если нет - добавляем случайные
        if vertex_weight_range != (1, 1):
            for u, v, w in list(graph.edges()):
                if w == 1:
                    new_w = random.randint(*edge_weight_range)
                    graph.add_edge(u, v, new_w - 1)  # Обновляем вес
        
        return graph
    
    def get_stats(self, graph: Graph) -> Dict[str, Any]:
        """Получение статистики графа"""
        degrees = [graph.get_degree(v) for v in range(graph.num_vertices)]
        
        return {
            'num_vertices': graph.num_vertices,
            'num_edges': graph.num_edges,
            'density': 2 * graph.num_edges / (graph.num_vertices * (graph.num_vertices - 1)) if graph.num_vertices > 1 else 0,
            'avg_degree': sum(degrees) / graph.num_vertices if graph.num_vertices > 0 else 0,
            'max_degree': max(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0,
            'vertex_weights': [graph.get_vertex_weight(v) for v in range(graph.num_vertices)],
            'edge_weights': [w for _, _, w in graph.edges()]
        }