"""
Этап стягивания (coarsening) для многоуровневого разбиения
"""

import random
from typing import List, Tuple, Dict, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.graph import Graph
from core.coarse_graph import CoarseGraph


class Coarsener:
    """
    Многоуровневое стягивание графа
    """
    
    def __init__(self, strategy: str = 'heavy_edge', 
                 min_vertices: int = 20,
                 max_levels: int = 10):
        self.strategy = strategy
        self.min_vertices = min_vertices
        self.max_levels = max_levels
        self.statistics = {
            'levels': 0,
            'compression_ratios': [],
            'vertex_counts': []
        }
    
    def coarsen(self, graph: Graph) -> List[CoarseGraph]:
        """
        Многоуровневое стягивание графа
        """
        coarse_graphs = []
        current = graph
        
        for level in range(self.max_levels):
            print(f"  Coarsening level {level + 1}: {current.num_vertices} vertices, {current.num_edges} edges")
            
            # Находим паросочетание
            matching = self._find_matching(current)
            
            print(f"    Found {len(matching)} matching pairs")
            
            if not matching or len(matching) < current.num_vertices // 20:
                print(f"    Stopping: insufficient matching")
                break
            
            # Стягиваем граф
            coarse = CoarseGraph.from_matching(current, matching)
            coarse_graphs.append(coarse)
            
            # Сохраняем статистику
            compression_ratio = coarse.num_vertices / current.num_vertices
            self.statistics['compression_ratios'].append(compression_ratio)
            self.statistics['vertex_counts'].append(coarse.num_vertices)
            
            print(f"    Coarse graph: {coarse.num_vertices} vertices, {coarse.num_edges} edges")
            print(f"    Compression ratio: {compression_ratio:.3f}")
            
            # Критерий остановки
            if coarse.num_vertices <= self.min_vertices:
                print(f"    Reached minimum vertices threshold")
                break
            
            # Преобразуем для следующего уровня
            current = coarse.to_graph()
        
        self.statistics['levels'] = len(coarse_graphs)
        if coarse_graphs:
            print(f"  Coarsening completed: {len(coarse_graphs)} levels, final size: {coarse_graphs[-1].num_vertices}")
        else:
            print(f"  Coarsening completed: no levels created")
        
        return coarse_graphs
    
    def _find_matching(self, graph: Graph) -> List[Tuple[int, int]]:
        """
        Улучшенный поиск паросочетания
        """
        used = set()
        matching = []
        
        # Сортируем вершины по степени (убывание)
        vertices = list(range(graph.num_vertices))
        vertices.sort(key=lambda v: graph.get_degree(v), reverse=True)
        
        for v in vertices:
            if v in used:
                continue
            
            # Ищем лучшего непомеченного соседа
            best_neighbor = None
            best_weight = -1
            
            for neighbor, weight in graph.get_neighbors(v).items():
                if neighbor not in used and weight > best_weight:
                    best_weight = weight
                    best_neighbor = neighbor
            
            if best_neighbor is not None:
                matching.append((v, best_neighbor))
                used.add(v)
                used.add(best_neighbor)
        
        return matching
    
    def get_statistics(self) -> Dict:
        return self.statistics.copy()


class AdaptiveCoarsener(Coarsener):
    """Адаптивный коарсенер"""
    
    def __init__(self, min_vertices: int = 20, max_levels: int = 10):
        super().__init__(strategy='heavy_edge', min_vertices=min_vertices, 
                        max_levels=max_levels)
        self.name = "AdaptiveCoarsener"