"""
Этап проекции разбиения с грубого графа на исходный и улучшения
"""

from typing import List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.graph import Graph
from core.coarse_graph import CoarseGraph
from core.partition import Partition
from ..kernighan_lin import KernighanLin


class Uncoarsener:
    """
    Проекция разбиения с грубых уровней на исходный граф
    """
    
    def __init__(self, refinement_strategy: str = 'kl', 
                 max_refinement_passes: int = 5):
        self.refinement_strategy = refinement_strategy
        self.max_refinement_passes = max_refinement_passes
        
        if refinement_strategy == 'kl':
            self.refiner = KernighanLin(max_passes=2)
    
    def uncoarsen_and_refine(self, coarse_graphs: List[CoarseGraph],
                            partition: Partition) -> Partition:
        """
        Проекция разбиения на все уровни с улучшением
        """
        if not coarse_graphs:
            return partition
        
        current_partition = partition
        
        # Проходим по уровням от грубого к исходному
        for level, coarse_graph in enumerate(reversed(coarse_graphs)):
            print(f"  Uncoarsening level {level + 1}/{len(coarse_graphs)}")
            print(f"    Coarse graph: {coarse_graph.num_vertices} vertices")
            print(f"    Current partition: {current_partition.num_vertices} vertices")
            
            # Проекция разбиения
            current_partition = coarse_graph.expand_partition(current_partition)
            
            print(f"    After projection: {current_partition.num_vertices} vertices")
        
        return current_partition


class BoundaryRefinement(Uncoarsener):
    """Улучшение только на границе разбиения"""
    
    def __init__(self, max_passes: int = 5):
        super().__init__(refinement_strategy='boundary', max_refinement_passes=max_passes)
        self.name = "BoundaryRefinement"
    
    def _refine_partition(self, graph: Graph, partition: Partition) -> Partition:
        """Улучшение только граничных вершин"""
        boundary_vertices = self._find_boundary_vertices(graph, partition)
        
        if not boundary_vertices:
            return partition
        
        improved = True
        passes = 0
        
        while improved and passes < self.max_refinement_passes:
            improved = False
            
            for v in boundary_vertices:
                old_part = partition.get_part(v)
                if old_part == -1:
                    continue
                    
                best_gain = 0
                best_part = old_part
                
                for part in range(partition.num_parts):
                    if part == old_part:
                        continue
                    
                    gain = self._compute_move_gain(graph, partition, v, part)
                    if gain > best_gain:
                        best_gain = gain
                        best_part = part
                
                if best_gain > 0:
                    partition.move_vertex(v, best_part, graph)
                    improved = True
            
            boundary_vertices = self._find_boundary_vertices(graph, partition)
            passes += 1
        
        return partition
    
    def _find_boundary_vertices(self, graph: Graph, partition: Partition) -> List[int]:
        """Нахождение граничных вершин"""
        boundary = []
        for v in range(graph.num_vertices):
            v_part = partition.get_part(v)
            if v_part == -1:
                continue
            
            for neighbor in graph.get_neighbors(v):
                neighbor_part = partition.get_part(neighbor)
                if neighbor_part != -1 and neighbor_part != v_part:
                    boundary.append(v)
                    break
        
        return boundary
    
    def _compute_move_gain(self, graph: Graph, partition: Partition,
                          vertex: int, new_part: int) -> int:
        """Вычисление gain от перемещения"""
        old_part = partition.get_part(vertex)
        if old_part == new_part or old_part == -1:
            return 0
        
        gain = 0
        for neighbor, weight in graph.get_neighbors(vertex).items():
            neighbor_part = partition.get_part(neighbor)
            if neighbor_part == old_part:
                gain -= weight
            elif neighbor_part == new_part:
                gain += weight
        
        return gain