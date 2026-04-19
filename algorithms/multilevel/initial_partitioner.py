"""
Начальное разбиение на самом грубом графе
"""

import random
import numpy as np
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.graph import Graph
from core.partition import Partition
from ..kernighan_lin import KernighanLin


class InitialPartitioner:
    """
    Создание начального разбиения на грубом графе
    
    Стратегии:
    - Random: случайное разбиение
    - BFS: разбиение на основе BFS
    - Spectral: спектральное разбиение
    - Greedy: жадное разбиение
    """
    
    def __init__(self, strategy: str = 'random'):
        """
        Инициализация
        
        Args:
            strategy: стратегия начального разбиения
        """
        self.strategy = strategy
        self.kl_refiner = KernighanLin(max_passes=5)
    
    def partition(self, graph: Graph, num_parts: int = 2,
                  balance_ratio: float = 0.5) -> Partition:
        """
        Создание начального разбиения
        
        Args:
            graph: грубый граф
            num_parts: количество частей
            balance_ratio: допустимый дисбаланс
        
        Returns:
            Partition: начальное разбиение
        """
        if self.strategy == 'random':
            partition = self._random_partition(graph, num_parts, balance_ratio)
        elif self.strategy == 'bfs':
            partition = self._bfs_partition(graph, num_parts, balance_ratio)
        elif self.strategy == 'spectral':
            partition = self._spectral_partition(graph, num_parts, balance_ratio)
        elif self.strategy == 'greedy':
            partition = self._greedy_partition(graph, num_parts, balance_ratio)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Улучшаем разбиение KL алгоритмом
        partition = self.kl_refiner.partition(graph, num_parts, balance_ratio)
        
        return partition
    
    def _random_partition(self, graph: Graph, num_parts: int,
                          balance_ratio: float) -> Partition:
        """
        Случайное сбалансированное разбиение
        """
        n = graph.num_vertices
        partition = Partition(n, num_parts)
        
        # Вычисляем размер каждой части
        part_size = n // num_parts
        
        # Случайно перемешиваем вершины
        vertices = list(range(n))
        random.shuffle(vertices)
        
        # Распределяем
        for i, v in enumerate(vertices):
            part = min(i // part_size, num_parts - 1)
            partition.assign(v, part)
        
        return partition
    
    def _bfs_partition(self, graph: Graph, num_parts: int,
                       balance_ratio: float) -> Partition:
        """
        Разбиение на основе BFS от случайных стартовых вершин
        """
        n = graph.num_vertices
        partition = Partition(n, num_parts)
        
        # Выбираем стартовые вершины
        start_vertices = random.sample(range(n), num_parts)
        
        # BFS от каждой стартовой вершины
        assigned = set()
        for part, start in enumerate(start_vertices):
            # BFS
            queue = [start]
            visited = {start}
            
            while queue and len(assigned) < (part + 1) * (n // num_parts):
                v = queue.pop(0)
                if v not in assigned:
                    partition.assign(v, part)
                    assigned.add(v)
                    
                    for neighbor in graph.get_neighbors(v):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
        
        # Назначаем оставшиеся вершины
        for v in range(n):
            if v not in assigned:
                partition.assign(v, random.randint(0, num_parts - 1))
        
        return partition
    
    def _spectral_partition(self, graph: Graph, num_parts: int,
                            balance_ratio: float) -> Partition:
        """
        Спектральное разбиение (использует собственные векторы)
        Только для 2-х частей
        """
        if num_parts != 2:
            # Если больше 2 частей, используем рекурсивное разбиение
            return self._recursive_spectral(graph, num_parts, balance_ratio)
        
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
            import scipy.sparse as sp
            
            # Строим матрицу Лапласиана
            n = graph.num_vertices
            
            # Создаём разреженную матрицу смежности
            rows = []
            cols = []
            data = []
            
            for u, v, w in graph.edges():
                rows.append(u)
                cols.append(v)
                data.append(-w)
                rows.append(v)
                cols.append(u)
                data.append(-w)
            
            # Диагональ (степени)
            degrees = [sum(graph.get_neighbors(v).values()) for v in range(n)]
            for i in range(n):
                rows.append(i)
                cols.append(i)
                data.append(degrees[i])
            
            L = csr_matrix((data, (rows, cols)), shape=(n, n))
            
            # Находим второй наименьший собственный вектор (Fiedler)
            eigenvalues, eigenvectors = eigsh(L, k=2, which='SM')
            fiedler = eigenvectors[:, 1]
            
            # Разбиваем по знаку
            partition = Partition(n, 2)
            for v in range(n):
                part = 0 if fiedler[v] < 0 else 1
                partition.assign(v, part)
            
            return partition
            
        except ImportError:
            print("SciPy not available, falling back to random partitioning")
            return self._random_partition(graph, num_parts, balance_ratio)
    
    def _recursive_spectral(self, graph: Graph, num_parts: int,
                            balance_ratio: float) -> Partition:
        """
        Рекурсивное спектральное разбиение для k частей
        """
        if num_parts == 1:
            partition = Partition(graph.num_vertices, 1)
            for v in range(graph.num_vertices):
                partition.assign(v, 0)
            return partition
        
        # Разбиваем на 2 части
        partition_2 = self._spectral_partition(graph, 2, balance_ratio)
        
        # Создаём подграфы
        part0_vertices = set(partition_2.get_vertices_in_part(0))
        part1_vertices = set(partition_2.get_vertices_in_part(1))
        
        subgraph0 = graph.subgraph(part0_vertices)
        subgraph1 = graph.subgraph(part1_vertices)
        
        # Рекурсивно разбиваем каждую часть
        subpart0 = self._recursive_spectral(subgraph0, num_parts // 2, balance_ratio)
        subpart1 = self._recursive_spectral(subgraph1, num_parts - num_parts // 2, balance_ratio)
        
        # Объединяем результаты
        final_partition = Partition(graph.num_vertices, num_parts)
        
        # Маппинг вершин
        old_to_new0 = {old: new for new, old in enumerate(sorted(part0_vertices))}
        old_to_new1 = {old: new for new, old in enumerate(sorted(part1_vertices))}
        
        for v in part0_vertices:
            part = subpart0.get_part(old_to_new0[v])
            final_partition.assign(v, part)
        
        for v in part1_vertices:
            part = subpart1.get_part(old_to_new1[v]) + (num_parts // 2)
            final_partition.assign(v, part)
        
        return final_partition
    
    def _greedy_partition(self, graph: Graph, num_parts: int,
                          balance_ratio: float) -> Partition:
        """
        Жадное разбиение: последовательное добавление вершин в части
        """
        n = graph.num_vertices
        partition = Partition(n, num_parts)
        
        # Целевой размер каждой части
        target_size = n // num_parts
        
        # Сортируем вершины по степени (хабы распределяем равномерно)
        vertices = list(range(n))
        vertices.sort(key=lambda v: graph.get_degree(v), reverse=True)
        
        # Текущие размеры частей
        part_sizes = [0] * num_parts
        
        for v in vertices:
            # Выбираем часть с минимальным текущим размером
            min_part = np.argmin(part_sizes)
            partition.assign(v, min_part)
            part_sizes[min_part] += 1
        
        return partition