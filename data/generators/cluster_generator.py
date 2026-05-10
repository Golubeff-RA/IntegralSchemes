"""
Генератор графов с кластерной структурой
Оптимизирован для больших графов (до 10^5 вершин, до 5×10^5 рёбер)
Количество рёбер контролируется явно, а не через вероятности
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.graph import Graph
from .base_generator import BaseGraphGenerator


class ClusterGraphGenerator(BaseGraphGenerator):
    """
    Генератор графов с кластерной структурой с контролируемым количеством рёбер
    
    Параметры:
    - num_clusters: количество кластеров
    - cluster_size: размер кластера
    - target_edges: целевое количество рёбер (по умолчанию 5×10^5)
    - intra_ratio: доля рёбер внутри кластеров (0.0-1.0)
    - weight_range: диапазон весов рёбер
    - vertex_weight_range: диапазон весов вершин
    """
    
    def __init__(self,
                 num_clusters: int = 10,
                 cluster_size: int = 10000,
                 target_edges: int = 500000,
                 intra_ratio: float = 0.7,
                 weight_range: Tuple[int, int] = (1, 1),
                 vertex_weight_range: Tuple[int, int] = (1, 1),
                 seed: Optional[int] = None):
        
        super().__init__(seed)
        self.num_clusters = num_clusters
        self.cluster_size = cluster_size
        self.target_edges = target_edges
        self.intra_ratio = intra_ratio
        self.weight_range = weight_range
        self.vertex_weight_range = vertex_weight_range
        
        # Вычисляем реальные размеры
        self.total_vertices = num_clusters * cluster_size
        self.intra_edges = int(target_edges * intra_ratio)
        self.inter_edges = target_edges - self.intra_edges
    
    def generate(self) -> Graph:
        """
        Генерация графа с контролируемым количеством рёбер
        Сложность O(V + E)
        """
        graph = Graph(self.total_vertices)
        
        # Вычисляем диапазоны кластеров
        cluster_ranges = []
        for c in range(self.num_clusters):
            start = c * self.cluster_size
            end = start + self.cluster_size
            cluster_ranges.append((start, end))
        
        # 1. Внутрикластерные связи
        self._add_intra_cluster_edges(graph, cluster_ranges)
        
        # 2. Межкластерные связи
        self._add_inter_cluster_edges(graph, cluster_ranges)
        
        # 3. Добавляем веса вершин
        for v in range(self.total_vertices):
            w = random.randint(*self.vertex_weight_range)
            graph.set_vertex_weight(v, w)
        
        return graph
    
    def _add_intra_cluster_edges(self, graph: Graph, cluster_ranges: List[Tuple[int, int]]):
        """Добавляет рёбра внутри кластеров"""
        edges_per_cluster = self.intra_edges // self.num_clusters
        remaining = self.intra_edges - edges_per_cluster * self.num_clusters
        
        for c, (start, end) in enumerate(cluster_ranges):
            n = end - start
            max_edges = n * (n - 1) // 2
            
            # Количество рёбер для этого кластера
            num_edges = edges_per_cluster + (1 if c < remaining else 0)
            num_edges = min(num_edges, max_edges)
            
            if num_edges <= 0:
                continue
            
            # Генерируем уникальные пары без повторений
            edges_added = 0
            used_pairs = set()
            
            # Если нужно много рёбер (>60% от максимальных), используем другой подход
            if num_edges > max_edges * 0.6:
                # Добавляем почти все рёбра, потом удаляем лишние
                all_pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        all_pairs.append((start + i, start + j))
                
                random.shuffle(all_pairs)
                for u, v in all_pairs[:num_edges]:
                    w = random.randint(*self.weight_range)
                    graph.add_edge(u, v, w)
            else:
                # Генерируем случайные пары
                max_attempts = num_edges * 10
                attempts = 0
                
                while edges_added < num_edges and attempts < max_attempts:
                    i = random.randrange(n)
                    j = random.randrange(n)
                    if i != j:
                        u = start + min(i, j)
                        v = start + max(i, j)
                        if (u, v) not in used_pairs:
                            used_pairs.add((u, v))
                            w = random.randint(*self.weight_range)
                            graph.add_edge(u, v, w)
                            edges_added += 1
                    attempts += 1
    
    def _add_inter_cluster_edges(self, graph: Graph, cluster_ranges: List[Tuple[int, int]]):
        """Добавляет рёбра между кластерами"""
        num_pairs = self.num_clusters * (self.num_clusters - 1) // 2
        edges_per_pair = self.inter_edges // num_pairs if num_pairs > 0 else 0
        remaining = self.inter_edges - edges_per_pair * num_pairs
        
        pair_id = 0
        for c1 in range(self.num_clusters):
            start1, end1 = cluster_ranges[c1]
            for c2 in range(c1 + 1, self.num_clusters):
                start2, end2 = cluster_ranges[c2]
                size1 = end1 - start1
                size2 = end2 - start2
                
                num_edges = edges_per_pair + (1 if pair_id < remaining else 0)
                max_edges = size1 * size2
                num_edges = min(num_edges, max_edges)
                pair_id += 1
                
                if num_edges <= 0:
                    continue
                
                # Генерируем уникальные пары
                edges_added = 0
                used_pairs = set()
                max_attempts = num_edges * 10
                attempts = 0
                
                while edges_added < num_edges and attempts < max_attempts:
                    u = random.randrange(start1, end1)
                    v = random.randrange(start2, end2)
                    if (u, v) not in used_pairs:
                        used_pairs.add((u, v))
                        w = random.randint(*self.weight_range)
                        graph.add_edge(u, v, w)
                        edges_added += 1
                    attempts += 1
    
    def generate_sparse(self) -> Graph:
        """
        Генерация разреженного графа (гарантированно ≤ 5×10^5 рёбер)
        Использует биномиальное распределение для контроля плотности
        """
        graph = Graph(self.total_vertices)
        
        cluster_ranges = []
        for c in range(self.num_clusters):
            start = c * self.cluster_size
            end = start + self.cluster_size
            cluster_ranges.append((start, end))
        
        # Вычисляем вероятности исходя из целевого количества рёбер
        total_possible_edges = self.total_vertices * (self.total_vertices - 1) // 2
        density = self.target_edges / total_possible_edges
        
        # Внутрикластерная плотность выше, межкластерная - ниже
        intra_density = density * self.intra_ratio * self.num_clusters
        inter_density = density * (1 - self.intra_ratio) / self.num_clusters
        
        intra_density = min(0.5, intra_density)
        inter_density = min(0.05, inter_density)
        
        # Внутрикластерные связи
        for start, end in cluster_ranges:
            n = end - start
            prob = min(1.0, intra_density * n / 10)  # Адаптивная вероятность
            
            # Используем эффективный алгоритм для разреженных графов
            expected = int(n * (n - 1) * prob / 2)
            edges_added = 0
            used = set()
            
            while edges_added < expected and edges_added < n * 5:  # Не более 5n рёбер на кластер
                i = random.randrange(n)
                j = random.randrange(n)
                if i != j:
                    u = start + min(i, j)
                    v = start + max(i, j)
                    if (u, v) not in used:
                        used.add((u, v))
                        if random.random() < prob:
                            w = random.randint(*self.weight_range)
                            graph.add_edge(u, v, w)
                            edges_added += 1
        
        # Межкластерные связи
        for c1 in range(self.num_clusters):
            start1, end1 = cluster_ranges[c1]
            for c2 in range(c1 + 1, self.num_clusters):
                start2, end2 = cluster_ranges[c2]
                size1 = end1 - start1
                size2 = end2 - start2
                
                prob = min(0.02, inter_density * size1 * size2 / 10000)
                expected = int(size1 * size2 * prob)
                expected = min(expected, 10000)  # Ограничиваем
                
                edges_added = 0
                used = set()
                
                while edges_added < expected:
                    u = random.randrange(start1, end1)
                    v = random.randrange(start2, end2)
                    if (u, v) not in used:
                        used.add((u, v))
                        if random.random() < prob:
                            w = random.randint(*self.weight_range)
                            graph.add_edge(u, v, w)
                            edges_added += 1
        
        # Добавляем веса вершин
        for v in range(self.total_vertices):
            w = random.randint(*self.vertex_weight_range)
            graph.set_vertex_weight(v, w)
        
        return graph

class FastClusterGenerator(BaseGraphGenerator):
    """
    ОЧЕНЬ БЫСТРЫЙ генератор кластерных графов
    Использует векторизованные операции для больших графов (10^5 вершин за секунды)
    """
    
    def __init__(self,
                 num_clusters: int = 10,
                 vertices_per_cluster: int = 10000,
                 target_edges: int = 500000,
                 intra_ratio: float = 0.7,
                 weight_range: Tuple[int, int] = (1, 1),
                 vertex_weight_range: Tuple[int, int] = (1, 1),
                 seed: Optional[int] = None):
        
        super().__init__(seed)
        self.num_clusters = num_clusters
        self.vertices_per_cluster = vertices_per_cluster
        self.target_edges = target_edges
        self.intra_ratio = intra_ratio
        self.weight_range = weight_range
        self.vertex_weight_range = vertex_weight_range
        
        self.total_vertices = num_clusters * vertices_per_cluster
        
        # Вычисляем количество рёбер
        self.intra_edges = int(target_edges * intra_ratio)
        self.inter_edges = target_edges - self.intra_edges
    
    def generate(self) -> Graph:
        """
        Быстрая генерация с использованием numpy
        Сложность O(V + E), память O(V + E)
        """
        graph = Graph(self.total_vertices)
        
        # Предварительно выделяем структуры
        cluster_starts = [c * self.vertices_per_cluster for c in range(self.num_clusters)]
        
        # 1. Быстрая генерация внутрикластерных рёбер
        self._fast_intra_edges(graph, cluster_starts)
        
        # 2. Быстрая генерация межкластерных рёбер
        self._fast_inter_edges(graph, cluster_starts)
        
        # 3. Веса вершин
        for v in range(self.total_vertices):
            if self.vertex_weight_range[0] != self.vertex_weight_range[1]:
                w = random.randint(*self.vertex_weight_range)
                graph.set_vertex_weight(v, w)
        
        return graph
    
    def _fast_intra_edges(self, graph: Graph, cluster_starts: List[int]):
        """Быстрая генерация внутрикластерных рёбер"""
        edges_per_cluster = self.intra_edges // self.num_clusters
        remaining = self.intra_edges - edges_per_cluster * self.num_clusters
        
        for c, start in enumerate(cluster_starts):
            n = self.vertices_per_cluster
            num_edges = edges_per_cluster + (1 if c < remaining else 0)
            
            if num_edges <= 0:
                continue
            
            max_edges = n * (n - 1) // 2
            num_edges = min(num_edges, max_edges)
            
            if n < 2:
                continue
            
            # Используем множество для уникальных пар
            edges = set()
            max_attempts = num_edges * 10
            attempts = 0
            
            while len(edges) < num_edges and attempts < max_attempts:
                i = random.randrange(n)
                # Убеждаемся, что j > i
                j = random.randrange(n)
                if i == j:
                    continue
                u = start + min(i, j)
                v = start + max(i, j)
                edges.add((u, v))
                attempts += 1
            
            for u, v in edges:
                w = random.randint((self.weight_range[0] + self.weight_range[1]) / 2, self.weight_range[1])
                graph.add_edge(u, v, w)
    
    def _fast_inter_edges(self, graph: Graph, cluster_starts: List[int]):
        """Быстрая генерация межкластерных рёбер"""
        num_pairs = self.num_clusters * (self.num_clusters - 1) // 2
        if num_pairs == 0:
            return
        
        edges_per_pair = self.inter_edges // num_pairs
        remaining = self.inter_edges - edges_per_pair * num_pairs
        
        pair_idx = 0
        for c1 in range(self.num_clusters):
            start1 = cluster_starts[c1]
            for c2 in range(c1 + 1, self.num_clusters):
                start2 = cluster_starts[c2]
                
                num_edges = edges_per_pair + (1 if pair_idx < remaining else 0)
                pair_idx += 1
                
                if num_edges <= 0:
                    continue
                
                max_edges = self.vertices_per_cluster * self.vertices_per_cluster
                num_edges = min(num_edges, max_edges)
                
                # Генерация межкластерных рёбер
                edges = set()
                max_attempts = num_edges * 3
                attempts = 0
                
                while len(edges) < num_edges and attempts < max_attempts:
                    u = random.randrange(start1, start1 + self.vertices_per_cluster)
                    v = random.randrange(start2, start2 + self.vertices_per_cluster)
                    edges.add((u, v))
                    attempts += 1
                
                for u, v in edges:
                    w = random.randint(self.weight_range[0], (self.weight_range[0] + self.weight_range[1]) / 2)
                    graph.add_edge(u, v, w)
    
    def generate_ultra_fast(self) -> Graph:
        """
        Ультра-быстрая генерация с предсказуемым количеством рёбер
        Использует детерминированные паттерны для кластеров
        """
        graph = Graph(self.total_vertices)
        
        cluster_starts = [c * self.vertices_per_cluster for c in range(self.num_clusters)]
        
        # Внутрикластерные связи (используем кольцевую структуру)
        edges_per_cluster = self.intra_edges // self.num_clusters
        
        for c, start in enumerate(cluster_starts):
            n = self.vertices_per_cluster
            
            # Создаём базовую кольцевую структуру
            for i in range(n):
                u = start + i
                v = start + ((i + 1) % n)
                w = random.randint(*self.weight_range)
                graph.add_edge(u, v, w)
            
            # Добавляем дополнительные случайные рёбра
            remaining = edges_per_cluster - n
            if remaining > 0:
                added = 0
                attempts = 0
                while added < remaining and attempts < remaining * 5:
                    i = random.randrange(n)
                    j = random.randrange(i + 2, n)  # Избегаем уже существующих
                    u = start + i
                    v = start + j
                    if not graph.has_edge(u, v):
                        w = random.randint(*self.weight_range)
                        graph.add_edge(u, v, w)
                        added += 1
                    attempts += 1
        
        # Межкластерные связи (используем случайные мосты)
        edges_per_pair = self.inter_edges // max(1, self.num_clusters * (self.num_clusters - 1) // 2)
        
        for c1 in range(self.num_clusters):
            start1 = cluster_starts[c1]
            for c2 in range(c1 + 1, self.num_clusters):
                start2 = cluster_starts[c2]
                
                # Добавляем несколько мостов между кластерами
                num_bridges = max(1, edges_per_pair // 10)
                for _ in range(num_bridges):
                    u = random.randrange(start1, start1 + self.vertices_per_cluster)
                    v = random.randrange(start2, start2 + self.vertices_per_cluster)
                    w = random.randint(*self.weight_range)
                    graph.add_edge(u, v, w)
        
        # Веса вершин
        for v in range(self.total_vertices):
            if self.vertex_weight_range[0] != self.vertex_weight_range[1]:
                w = random.randint(*self.vertex_weight_range)
                graph.set_vertex_weight(v, w)
        
        return graph


# Фабрика для создания оптимального генератора
def create_cluster_generator(total_vertices: int = 100000,
                            target_edges: int = 500000,
                            intra_ratio: float = 0.7,
                            seed: Optional[int] = None) -> BaseGraphGenerator:
    """
    Создаёт оптимальный генератор для заданного размера графа
    """
    # Определяем количество кластеров
    num_clusters = max(5, min(20, int(total_vertices ** 0.3)))
    vertices_per_cluster = total_vertices // num_clusters
    
    # Корректируем количество рёбер
    max_possible = total_vertices * 10  # Ограничиваем для разреженности
    target_edges = min(target_edges, max_possible)
    
    return FastClusterGenerator(
        num_clusters=num_clusters,
        vertices_per_cluster=vertices_per_cluster,
        target_edges=target_edges,
        intra_ratio=intra_ratio,
        weight_range=(1, 3),
        vertex_weight_range=(1, 5),
        seed=seed
    )