"""
Генератор графов с явной кластерной структурой (сообществами)
Позволяет тестировать, находит ли алгоритм естественные разбиения
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
    Генератор графов с кластерной структурой

    Создаёт несколько плотных кластеров (почти полные подграфы),
    которые слабо связаны между собой. Идеально подходит для тестирования
    алгоритмов разбиения.

    Параметры:
        num_clusters: количество кластеров
        cluster_size: размер каждого кластера (может быть списком для разного размера)
        intra_prob: вероятность связи внутри кластера (0.8-1.0)
        inter_prob: вероятность связи между кластерами (0.0-0.1)
        weight_range: диапазон весов рёбер (min, max)
    """

    def __init__(
        self,
        num_clusters: int = 5,
        cluster_size: int = 20,
        intra_prob: float = 0.8,
        inter_prob: float = 0.05,
        weight_range: Tuple[int, int] = (1, 1),
        seed: Optional[int] = None,
    ):
        """
        Инициализация генератора кластерных графов

        Args:
            num_clusters: количество кластеров
            cluster_size: размер каждого кластера
            intra_prob: вероятность ребра внутри кластера
            inter_prob: вероятность ребра между кластерами
            weight_range: диапазон весов рёбер (min, max)
            seed: seed для воспроизводимости
        """
        super().__init__(name="ClusterGraphGenerator")

        self.num_clusters = num_clusters
        self.cluster_size = cluster_size
        self.intra_prob = intra_prob
        self.inter_prob = inter_prob
        self.weight_range = weight_range
        self.seed = seed

        self._generation_params = {
            "num_clusters": num_clusters,
            "cluster_size": cluster_size,
            "intra_prob": intra_prob,
            "inter_prob": inter_prob,
            "weight_range": weight_range,
            "seed": seed,
        }

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(self) -> Graph:
        """
        Генерация графа с кластерной структурой

        Returns:
            Graph: сгенерированный граф
        """
        total_vertices = self.num_clusters * self.cluster_size
        graph = Graph(total_vertices)

        # Предварительно создаём все вершины
        for v in range(total_vertices):
            graph.set_vertex_weight(v, 1)

        # 1. Добавляем внутрикластерные связи
        self._add_intra_cluster_edges(graph)

        # 2. Добавляем межкластерные связи
        self._add_inter_cluster_edges(graph)

        return graph

    def _add_intra_cluster_edges(self, graph: Graph) -> None:
        """Добавление рёбер внутри кластеров"""
        for c in range(self.num_clusters):
            start_idx = c * self.cluster_size
            vertices = list(range(start_idx, start_idx + self.cluster_size))

            # Для полного графа используем вероятностный подход
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    if random.random() < self.intra_prob:
                        weight = random.randint(*self.weight_range)
                        graph.add_edge(vertices[i], vertices[j], weight)

    def _add_inter_cluster_edges(self, graph: Graph) -> None:
        """Добавление рёбер между разными кластерами"""
        # Для каждой пары кластеров
        for c1 in range(self.num_clusters):
            for c2 in range(c1 + 1, self.num_clusters):
                # Определяем количество связей между кластерами
                # (примерно inter_prob * cluster_size^2)
                expected_edges = self.inter_prob * self.cluster_size * self.cluster_size
                num_edges = int(np.random.poisson(expected_edges))

                # Случайно выбираем вершины для соединения
                vertices_c1 = list(range(c1 * self.cluster_size, (c1 + 1) * self.cluster_size))
                vertices_c2 = list(range(c2 * self.cluster_size, (c2 + 1) * self.cluster_size))

                for _ in range(num_edges):
                    u = random.choice(vertices_c1)
                    v = random.choice(vertices_c2)
                    weight = random.randint(*self.weight_range)
                    graph.add_edge(u, v, weight)

    def generate_with_imbalance(self, imbalance_ratio: float = 0.3) -> Graph:
        """
        Генерация графа с кластерами разного размера (несбалансированными)

        Args:
            imbalance_ratio: коэффициент дисбаланса (0-1)

        Returns:
            Graph: граф с кластерами разного размера
        """
        # Создаём кластеры разного размера
        base_size = self.cluster_size
        cluster_sizes = []

        for i in range(self.num_clusters):
            # Размер кластера варьируется от base_size*(1-imbalance) до base_size*(1+imbalance)
            size_variation = 1 + imbalance_ratio * (2 * random.random() - 1)
            size = max(3, int(base_size * size_variation))
            cluster_sizes.append(size)

        total_vertices = sum(cluster_sizes)
        graph = Graph(total_vertices)

        # Создаём маппинг кластеров на вершины
        vertex_offset = 0
        cluster_vertices = []

        for c, size in enumerate(cluster_sizes):
            vertices = list(range(vertex_offset, vertex_offset + size))
            cluster_vertices.append(vertices)
            vertex_offset += size

        # Добавляем внутрикластерные связи
        for vertices in cluster_vertices:
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    if random.random() < self.intra_prob:
                        weight = random.randint(*self.weight_range)
                        graph.add_edge(vertices[i], vertices[j], weight)

        # Добавляем межкластерные связи
        for i in range(self.num_clusters):
            for j in range(i + 1, self.num_clusters):
                expected_edges = self.inter_prob * len(cluster_vertices[i]) * len(cluster_vertices[j])
                num_edges = int(np.random.poisson(expected_edges))

                for _ in range(num_edges):
                    u = random.choice(cluster_vertices[i])
                    v = random.choice(cluster_vertices[j])
                    weight = random.randint(*self.weight_range)
                    graph.add_edge(u, v, weight)

        return graph


class HierarchicalClusterGenerator(BaseGraphGenerator):
    """
    Генератор иерархических кластеров (кластеры внутри кластеров)
    Позволяет тестировать многоуровневые алгоритмы
    """

    def __init__(
        self,
        hierarchy_levels: int = 2,
        branching_factor: int = 3,
        cluster_size: int = 10,
        intra_prob: float = 0.7,
        inter_prob: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Инициализация генератора иерархических кластеров

        Args:
            hierarchy_levels: количество уровней иерархии
            branching_factor: количество подкластеров на уровне
            cluster_size: размер базовых кластеров
            intra_prob: вероятность связи внутри кластера
            inter_prob: вероятность связи между кластерами одного уровня
            seed: seed для воспроизводимости
        """
        super().__init__(name="HierarchicalClusterGenerator")

        self.hierarchy_levels = hierarchy_levels
        self.branching_factor = branching_factor
        self.cluster_size = cluster_size
        self.intra_prob = intra_prob
        self.inter_prob = inter_prob
        self.seed = seed

        self._generation_params = {
            "hierarchy_levels": hierarchy_levels,
            "branching_factor": branching_factor,
            "cluster_size": cluster_size,
            "intra_prob": intra_prob,
            "inter_prob": inter_prob,
            "seed": seed,
        }

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(self) -> Graph:
        """Генерация иерархического кластерного графа"""
        # Создаём базовые кластеры
        num_base_clusters = self.branching_factor ** (self.hierarchy_levels - 1)
        total_vertices = num_base_clusters * self.cluster_size
        graph = Graph(total_vertices)

        # Создаём структуру кластеров
        clusters = self._create_cluster_hierarchy()

        # Добавляем рёбра внутри базовых кластеров
        for cluster in clusters["base"]:
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    if random.random() < self.intra_prob:
                        graph.add_edge(cluster[i], cluster[j])

        # Добавляем рёбра между кластерами на разных уровнях
        for level in range(1, self.hierarchy_levels + 1):
            level_clusters = clusters[f"level_{level}"]
            self._add_inter_cluster_edges_level(graph, level_clusters)

        return graph

    def _create_cluster_hierarchy(self):
        """Создание иерархической структуры кластеров"""
        clusters = {"base": []}

        # Создаём базовые кластеры
        vertex_id = 0
        base_clusters = []

        for _ in range(self.branching_factor ** (self.hierarchy_levels - 1)):
            cluster = list(range(vertex_id, vertex_id + self.cluster_size))
            base_clusters.append(cluster)
            clusters["base"].append(cluster)
            vertex_id += self.cluster_size

        # Создаём кластеры верхних уровней
        current_clusters = base_clusters

        for level in range(1, self.hierarchy_levels + 1):
            level_clusters = []
            for i in range(0, len(current_clusters), self.branching_factor):
                merged = []
                for j in range(self.branching_factor):
                    if i + j < len(current_clusters):
                        merged.extend(current_clusters[i + j])
                if merged:
                    level_clusters.append(merged)

            clusters[f"level_{level}"] = level_clusters
            current_clusters = level_clusters

        return clusters

    def _add_inter_cluster_edges_level(self, graph: Graph, clusters: List[List[int]]):
        """Добавление рёбер между кластерами одного уровня"""
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                expected_edges = self.inter_prob * len(clusters[i]) * len(clusters[j])
                num_edges = int(np.random.poisson(expected_edges))

                for _ in range(num_edges):
                    u = random.choice(clusters[i])
                    v = random.choice(clusters[j])
                    graph.add_edge(u, v)
