"""
Генератор графов по модели Барабаши-Альберт (масштабно-инвариантные графы)
Реализует preferential attachment - вершины с высокой степенью привлекают больше связей
"""

import random
import numpy as np
from typing import List, Optional

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from core.graph import Graph
from .base_generator import BaseGraphGenerator


class BarabasiAlbertGenerator(BaseGraphGenerator):
    """
    Генератор графов Барабаши-Альберт

    Характеристики:
    - Степени вершин распределены по степенному закону
    - Наличие "хабов" (вершин с очень высокой степенью)
    - Реалистичная модель для многих реальных сетей

    Параметры:
        n: количество вершин
        m0: начальное количество вершин (создаётся полный граф)
        m: количество рёбер для каждой новой вершины
        seed: seed для воспроизводимости
    """

    def __init__(
        self,
        n: int = 1000,
        m0: int = 5,
        m: int = 2,
        weighted: bool = False,
        weight_range: tuple = (1, 10),
        seed: Optional[int] = None,
    ):
        """
        Инициализация генератора Барабаши-Альберт

        Args:
            n: общее количество вершин
            m0: начальное количество вершин (должно быть >= m)
            m: количество рёбер для каждой новой вершины
            weighted: генерировать ли веса на рёбрах
            weight_range: диапазон весов рёбер (min, max)
            seed: seed для воспроизводимости
        """
        super().__init__(name="BarabasiAlbertGenerator")

        if m0 < m:
            raise ValueError(f"m0 ({m0}) must be >= m ({m})")

        self.n = n
        self.m0 = m0
        self.m = m
        self.weighted = weighted
        self.weight_range = weight_range
        self.seed = seed

        self._generation_params = {
            "n": n,
            "m0": m0,
            "m": m,
            "weighted": weighted,
            "weight_range": weight_range,
            "seed": seed,
        }

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(self) -> Graph:
        """
        Генерация графа по модели Барабаши-Альберт

        Returns:
            Graph: сгенерированный граф
        """
        graph = Graph(self.n)

        # 1. Начальный полный граф из m0 вершин
        for i in range(self.m0):
            for j in range(i + 1, self.m0):
                weight = self._get_random_weight()
                graph.add_edge(i, j, weight)

        # 2. Добавляем остальные вершины с preferential attachment
        degrees = [graph.get_degree(i) for i in range(self.m0)]

        for new_vertex in range(self.m0, self.n):
            # Выбираем m вершин для соединения с вероятностью пропорциональной степени
            targets = self._preferential_choice(degrees, self.m)

            for target in targets:
                weight = self._get_random_weight()
                graph.add_edge(new_vertex, target, weight)

            # Обновляем степени
            degrees.append(self.m)
            for target in targets:
                degrees[target] += 1

        return graph

    def _preferential_choice(self, degrees: List[int], k: int) -> List[int]:
        """
        Выбор k вершин с вероятностью пропорциональной степени

        Args:
            degrees: список степеней вершин
            k: количество вершин для выбора

        Returns:
            List[int]: выбранные вершины
        """
        total_degree = sum(degrees)
        if total_degree == 0:
            # Если все степени нулевые, выбираем равномерно
            return random.sample(range(len(degrees)), min(k, len(degrees)))

        # Вычисляем вероятности
        probabilities = [d / total_degree for d in degrees]

        # Выбираем без повторений
        chosen = set()
        while len(chosen) < k and len(chosen) < len(degrees):
            # Выбираем на основе вероятностей
            chosen.add(np.random.choice(len(degrees), p=probabilities))

        return list(chosen)

    def _get_random_weight(self) -> int:
        """Генерация случайного веса ребра"""
        if self.weighted:
            return random.randint(*self.weight_range)
        return 1


class ErdosRenyiGenerator(BaseGraphGenerator):
    """
    Генератор случайных графов Эрдёша-Реньи G(n, p)
    """

    def __init__(self, n: int = 1000, p: float = 0.01, seed: Optional[int] = None):
        """
        Инициализация генератора Эрдёша-Реньи

        Args:
            n: количество вершин
            p: вероятность существования ребра
            seed: seed для воспроизводимости
        """
        super().__init__(name="ErdosRenyiGenerator")

        self.n = n
        self.p = p
        self.seed = seed

        self._generation_params = {"n": n, "p": p, "seed": seed}

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(self) -> Graph:
        """Генерация случайного графа Эрдёша-Реньи"""
        graph = Graph(self.n)

        # Для каждой пары вершин с вероятностью p добавляем ребро
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if random.random() < self.p:
                    graph.add_edge(i, j)

        return graph


class PowerLawClusterGenerator(BaseGraphGenerator):
    """
    Генератор графов со степенным распределением кластеров
    Комбинация модели Барабаши-Альберт и кластерной структуры
    """

    def __init__(
        self,
        num_clusters: int = 10,
        power_law_exponent: float = 2.5,
        min_cluster_size: int = 10,
        max_cluster_size: int = 100,
        intra_density: float = 0.3,
        inter_density: float = 0.01,
        seed: Optional[int] = None,
    ):
        """
        Инициализация генератора

        Args:
            num_clusters: количество кластеров
            power_law_exponent: показатель степенного закона для размеров кластеров
            min_cluster_size: минимальный размер кластера
            max_cluster_size: максимальный размер кластера
            intra_density: плотность связей внутри кластера
            inter_density: плотность связей между кластерами
            seed: seed для воспроизводимости
        """
        super().__init__(name="PowerLawClusterGenerator")

        self.num_clusters = num_clusters
        self.exponent = power_law_exponent
        self.min_size = min_cluster_size
        self.max_size = max_cluster_size
        self.intra_density = intra_density
        self.inter_density = inter_density
        self.seed = seed

        self._generation_params = {
            "num_clusters": num_clusters,
            "exponent": power_law_exponent,
            "min_cluster_size": min_cluster_size,
            "max_cluster_size": max_cluster_size,
            "intra_density": intra_density,
            "inter_density": inter_density,
            "seed": seed,
        }

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(self) -> Graph:
        """Генерация графа со степенным распределением размеров кластеров"""
        # Генерируем размеры кластеров по степенному закону
        cluster_sizes = self._generate_power_law_sizes()

        # Создаём граф
        total_vertices = sum(cluster_sizes)
        graph = Graph(total_vertices)

        # Создаём кластеры
        vertex_offset = 0
        clusters = []

        for size in cluster_sizes:
            vertices = list(range(vertex_offset, vertex_offset + size))
            clusters.append(vertices)

            # Внутрикластерные связи (модель Эрдёша-Реньи)
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    if random.random() < self.intra_density:
                        graph.add_edge(vertices[i], vertices[j])

            vertex_offset += size

        # Межкластерные связи
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                expected_edges = self.inter_density * len(clusters[i]) * len(clusters[j])
                num_edges = int(np.random.poisson(expected_edges))

                for _ in range(num_edges):
                    u = random.choice(clusters[i])
                    v = random.choice(clusters[j])
                    graph.add_edge(u, v)

        return graph

    def _generate_power_law_sizes(self) -> List[int]:
        """Генерация размеров кластеров по степенному закону"""
        # Используем распределение Парето
        alpha = self.exponent - 1  # Параметр для распределения Парето

        sizes = []
        for _ in range(self.num_clusters):
            # Генерируем по степенному закону
            r = random.random()
            size = self.min_size * ((self.max_size / self.min_size) ** r)
            size = int(size ** (1 / alpha))  # Преобразование для степенного закона
            size = max(self.min_size, min(self.max_size, size))
            sizes.append(size)

        # Нормализуем, чтобы сумма была примерно как ожидается
        # (оставляем как есть, просто возвращаем)
        return sizes
