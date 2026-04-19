"""
Класс графа с удобным API для алгоритмов разбиения
Поддерживает веса вершин и рёбер
"""

from typing import List, Tuple, Dict, Set, Optional, Iterator
from collections import defaultdict
import numpy as np


class Graph:
    """
    Неориентированный граф с весами вершин и рёбер

    Внутреннее представление:
    - Список смежности с весами рёбер
    - Отдельный массив для весов вершин

    Пример использования:
        g = Graph(5)
        g.add_edge(0, 1, weight=2)
        g.add_edge(1, 2)
        g.set_vertex_weight(0, 10)

        for u, v, w in g.edges():
            print(f"{u} -- {v} (weight={w})")
    """

    def __init__(self, num_vertices: int = 0):
        """
        Инициализация пустого графа

        Args:
            num_vertices: количество вершин (можно добавить позже)
        """
        self._num_vertices = num_vertices
        self._num_edges = 0

        # Список смежности: для каждой вершины словарь {сосед: вес}
        self._adjacency: List[Dict[int, int]] = [{} for _ in range(num_vertices)]

        # Веса вершин (по умолчанию 1)
        self._vertex_weights: np.ndarray = np.ones(num_vertices, dtype=np.int32)

        # Мета-информация
        self._name: str = ""

    @property
    def num_vertices(self) -> int:
        """Количество вершин"""
        return self._num_vertices

    @property
    def num_edges(self) -> int:
        """Количество рёбер"""
        return self._num_edges

    @property
    def vertex_weights(self) -> np.ndarray:
        """Массив весов вершин (только для чтения)"""
        return self._vertex_weights.copy()

    def add_vertex(self) -> int:
        """
        Добавление новой вершины

        Returns:
            Индекс новой вершины
        """
        self._adjacency.append({})
        self._num_vertices += 1
        self._vertex_weights = np.append(self._vertex_weights, 1)
        return self._num_vertices - 1

    def add_edge(self, u: int, v: int, weight: int = 1) -> None:
        """
        Добавление ребра (неориентированного)

        Args:
            u, v: вершины (индексация с 0)
            weight: вес ребра (по умолчанию 1)
        """
        if u < 0 or u >= self._num_vertices or v < 0 or v >= self._num_vertices:
            raise IndexError(f"Vertex index out of range: {u}, {v}")

        if u == v:
            return  # Не добавляем петли

        # Проверяем, существует ли уже ребро
        if v in self._adjacency[u]:
            return

        # Добавляем в обе стороны
        self._adjacency[u][v] = weight
        self._adjacency[v][u] = weight
        self._num_edges += 1

    def remove_edge(self, u: int, v: int) -> bool:
        """
        Удаление ребра

        Returns:
            True если ребро существовало
        """
        if u < 0 or u >= self._num_vertices or v < 0 or v >= self._num_vertices:
            return False

        if v in self._adjacency[u]:
            del self._adjacency[u][v]
            del self._adjacency[v][u]
            self._num_edges -= 1
            return True
        return False

    def has_edge(self, u: int, v: int) -> bool:
        """Проверка существования ребра"""
        return v in self._adjacency[u]

    def get_edge_weight(self, u: int, v: int) -> int:
        """Вес ребра (0 если ребра нет)"""
        return self._adjacency[u].get(v, 0)

    def set_vertex_weight(self, v: int, weight: int) -> None:
        """Установка веса вершины"""
        if 0 <= v < self._num_vertices:
            self._vertex_weights[v] = weight

    def get_vertex_weight(self, v: int) -> int:
        """Получение веса вершины"""
        return self._vertex_weights[v]

    def get_neighbors(self, v: int) -> Dict[int, int]:
        """
        Получение всех соседей вершины с весами

        Returns:
            Словарь {сосед: вес}
        """
        return self._adjacency[v].copy()

    def get_degree(self, v: int) -> int:
        """Степень вершины (количество соседей)"""
        return len(self._adjacency[v])

    def vertices(self) -> Iterator[int]:
        """Итератор по всем вершинам"""
        yield from range(self._num_vertices)

    def edges(self) -> Iterator[Tuple[int, int, int]]:
        """
        Итератор по всем рёбрам

        Yields:
            (u, v, weight) для каждого ребра (каждое ребро один раз)
        """
        for u in range(self._num_vertices):
            for v, w in self._adjacency[u].items():
                if u < v:  # Каждое ребро только один раз
                    yield (u, v, w)

    def get_adjacency_list(self) -> List[Dict[int, int]]:
        """Получение полного списка смежности (для алгоритмов)"""
        return self._adjacency

    def degree_sequence(self) -> List[int]:
        """Степени всех вершин"""
        return [len(adj) for adj in self._adjacency]

    def total_weight(self) -> int:
        """Суммарный вес всех вершин"""
        return int(np.sum(self._vertex_weights))

    def subgraph(self, vertices: Set[int]) -> "Graph":
        """
        Создание подграфа на заданном множестве вершин

        Args:
            vertices: множество вершин для подграфа

        Returns:
            Новый граф, содержащий только указанные вершины и рёбра между ними
        """
        # Создаём маппинг старых индексов на новые
        vertex_list = sorted(vertices)
        old_to_new = {old: new for new, old in enumerate(vertex_list)}

        sub_g = Graph(len(vertex_list))

        # Копируем веса вершин
        for old_v, new_v in old_to_new.items():
            sub_g.set_vertex_weight(new_v, self.get_vertex_weight(old_v))

        # Добавляем рёбра
        for u in vertices:
            for v, w in self._adjacency[u].items():
                if v in vertices and u < v:  # Каждое ребро один раз
                    sub_g.add_edge(old_to_new[u], old_to_new[v], w)

        return sub_g

    def save_to_file(self, filename: str) -> None:
        """
        Сохранение графа в текстовый файл

        Формат:
            <num_vertices> <num_edges>
            <u1> <v1> [weight1]
            <u2> <v2> [weight2]
            ...
        """
        with open(filename, "w") as f:
            # Заголовок
            f.write(f"{self._num_vertices} {self._num_edges}\n")

            # Сохраняем рёбра (нумерация с 1 для совместимости)
            for u, v, w in self.edges():
                if w != 1:
                    f.write(f"{u+1} {v+1} {w}\n")
                else:
                    f.write(f"{u+1} {v+1}\n")

    @staticmethod
    def load_from_file(filename: str, weighted_edges: bool = False) -> "Graph":
        """
        Загрузка графа из текстового файла

        Args:
            filename: путь к файлу
            weighted_edges: учитывать ли веса рёбер (если есть)

        Returns:
            Загруженный граф
        """
        with open(filename, "r") as f:
            # Читаем заголовок
            line = f.readline().strip()
            while line == "":
                line = f.readline().strip()

            parts = line.split()
            n = int(parts[0])
            m = int(parts[1])

            graph = Graph(n)

            # Читаем рёбра
            for _ in range(m):
                line = f.readline().strip()
                while line == "":
                    line = f.readline().strip()

                parts = line.split()
                u = int(parts[0]) - 1  # В файле нумерация с 1
                v = int(parts[1]) - 1

                if weighted_edges and len(parts) >= 3:
                    w = int(parts[2])
                    graph.add_edge(u, v, w)
                else:
                    graph.add_edge(u, v)

            return graph

    def __repr__(self) -> str:
        return f"Graph(n={self._num_vertices}, m={self._num_edges})"

    def __str__(self) -> str:
        return f"Graph with {self._num_vertices} vertices and {self._num_edges} edges"
