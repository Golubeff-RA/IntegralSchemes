# test_coarse_graph.py
import sys
sys.path.append('.')

from core.graph import Graph
from core.coarse_graph import CoarseGraph


def test_coarse_graph():
    """Тест создания грубого графа"""
    
    print("=" * 60)
    print("ТЕСТ ГРУБОГО ГРАФА")
    print("=" * 60)
    
    # Создаём исходный граф
    g = Graph(8)
    edges = [
        (0, 1, 5), (0, 2, 3), (1, 3, 2), (2, 3, 4),
        (3, 4, 1), (4, 5, 6), (4, 6, 2), (5, 7, 3), (6, 7, 4)
    ]
    for u, v, w in edges:
        g.add_edge(u, v, w)
    
    # Устанавливаем веса вершин
    for v in range(8):
        g.set_vertex_weight(v, v + 1)  # веса 1..8
    
    print(f"Исходный граф: {g}")
    print(f"  Веса вершин: {[g.get_vertex_weight(v) for v in range(8)]}")
    
    # Создаём паросочетание
    matching = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    # Создаём грубый граф
    coarse = CoarseGraph.from_matching(g, matching)
    
    print(f"\nГрубый граф: {coarse}")
    info = coarse.get_info()
    print(f"  Информация: {info}")
    print(f"  Веса грубых вершин: {[coarse.get_vertex_weight(v) for v in range(coarse.num_vertices)]}")
    
    # Выводим рёбра грубого графа
    print("\n  Рёбра грубого графа:")
    for u in range(coarse.num_vertices):
        for v, w in coarse.get_neighbors(u).items():
            if u < v:
                print(f"    {u} -- {v} (вес={w})")
    
    # Проверяем маппинг
    print("\n  Маппинг исходных вершин:")
    for v in range(8):
        cv = coarse.get_coarse_vertex(v)
        print(f"    {v} -> {cv}")
    
    # Преобразуем обратно в обычный граф
    g2 = coarse.to_graph()
    print(f"\nПреобразованный граф: {g2}")
    print(f"  Веса вершин: {[g2.get_vertex_weight(v) for v in range(g2.num_vertices)]}")
    
    return g, coarse


def test_expand_partition():
    """Тест проекции разбиения"""
    
    print("\n" + "=" * 60)
    print("ТЕСТ ПРОЕКЦИИ РАЗБИЕНИЯ")
    print("=" * 60)
    
    # Создаём исходный граф
    g = Graph(8)
    for i in range(7):
        g.add_edge(i, i + 1, 1)
    
    # Создаём грубый граф (стягиваем пары)
    matching = [(0, 1), (2, 3), (4, 5)]
    coarse = CoarseGraph.from_matching(g, matching)
    
    # Создаём разбиение грубого графа
    from core.partition import Partition
    partition = Partition(coarse.num_vertices, 2)
    partition.assign(0, 0)
    partition.assign(1, 0)
    partition.assign(2, 1)
    partition.assign(3, 1)
    partition.assign(4, 0)  # вершина 4 (из пары 6,7)
    
    print(f"Разбиение грубого графа: {partition.part_sizes}")
    
    # Проецируем на исходный граф
    original_partition = coarse.expand_partition(partition)
    
    print(f"Разбиение исходного графа ({original_partition.num_vertices} вершин):")
    print(f"  Part 0: {original_partition.get_vertices_in_part(0)}")
    print(f"  Part 1: {original_partition.get_vertices_in_part(1)}")
    
    # Проверяем, что все вершины назначены
    for v in range(original_partition.num_vertices):
        assert original_partition.get_part(v) != -1, f"Vertex {v} not assigned!"
    
    print("✅ Все вершины назначены!")


if __name__ == "__main__":
    test_coarse_graph()
    test_expand_partition()