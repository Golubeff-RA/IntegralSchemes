# test_generators.py
import sys
sys.path.append('.')

from data.generators import ClusterGraphGenerator, BarabasiAlbertGenerator

print("=" * 60)
print("ГЕНЕРАЦИЯ КЛАСТЕРНОГО ГРАФА")
print("=" * 60)

gen = ClusterGraphGenerator(weight_range=(1, 500), vertex_weight_range=(1, 100))

graph = gen.generate()
stats = gen.get_stats(graph)

print(f"Вершин: {stats['num_vertices']}")
print(f"Рёбер: {stats['num_edges']}")
print(f"Плотность: {stats['density']:.6f}")
print(f"Средняя степень: {stats['avg_degree']:.2f}")
print(f"Диапазон весов вершин: [{min(stats['vertex_weights'])}, {max(stats['vertex_weights'])}]")
print(f"Диапазон весов рёбер: [{min(stats['edge_weights'])}, {max(stats['edge_weights'])}]")

print("\n" + "=" * 60)
print("ГЕНЕРАЦИЯ ГРАФА БАРАБАШИ-АЛЬБЕРТ")
print("=" * 60)

gen2 = BarabasiAlbertGenerator(
    n=10000,
    m0=10,
    m=3,
    weight_range=(1, 3),
    vertex_weight_range=(1, 5),
    seed=42
)

graph2 = gen2.generate()
stats2 = gen2.get_stats(graph2)

print(f"Вершин: {stats2['num_vertices']}")
print(f"Рёбер: {stats2['num_edges']}")
print(f"Средняя степень: {stats2['avg_degree']:.2f}")
print(f"Макс степень: {stats2['max_degree']}")