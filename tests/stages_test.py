# test_coarsening.py
import sys
sys.path.append('.')

from core.graph import Graph
from algorithms.multilevel_slow import MultilevelPartitioner
from data.generators import FastClusterGenerator, ClusterGraphGenerator

print("Testing coarsening...")

# Генерируем граф
gen = FastClusterGenerator(
    num_clusters=4,
    vertices_per_cluster=30,
    target_edges=700,
    intra_ratio=0.8,
    seed=42
)
graph = gen.generate()
print(f"Graph: {graph.num_vertices} vertices, {graph.num_edges} edges")

# Запускаем Multilevel
ml = MultilevelPartitioner(min_coarse_vertices=20, max_levels=10)
partition, metrics = ml.partition(graph, balance_ratio=0.5)

print(f"Levels: {len(ml.levels)}")
for i, level in enumerate(ml.levels):
    print(f"  Level {i}: {level.graph.num_vertices} vertices, compression={level.compression_ratio:.3f}")