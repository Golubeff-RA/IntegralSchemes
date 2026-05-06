import sys
from core.graph import Graph
from algorithms import MultilevelPartitioner, KernighanLin
from data.generators import ClusterGraphGenerator

print('='*60)
print('TEST 1: Cluster graph')
print('='*60)
gen = ClusterGraphGenerator(num_clusters=2, cluster_size=10000, intra_prob=0.005, inter_prob=0.0003, seed=42)
graph = gen.generate()
print(f'Graph: {graph.num_vertices} vertices, {graph.num_edges} edges')

kl = KernighanLin(max_passes=30)
p_kl = kl.partition(graph, 2)
cut_kl = p_kl.cut_edges(graph)


ml = MultilevelPartitioner(min_coarse_vertices=50, num_trials=1)
p_ml = ml.partition(graph, 2)
print(f"Coarsening_history: {ml.get_coarsening_history()}")
cut_ml = p_ml.cut_edges(graph)
print(f'KL cut: {cut_kl}')
print(f'ML cut: {cut_ml}')
print(f'KL balance: {p_kl.balance_quality()}, ML balance: {p_ml.balance_quality()}')
print(f'\nImprovement: {(cut_kl - cut_ml) / cut_kl * 100:.2f}%')