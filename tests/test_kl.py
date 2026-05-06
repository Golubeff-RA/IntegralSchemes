# test_kernighan_lin.py
import sys

sys.path.append(".")

from core.graph import Graph
from algorithms.kernighan_lin import KernighanLin, FastKernighanLin
from data.generators import ClusterGraphGenerator


def test_kernighan_lin():
    """Тест KL алгоритма"""

    print("=" * 60)
    print("ТЕСТ КЕРНИГАНА-ЛИНА")
    print("=" * 60)

    # Создаём тестовый граф
    gen = ClusterGraphGenerator(
        num_clusters=10,
        cluster_size=1000,
        target_edges=50000,
        intra_ratio=0.7,
        weight_range=(1, 10),
        vertex_weight_range=(1, 20),
    )
    graph = gen.generate()

    print(f"Граф: {graph.num_vertices} вершин, {graph.num_edges} рёбер")

    # KL алгоритм
    print("\n--- Kernighan-Lin ---")
    kl = KernighanLin(max_passes=10, seed=42)
    partition, metrics = kl.partition(graph, balance_ratio=0.5)

    print(f"  {metrics}")
    print(f"  Part 0: {partition.size0} вершин (вес={partition.weight0})")
    print(f"  Part 1: {partition.size1} вершин (вес={partition.weight1})")
    print(f"  Iterations: {metrics.extra.get('iterations', 0)}")

    # Fast KL
    print("\n--- Fast Kernighan-Lin ---")
    fkl = FastKernighanLin(max_passes=10, seed=42)
    partition2, metrics2 = fkl.partition(graph, balance_ratio=0.5)

    print(f"  {metrics2}")
    print(f"  Part 0: {partition2.size0} вершин (вес={partition2.weight0})")
    print(f"  Part 1: {partition2.size1} вершин (вес={partition2.weight1})")

    # Сравнение
    print("\n--- Сравнение ---")
    speedup = metrics.time_seconds / metrics2.time_seconds if metrics2.time_seconds > 0 else 0
    print(f"  Fast KL speedup: {speedup:.2f}x")
    print(f"  Quality difference: {metrics.cut_weight - metrics2.cut_weight:+d}")


if __name__ == "__main__":
    test_kernighan_lin()
