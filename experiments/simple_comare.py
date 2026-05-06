#!/usr/bin/env python3
"""
Простое сравнение KL и Multilevel на разреженных графах
"""

import sys
import time
import random
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from algorithms.kernighan_lin import KernighanLin
from algorithms.multilevel import FastMultilevelPartitioner


def create_sparse_cluster_graph(n: int, num_clusters: int = None, seed: int = 42) -> Graph:
    """
    Создание разреженного кластерного графа
    """
    random.seed(seed)
    
    if num_clusters is None:
        num_clusters = max(2, n // 20)
    
    vertices_per_cluster = n // num_clusters
    graph = Graph(n)
    
    # Внутрикластерные связи - каждая вершина соединена с несколькими соседями
    for c in range(num_clusters):
        start = c * vertices_per_cluster
        end = min(start + vertices_per_cluster, n)
        vertices = list(range(start, end))
        
        # Создаём простую цепочку (1-2 связи на вершину)
        for i in range(len(vertices) - 1):
            graph.add_edge(vertices[i], vertices[i+1], 1)
        
        # Добавляем несколько случайных связей внутри кластера
        extra_edges = min(len(vertices) // 2, 5)
        for _ in range(extra_edges):
            u = random.choice(vertices)
            v = random.choice(vertices)
            if u != v and not graph.has_edge(u, v):
                graph.add_edge(u, v, 1)
    
    # Межкластерные связи - редкие мосты
    for c in range(num_clusters - 1):
        start1 = c * vertices_per_cluster
        start2 = (c + 1) * vertices_per_cluster
        end1 = min(start1 + vertices_per_cluster, n)
        end2 = min(start2 + vertices_per_cluster, n)
        
        # 2-3 моста между соседними кластерами
        for _ in range(3):
            u = random.randrange(start1, end1)
            v = random.randrange(start2, end2)
            if not graph.has_edge(u, v):
                graph.add_edge(u, v, 1)
    
    return graph


def create_sparse_graph(n: int, edge_prob: float = 0.02, seed: int = 42) -> Graph:
    """
    Создание случайного разреженного графа
    """
    random.seed(seed)
    graph = Graph(n)
    
    expected_edges = int(n * (n-1) / 2 * edge_prob)
    edges_added = 0
    
    while edges_added < expected_edges and edges_added < n * 5:
        u = random.randrange(n)
        v = random.randrange(n)
        if u != v and not graph.has_edge(u, v):
            graph.add_edge(u, v, random.randint(1, 3))
            edges_added += 1
    
    return graph


def run_comparison():
    """Запуск сравнения"""
    
    sizes = [100, 200, 500]
    results = []
    
    print("=" * 70)
    print("KL vs Multilevel Comparison")
    print("=" * 70)
    
    for n in sizes:
        print(f"\n{'='*50}")
        print(f"Graph size: {n} vertices")
        print(f"{'='*50}")
        
        # Создаём граф
        graph = create_sparse_cluster_graph(n, seed=42)
        print(f"Edges: {graph.num_edges} (density: {2*graph.num_edges/(n*(n-1)):.6f})")
        
        # KL алгоритм
        print("\n--- Kernighan-Lin ---")
        kl = KernighanLin(max_passes=15, seed=42)
        
        start = time.time()
        partition_kl, metrics_kl = kl.partition(graph)
        kl_time = time.time() - start
        
        print(f"  Cut: {metrics_kl.cut_weight}")
        print(f"  Time: {kl_time:.4f}s")
        print(f"  Balance: {partition_kl.balance_quality():.4f}")
        
        # Fast Multilevel
        print("\n--- Fast Multilevel ---")
        ml = FastMultilevelPartitioner(
            min_coarse_vertices=max(20, n // 10),
            refinement_passes=1,
            seed=42
        )
        
        start = time.time()
        partition_ml, metrics_ml = ml.partition(graph)
        ml_time = time.time() - start
        
        print(f"  Cut: {metrics_ml.cut_weight}")
        print(f"  Time: {ml_time:.4f}s")
        print(f"  Balance: {partition_ml.balance_quality():.4f}")
        
        # Сравнение
        improvement = (metrics_kl.cut_weight - metrics_ml.cut_weight) / metrics_kl.cut_weight * 100
        speedup = kl_time / max(0.001, ml_time)
        
        print(f"\n--- Result for n={n} ---")
        print(f"  Improvement: {improvement:+.2f}%")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Winner: {'Multilevel' if improvement > 0 else 'KL'}")
        
        results.append({
            'n': n,
            'edges': graph.num_edges,
            'kl_cut': metrics_kl.cut_weight,
            'kl_time': kl_time,
            'ml_cut': metrics_ml.cut_weight,
            'ml_time': ml_time,
            'improvement': improvement,
            'speedup': speedup
        })
    
    # Общая сводка
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Size':<8} {'Edges':<10} {'KL cut':<12} {'ML cut':<12} {'Improve':<12} {'Speedup':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['n']:<8} {r['edges']:<10} {r['kl_cut']:<12} {r['ml_cut']:<12} {r['improvement']:>+10.2f}%   {r['speedup']:>8.2f}x")
    
    # Средние значения
    avg_improvement = sum(r['improvement'] for r in results) / len(results)
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    
    print("-" * 70)
    print(f"{'Average':<8} {'':<10} {'':<12} {'':<12} {avg_improvement:>+10.2f}%   {avg_speedup:>8.2f}x")
    print("=" * 70)


def test_small():
    """Тест на маленьком графе с известной структурой"""
    print("=" * 70)
    print("TEST ON SMALL GRAPH")
    print("=" * 70)
    
    # Создаём граф-путь (оптимальный разрез = 1)
    n = 10
    graph = Graph(n)
    for i in range(n-1):
        graph.add_edge(i, i+1, 1)
    
    print(f"Graph: path with {n} vertices, {graph.num_edges} edges")
    print("Optimal cut should be 1 (split in the middle)")
    
    # KL
    kl = KernighanLin(max_passes=10, seed=42)
    p_kl, m_kl = kl.partition(graph)
    print(f"\nKL: cut={m_kl.cut_weight}, size0={p_kl.size0}, size1={p_kl.size1}")
    
    # Multilevel
    ml = FastMultilevelPartitioner(min_coarse_vertices=3, refinement_passes=1, seed=42)
    p_ml, m_ml = ml.partition(graph)
    print(f"ML: cut={m_ml.cut_weight}, size0={p_ml.size0}, size1={p_ml.size1}")
    
    print(f"\nImprovement: {(m_kl.cut_weight - m_ml.cut_weight) / m_kl.cut_weight * 100:+.2f}%")


if __name__ == "__main__":
    test_small()
    print("\n" + "=" * 70 + "\n")
    run_comparison()