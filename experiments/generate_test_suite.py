#!/usr/bin/env python3
"""
Генерация тестового набора графов для сравнения алгоритмов разбиения
"""

import os
import sys
import random
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from data.generators import (
    ClusterGraphGenerator,
    FastClusterGenerator,
    BarabasiAlbertGenerator,
)


class TestSuiteGenerator:
    """Генератор тестового набора графов"""
    
    def __init__(self, output_dir: str = "data/test_suite", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.seed = seed
        random.seed(seed)
        
        # Создаём директории для каждого типа графов
        self.graph_types = [
            "cluster",
            "fast_cluster", 
            "barabasi_albert",
            "erdos_renyi",
            "complete",
            "path",
            "grid"
        ]
        
        for gt in self.graph_types:
            (self.output_dir / gt).mkdir(parents=True, exist_ok=True)
        
        # Размеры графов для генерации
        self.sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
        
        # Параметры для каждого типа
        self.params = {
            'cluster': {
                'intra_prob': [0.9, 0.8, 0.7, 0.6],
                'inter_prob': [0.1, 0.05, 0.03, 0.01]
            },
            'fast_cluster': {
                'intra_ratio': [0.9, 0.8, 0.7, 0.6],
                'target_edges_factor': [8, 6, 5, 4]  # рёбер на вершину
            },
            'barabasi_albert': {
                'm0': [5, 10, 15],
                'm_factor': [0.05, 0.03, 0.02]  # доля от n
            },
            'erdos_renyi': {
                'p': [0.1, 0.05, 0.03, 0.02, 0.01]
            }
        }
    
    def _save_graph(self, graph: Graph, filename: str):
        """Сохранение графа с весами вершин"""
        with open(filename, 'w') as f:
            # Заголовок: вершины, рёбра, флаг весов вершин
            f.write(f"{graph.num_vertices} {graph.num_edges} 1\n")
            
            # Веса вершин
            weights = [str(graph.get_vertex_weight(v)) for v in range(graph.num_vertices)]
            f.write(" ".join(weights) + "\n")
            
            # Рёбра
            for u, v, w in graph.edges():
                f.write(f"{u+1} {v+1} {w}\n")
    
    def generate_cluster_graph(self, n: int, variant: int = 0) -> Graph:
        """Генерация кластерного графа"""
        num_clusters = max(2, n // 20)
        cluster_size = n // num_clusters
        
        intra = self.params['cluster']['intra_prob'][variant % len(self.params['cluster']['intra_prob'])]
        inter = self.params['cluster']['inter_prob'][variant % len(self.params['cluster']['inter_prob'])]
        
        gen = ClusterGraphGenerator(
            num_clusters=num_clusters,
            cluster_size=cluster_size,
            intra_prob=intra,
            inter_prob=inter,
            weight_range=(1, 5),
            vertex_weight_range=(1, 10),
            seed=self.seed + variant
        )
        return gen.generate()
    
    def generate_fast_cluster_graph(self, n: int, variant: int = 0) -> Graph:
        """Генерация быстрого кластерного графа"""
        num_clusters = max(2, n // 20)
        vertices_per_cluster = n // num_clusters
        
        intra_ratio = self.params['fast_cluster']['intra_ratio'][variant % len(self.params['fast_cluster']['intra_ratio'])]
        target_edges = n * self.params['fast_cluster']['target_edges_factor'][variant % len(self.params['fast_cluster']['target_edges_factor'])]
        
        gen = FastClusterGenerator(
            num_clusters=num_clusters,
            vertices_per_cluster=vertices_per_cluster,
            target_edges=target_edges,
            intra_ratio=intra_ratio,
            weight_range=(1, 5),
            vertex_weight_range=(1, 10),
            seed=self.seed + variant
        )
        return gen.generate()
    
    def generate_barabasi_albert_graph(self, n: int, variant: int = 0) -> Graph:
        """Генерация графа Барабаши-Альберт"""
        m0 = self.params['barabasi_albert']['m0'][variant % len(self.params['barabasi_albert']['m0'])]
        m0 = min(m0, n // 2)
        m = max(1, int(n * self.params['barabasi_albert']['m_factor'][variant % len(self.params['barabasi_albert']['m_factor'])]))
        m = min(m, m0)
        
        gen = BarabasiAlbertGenerator(
            n=n,
            m0=m0,
            m=m,
            weight_range=(1, 5),
            vertex_weight_range=(1, 10),
            seed=self.seed + variant
        )
        return gen.generate()
    
    def generate_complete_graph(self, n: int, variant: int = 0) -> Graph:
        """Генерация полного графа K_n"""
        graph = Graph(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                w = random.randint(1, 5)
                graph.add_edge(i, j, w)
            graph.set_vertex_weight(i, random.randint(1, 10))
        
        return graph
    
    def generate_path_graph(self, n: int, variant: int = 0) -> Graph:
        """Генерация линейного графа-пути"""
        graph = Graph(n)
        
        for i in range(n - 1):
            w = random.randint(1, 3)
            graph.add_edge(i, i + 1, w)
            graph.set_vertex_weight(i, random.randint(1, 10))
        
        graph.set_vertex_weight(n - 1, random.randint(1, 10))
        
        return graph
    
    def generate_grid_graph(self, n: int, variant: int = 0) -> Graph:
        """Генерация 2D решётки (примерно sqrt(n) x sqrt(n))"""
        side = int(n ** 0.5)
        total = side * side
        graph = Graph(total)
        
        for i in range(side):
            for j in range(side):
                idx = i * side + j
                graph.set_vertex_weight(idx, random.randint(1, 10))
                
                # Связь вправо
                if j < side - 1:
                    w = random.randint(1, 3)
                    graph.add_edge(idx, idx + 1, w)
                
                # Связь вниз
                if i < side - 1:
                    w = random.randint(1, 3)
                    graph.add_edge(idx, idx + side, w)
        
        return graph
    
    def generate_all(self, max_size: int = 10000):
        """Генерация всех графов тестового набора"""
        results = {}
        
        print("=" * 70)
        print("GENERATING TEST SUITE")
        print("=" * 70)
        
        for graph_type in self.graph_types:
            print(f"\n--- {graph_type.upper()} graphs ---")
            results[graph_type] = {}
            
            for size in self.sizes:
                if size > max_size:
                    continue
                
                # Для полных графов ограничиваем размер (K_1000 уже 500k рёбер)
                if graph_type == "complete" and size > 500:
                    continue
                
                # Для решётки корректируем размер
                if graph_type == "grid":
                    side = int(size ** 0.5)
                    actual_size = side * side
                    if actual_size < 25:
                        continue
                else:
                    actual_size = size
                
                print(f"  Generating {graph_type} with {actual_size} vertices...", end=" ", flush=True)
                
                start = time.time()
                
                try:
                    if graph_type == "cluster":
                        graph = self.generate_cluster_graph(actual_size, variant=0)
                    elif graph_type == "fast_cluster":
                        graph = self.generate_fast_cluster_graph(actual_size, variant=0)
                    elif graph_type == "barabasi_albert":
                        graph = self.generate_barabasi_albert_graph(actual_size, variant=0)
                    elif graph_type == "complete":
                        graph = self.generate_complete_graph(actual_size, variant=0)
                    elif graph_type == "path":
                        graph = self.generate_path_graph(actual_size, variant=0)
                    elif graph_type == "grid":
                        graph = self.generate_grid_graph(actual_size, variant=0)
                    else:
                        continue
                    
                    elapsed = time.time() - start
                    
                    # Сохраняем в файл
                    filename = self.output_dir / graph_type / f"{graph_type}_{actual_size}.txt"
                    self._save_graph(graph, str(filename))
                    
                    results[graph_type][actual_size] = {
                        'vertices': graph.num_vertices,
                        'edges': graph.num_edges,
                        'time': elapsed,
                        'file': str(filename)
                    }
                    
                    print(f"✓ {graph.num_edges} edges, {elapsed:.2f}s")
                    
                except Exception as e:
                    print(f"✗ Error: {e}")
        
        return results
    
    def generate_variants(self, size: int = 200, graph_type: str = "cluster", num_variants: int = 5):
        """Генерация нескольких вариантов одного типа графа с разными параметрами"""
        print(f"\n--- Generating {num_variants} variants of {graph_type} graph (size={size}) ---")
        
        variants = []
        
        for i in range(num_variants):
            print(f"  Variant {i+1}...", end=" ", flush=True)
            
            start = time.time()
            
            if graph_type == "cluster":
                graph = self.generate_cluster_graph(size, variant=i)
            elif graph_type == "fast_cluster":
                graph = self.generate_fast_cluster_graph(size, variant=i)
            elif graph_type == "barabasi_albert":
                graph = self.generate_barabasi_albert_graph(size, variant=i)
            else:
                continue
            
            elapsed = time.time() - start
            
            filename = self.output_dir / graph_type / f"{graph_type}_{size}_v{i+1}.txt"
            self._save_graph(graph, str(filename))
            
            variants.append({
                'variant': i+1,
                'vertices': graph.num_vertices,
                'edges': graph.num_edges,
                'time': elapsed,
                'file': str(filename)
            })
            
            print(f"✓ {graph.num_edges} edges, {elapsed:.2f}s")
        
        return variants
    
    def print_summary(self, results: dict):
        """Вывод сводки сгенерированных графов"""
        print("\n" + "=" * 70)
        print("GENERATION SUMMARY")
        print("=" * 70)
        
        for graph_type, sizes in results.items():
            print(f"\n{graph_type.upper()}:")
            print(f"  {'Size':<10} {'Vertices':<12} {'Edges':<12} {'Time (s)':<10}")
            print(f"  {'-'*50}")
            
            for size, info in sorted(sizes.items()):
                print(f"  {size:<10} {info['vertices']:<12} {info['edges']:<12} {info['time']:<10.2f}")
    
    def generate_readme(self, results: dict):
        """Генерация README для тестового набора"""
        readme_path = self.output_dir / "README.md"
        
        with open(readme_path, 'w') as f:
            f.write("# Test Suite for Graph Partitioning\n\n")
            f.write("## Format\n\n")
            f.write("```\n")
            f.write("<num_vertices> <num_edges> 1\n")
            f.write("<vertex_weight_1> <vertex_weight_2> ... <vertex_weight_n>\n")
            f.write("<u1> <v1> <edge_weight_1>\n")
            f.write("...\n")
            f.write("```\n\n")
            f.write("## Generated Graphs\n\n")
            
            for graph_type, sizes in results.items():
                f.write(f"### {graph_type.upper()}\n\n")
                f.write("| Size | Vertices | Edges | File |\n")
                f.write("|------|----------|-------|------|\n")
                
                for size, info in sorted(sizes.items()):
                    f.write(f"| {size} | {info['vertices']} | {info['edges']} | `{graph_type}/{graph_type}_{size}.txt` |\n")
                
                f.write("\n")
            
            f.write("\n## Usage\n\n")
            f.write("```python\n")
            f.write("from core.graph import Graph\n\n")
            f.write("graph = Graph.load_from_file('data/test_suite/cluster/cluster_100.txt')\n")
            f.write("print(f'Vertices: {graph.num_vertices}')\n")
            f.write("print(f'Edges: {graph.num_edges}')\n")
            f.write("print(f'Vertex weights: {[graph.get_vertex_weight(v) for v in range(5)]}')\n")
            f.write("```\n")
        
        print(f"\nREADME saved to {readme_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate test suite of graphs')
    parser.add_argument('--output', type=str, default='data/test_suite',
                       help='Output directory')
    parser.add_argument('--max-size', type=int, default=5000,
                       help='Maximum graph size to generate')
    parser.add_argument('--variants', action='store_true',
                       help='Generate multiple variants of each graph type')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    generator = TestSuiteGenerator(output_dir=args.output, seed=args.seed)
    
    if args.variants:
        for graph_type in ['cluster', 'fast_cluster', 'barabasi_albert', 'erdos_renyi']:
            generator.generate_variants(size=200, graph_type=graph_type, num_variants=5)
    else:
        results = generator.generate_all(max_size=args.max_size)
        generator.print_summary(results)
        generator.generate_readme(results)
    
    print(f"\n✓ Test suite saved to {generator.output_dir}")


if __name__ == "__main__":
    main()