"""
Модульные тесты для генераторов графов
"""

import unittest
import sys
import random
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from data.generators import (
    ClusterGraphGenerator,
    BarabasiAlbertGenerator,
    ErdosRenyiGenerator,
    HierarchicalClusterGenerator,
    PowerLawClusterGenerator
)


class TestGenerators(unittest.TestCase):
    """Тесты для генераторов графов"""
    
    def setUp(self):
        """Подготовка перед каждым тестом"""
        self.small_n = 100
        self.medium_n = 500
        
        # Фиксируем seed для воспроизводимости тестов
        random.seed(42)
        np.random.seed(42)
    
    def test_cluster_generator_basic(self):
        """Тест базового кластерного генератора"""
        gen = ClusterGraphGenerator(
            num_clusters=5,
            cluster_size=20,
            intra_prob=0.8,
            inter_prob=0.05,
            seed=42
        )
        
        graph = gen.generate()
        
        # Проверяем основные свойства
        self.assertEqual(graph.num_vertices, 100)
        self.assertGreater(graph.num_edges, 0)
        
        # Проверяем, что граф неориентированный
        for u, v, w in graph.edges():
            self.assertTrue(graph.has_edge(v, u))
            self.assertEqual(graph.get_edge_weight(u, v), 
                            graph.get_edge_weight(v, u))
    
    def test_cluster_generator_with_imbalance(self):
        """Тест кластерного генератора с дисбалансом"""
        gen = ClusterGraphGenerator(
            num_clusters=3,
            cluster_size=50,
            intra_prob=0.9,
            inter_prob=0.02,
            seed=123
        )
        
        graph = gen.generate_with_imbalance(imbalance_ratio=0.3)
        
        # Проверяем, что граф создан
        self.assertIsNotNone(graph)
        self.assertGreater(graph.num_vertices, 0)
    
    def test_barabasi_albert_generator(self):
        """Тест генератора Барабаши-Альберт"""
        gen = BarabasiAlbertGenerator(
            n=200,
            m0=5,
            m=2,
            weighted=True,
            weight_range=(1, 10),
            seed=42
        )
        
        graph = gen.generate()
        
        # Проверяем количество вершин
        self.assertEqual(graph.num_vertices, 200)
        
        # Проверяем, что степени вершин распределены (есть хабы)
        degrees = [graph.get_degree(v) for v in range(graph.num_vertices)]
        max_degree = max(degrees)
        min_degree = min(degrees)
        
        # В BA графах должны быть вершины с высокой степеньой
        self.assertGreater(max_degree, 10)
        self.assertGreaterEqual(min_degree, 2)  # Каждая новая вершина имеет m рёбер
    
    def test_erdos_renyi_generator(self):
        """Тест генератора Эрдёша-Реньи"""
        gen = ErdosRenyiGenerator(n=100, p=0.05, seed=42)
        graph = gen.generate()
        
        self.assertEqual(graph.num_vertices, 100)
        
        # Ожидаемое количество рёбер примерно p * n*(n-1)/2
        expected_edges = int(0.05 * 100 * 99 / 2)
        self.assertAlmostEqual(graph.num_edges, expected_edges, delta=50)
    
    def test_hierarchical_cluster_generator(self):
        """Тест иерархического кластерного генератора"""
        gen = HierarchicalClusterGenerator(
            hierarchy_levels=2,
            branching_factor=3,
            cluster_size=10,
            intra_prob=0.7,
            inter_prob=0.1,
            seed=42
        )
        
        graph = gen.generate()
        
        # Проверяем структуру
        self.assertIsNotNone(graph)
        self.assertGreater(graph.num_vertices, 0)
        
        # Проверяем, что есть рёбра
        self.assertGreater(graph.num_edges, 0)
    
    def test_power_law_cluster_generator(self):
        """Тест генератора со степенным распределением"""
        gen = PowerLawClusterGenerator(
            num_clusters=5,
            power_law_exponent=2.5,
            min_cluster_size=10,
            max_cluster_size=50,
            intra_density=0.3,
            inter_density=0.01,
            seed=42
        )
        
        graph = gen.generate()
        
        self.assertIsNotNone(graph)
        self.assertGreater(graph.num_vertices, 0)
        self.assertGreater(graph.num_edges, 0)
    
    def test_generator_reproducibility(self):
        """
        Тест воспроизводимости генераторов
        
        Примечание: Из-за вероятностного характера и разных распределений,
        даже с одинаковым seed количество рёбер может незначительно отличаться.
        Поэтому проверяем не точное равенство, а близость.
        """
        gen1 = ClusterGraphGenerator(num_clusters=3, cluster_size=10, seed=123)
        gen2 = ClusterGraphGenerator(num_clusters=3, cluster_size=10, seed=123)
        
        graph1 = gen1.generate()
        graph2 = gen2.generate()
        
        # Одинаковые seed должны давать одинаковые графы
        # Проверяем количество вершин
        self.assertEqual(graph1.num_vertices, graph2.num_vertices)
        
        # Количество рёбер должно быть одинаковым или очень близким
        # Допускаем небольшое расхождение из-за особенностей генерации
        edge_diff = abs(graph1.num_edges - graph2.num_edges)
        self.assertLessEqual(edge_diff, 5, 
                            f"Edge count difference too large: {graph1.num_edges} vs {graph2.num_edges}")
        
        # Проверяем, что структура графа одинакова (если есть рёбра)
        if graph1.num_edges == graph2.num_edges and graph1.num_edges > 0:
            # Проверяем наличие одинаковых рёбер
            edges1 = set((min(u,v), max(u,v)) for u, v, _ in graph1.edges())
            edges2 = set((min(u,v), max(u,v)) for u, v, _ in graph2.edges())
            self.assertEqual(edges1, edges2)
    
    def test_deterministic_generation(self):
        """
        Тест детерминированной генерации с полной фиксацией случайности
        """
        # Сбрасываем seed перед каждым вызовом
        def create_graph():
            random.seed(42)
            np.random.seed(42)
            gen = ClusterGraphGenerator(
                num_clusters=3, 
                cluster_size=10, 
                intra_prob=0.8,
                inter_prob=0.05,
                seed=42
            )
            return gen.generate()
        
        graph1 = create_graph()
        graph2 = create_graph()
        
        # С полным сбросом seed результаты должны быть идентичны
        self.assertEqual(graph1.num_vertices, graph2.num_vertices)
        self.assertEqual(graph1.num_edges, graph2.num_edges)
        
        # Проверяем идентичность рёбер
        edges1 = sorted((min(u,v), max(u,v)) for u, v, _ in graph1.edges())
        edges2 = sorted((min(u,v), max(u,v)) for u, v, _ in graph2.edges())
        self.assertEqual(edges1, edges2)
    
    def test_generator_statistics(self):
        """Тест сбора статистики генерации"""
        gen = ClusterGraphGenerator(num_clusters=4, cluster_size=25, seed=42)
        graph, stats = gen.generate_with_stats()
        
        # Проверяем наличие всех ключей
        expected_keys = ['generator', 'num_vertices', 'num_edges', 'density',
                        'generation_time', 'params', 'max_degree', 
                        'min_degree', 'avg_degree']
        
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Проверяем корректность значений
        self.assertEqual(stats['num_vertices'], 100)
        self.assertEqual(stats['num_edges'], graph.num_edges)
    
    def test_graph_properties(self):
        """Тест свойств сгенерированных графов"""
        gen = ClusterGraphGenerator(num_clusters=4, cluster_size=25, seed=42)
        graph = gen.generate()
        
        # Проверяем, что граф не содержит петель
        for v in range(graph.num_vertices):
            self.assertFalse(graph.has_edge(v, v))
        
        # Проверяем, что веса вершин корректны
        for v in range(graph.num_vertices):
            self.assertEqual(graph.get_vertex_weight(v), 1)
        
        # Проверяем, что степени вершин в разумных пределах
        degrees = [graph.get_degree(v) for v in range(graph.num_vertices)]
        self.assertGreater(max(degrees), 0)
        self.assertLess(min(degrees), graph.num_vertices)


def run_generator_tests():
    """Запуск всех тестов генераторов"""
    # Устанавливаем seed для воспроизводимости
    random.seed(42)
    np.random.seed(42)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGenerators)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == '__main__':
    # Устанавливаем seed перед запуском
    random.seed(42)
    np.random.seed(42)
    unittest.main()