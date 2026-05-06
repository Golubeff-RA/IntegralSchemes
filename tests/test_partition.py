"""
Модульные тесты для класса Partition (бисекция графа)
"""

import unittest
import sys
import random
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition


class TestPartitionBasic(unittest.TestCase):
    """Базовые тесты для Partition"""
    
    def setUp(self):
        self.n = 10
        self.partition = Partition(self.n)
    
    def test_initialization(self):
        """Тест: инициализация разбиения"""
        self.assertEqual(self.partition.num_vertices, self.n)
        self.assertEqual(self.partition.size0, 0)
        self.assertEqual(self.partition.size1, 0)
        self.assertEqual(self.partition.weight0, 0)
        self.assertEqual(self.partition.weight1, 0)
        
        for v in range(self.n):
            self.assertEqual(self.partition.get_part(v), -1)
    
    def test_assign_vertex(self):
        """Тест: назначение вершины в часть"""
        self.partition.assign(0, 0)
        self.assertEqual(self.partition.get_part(0), 0)
        self.assertEqual(self.partition.size0, 1)
        
        self.partition.assign(1, 1)
        self.assertEqual(self.partition.get_part(1), 1)
        self.assertEqual(self.partition.size1, 1)
        
        self.partition.assign(0, 1)
        self.assertEqual(self.partition.get_part(0), 1)
        self.assertEqual(self.partition.size0, 0)
        self.assertEqual(self.partition.size1, 2)
    
    def test_get_vertices(self):
        """Тест: получение вершин по частям"""
        self.partition.assign(0, 0)
        self.partition.assign(1, 0)
        self.partition.assign(2, 1)
        self.partition.assign(3, 1)
        
        vertices0 = self.partition.get_vertices(0)
        vertices1 = self.partition.get_vertices(1)
        
        self.assertEqual(set(vertices0), {0, 1})
        self.assertEqual(set(vertices1), {2, 3})


class TestPartitionWeights(unittest.TestCase):
    """Тесты для весов вершин"""
    
    def setUp(self):
        self.graph = Graph(5)
        self.weights = [10, 20, 30, 40, 50]
        for v, w in enumerate(self.weights):
            self.graph.set_vertex_weight(v, w)
        
        self.partition = Partition(5)
        self.partition.assign(0, 0)
        self.partition.assign(1, 0)
        self.partition.assign(2, 1)
        self.partition.assign(3, 1)
    
    def test_update_weights(self):
        """Тест: обновление весов частей"""
        self.partition.update_weights(self.graph)
        
        self.assertEqual(self.partition.weight0, 10 + 20)
        self.assertEqual(self.partition.weight1, 30 + 40)
        self.assertEqual(self.partition.total_weight, 100)
    
    def test_weight_after_assign(self):
        """Тест: вес после назначения вершины"""
        self.partition.update_weights(self.graph)
        self.partition.assign(4, 0)
        self.partition.update_weights(self.graph)
        
        self.assertEqual(self.partition.weight0, 10 + 20 + 50)
        self.assertEqual(self.partition.weight1, 30 + 40)


class TestPartitionCut(unittest.TestCase):
    """Тесты для cut weight"""
    
    def test_cut_weight_simple(self):
        """Тест: cut weight для простого разбиения"""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        graph.add_edge(0, 3, 1)
        
        partition = Partition(4)
        partition.assign(0, 0)
        partition.assign(1, 0)
        partition.assign(2, 1)
        partition.assign(3, 1)
        
        cut = partition.cut_weight(graph)
        self.assertEqual(cut, 2)  # рёбра 1-2 и 0-3
    
    def test_cut_weight_empty(self):
        """Тест: cut weight для пустого разбиения"""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        
        partition = Partition(4)
        cut = partition.cut_weight(graph)
        self.assertEqual(cut, 0)


class TestPartitionMoveVertex(unittest.TestCase):
    """Тесты для перемещения вершин"""
    
    def test_move_vertex_basic(self):
        """Тест: базовое перемещение вершины"""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        
        partition = Partition(4)
        partition.assign(0, 0)
        partition.assign(1, 0)
        partition.assign(2, 1)
        
        old_cut = partition.cut_weight(graph)
        delta = partition.move_vertex(1, graph)
        new_cut = partition.cut_weight(graph)
        
        self.assertEqual(partition.get_part(1), 1)
        self.assertEqual(new_cut, old_cut + delta)
    
    def test_move_vertex_to_specific(self):
        """Тест: перемещение в конкретную часть"""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        
        partition = Partition(4)
        partition.assign(0, 0)
        partition.assign(1, 1)
        
        partition.move_vertex_to(0, 1, graph)
        self.assertEqual(partition.get_part(0), 1)


class TestPartitionSwapVertices(unittest.TestCase):
    """Тесты для обмена вершинами"""
    
    def test_swap_vertices_basic(self):
        """Тест: базовый обмен вершинами"""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        
        partition = Partition(4)
        partition.assign(0, 0)
        partition.assign(1, 0)
        partition.assign(2, 1)
        partition.assign(3, 1)
        
        old_cut = partition.cut_weight(graph)
        partition.swap_vertices(1, 2, graph)
        new_cut = partition.cut_weight(graph)
        
        # После обмена вершины должны поменяться частями
        self.assertEqual(partition.get_part(1), 1)
        self.assertEqual(partition.get_part(2), 0)
        
        # cut должен измениться
        self.assertNotEqual(new_cut, old_cut)
    
    def test_swap_vertices_symmetric(self):
        """Тест: двойной обмен возвращает исходное состояние"""
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        
        partition = Partition(4)
        partition.assign(0, 0)
        partition.assign(1, 1)
        
        partition.swap_vertices(0, 1, graph)
        partition.swap_vertices(0, 1, graph)
        
        self.assertEqual(partition.get_part(0), 0)
        self.assertEqual(partition.get_part(1), 1)


class TestPartitionBalance(unittest.TestCase):
    """Тесты для балансировки"""
    
    def test_balance_quality(self):
        """Тест: качество балансировки"""
        partition = Partition(10)
        for v in range(5):
            partition.assign(v, 0)
        for v in range(5, 10):
            partition.assign(v, 1)
        
        balance = partition.balance_quality()
        self.assertEqual(balance, 1.0)
    
    def test_balance_quality_unequal(self):
        """Тест: несбалансированное разбиение"""
        partition = Partition(10)
        for v in range(8):
            partition.assign(v, 0)
        for v in range(8, 10):
            partition.assign(v, 1)
        
        balance = partition.balance_quality()
        self.assertAlmostEqual(balance, 2/8, places=2)
    
    def test_fix_unassigned(self):
        """Тест: fix_unassigned назначает все вершины"""
        graph = Graph(5)
        for v in range(5):
            graph.set_vertex_weight(v, 1)
        
        partition = Partition(5)
        for v in range(3):
            partition.assign(v, 0)
        
        self.assertFalse(partition.is_complete())
        
        partition.fix_unassigned(graph)
        
        self.assertTrue(partition.is_complete())
        self.assertEqual(partition.size0 + partition.size1, 5)


class TestPartitionComplete(unittest.TestCase):
    """Тесты для проверки полноты разбиения"""
    
    def test_is_complete(self):
        """Тест: проверка полноты разбиения"""
        partition = Partition(5)
        self.assertFalse(partition.is_complete())
        
        for v in range(5):
            partition.assign(v, v % 2)
        
        self.assertTrue(partition.is_complete())
    
    def test_copy(self):
        """Тест: копирование разбиения"""
        partition = Partition(5)
        for v in range(5):
            partition.assign(v, v % 2)
        
        copy = partition.copy()
        
        self.assertEqual(copy.size0, partition.size0)
        self.assertEqual(copy.size1, partition.size1)
        
        copy.assign(0, 1)
        self.assertNotEqual(copy.get_part(0), partition.get_part(0))


class TestPartitionPerformance(unittest.TestCase):
    """Тесты производительности"""
    
    def test_large_partition_creation(self):
        """Тест: создание большого разбиения"""
        n = 10000
        partition = Partition(n)
        self.assertEqual(partition.num_vertices, n)
    
    def test_many_assignments(self):
        """Тест: много назначений"""
        n = 1000
        partition = Partition(n)
        
        for v in range(n):
            partition.assign(v, v % 2)
        
        self.assertEqual(partition.size0 + partition.size1, n)


def run_all_tests():
    """Запуск всех тестов Partition"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestPartitionBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestPartitionWeights))
    suite.addTests(loader.loadTestsFromTestCase(TestPartitionCut))
    suite.addTests(loader.loadTestsFromTestCase(TestPartitionMoveVertex))
    suite.addTests(loader.loadTestsFromTestCase(TestPartitionSwapVertices))
    suite.addTests(loader.loadTestsFromTestCase(TestPartitionBalance))
    suite.addTests(loader.loadTestsFromTestCase(TestPartitionComplete))
    suite.addTests(loader.loadTestsFromTestCase(TestPartitionPerformance))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    run_all_tests()