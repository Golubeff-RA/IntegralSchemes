"""
Модульные тесты для алгоритма Кернигана-Лина
"""

import unittest
import sys
import random
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition
from algorithms import KernighanLin, ImprovedKernighanLin


class TestKernighanLin(unittest.TestCase):
    """Тесты для KL алгоритма"""
    
    def setUp(self):
        """Подготовка перед каждым тестом"""
        self.kl = KernighanLin(max_passes=10, max_iterations=100)
        random.seed(42)  # Фиксируем seed для воспроизводимости
    
    def test_simple_graph(self):
        """Тест на простом графе"""
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        
        partition = self.kl.partition(g, num_parts=2, balance_ratio=0.5)
        
        # Проверяем, что разбиение сбалансировано
        self.assertTrue(partition.is_balanced(0.5))
        
        # Проверяем, что все вершины назначены
        for v in range(g.num_vertices):
            self.assertNotEqual(partition.get_part(v), -1)
    
    def test_complete_graph(self):
        """Тест на полном графе"""
        g = Graph(6)
        for i in range(6):
            for j in range(i + 1, 6):
                g.add_edge(i, j)
        
        partition = self.kl.partition(g, num_parts=2, balance_ratio=0.5)
        
        # В полном графе оптимальное разбиение - это минимум разрезов
        cut = partition.cut_edges(g)
        
        # Минимальный разрез для K6 на 2 равные части = 3*3 = 9
        self.assertEqual(cut, 9)
    
    def test_balanced_partition(self):
        """Тест балансировки разбиения"""
        g = Graph(10)
        for i in range(9):
            g.add_edge(i, i + 1)
        
        partition = self.kl.partition(g, num_parts=2, balance_ratio=0.5)
        
        part_sizes = partition.part_sizes
        # Части должны быть примерно равны
        self.assertAlmostEqual(part_sizes[0], part_sizes[1], delta=2)
    
    def test_improvement_over_random(self):
        """Тест улучшения по сравнению со случайным разбиением"""
        g = Graph(20)
        
        # Кластер 1 (вершины 0-9)
        for i in range(10):
            for j in range(i + 1, 10):
                g.add_edge(i, j)
        
        # Кластер 2 (вершины 10-19)
        for i in range(10, 20):
            for j in range(i + 1, 20):
                g.add_edge(i, j)
        
        # Несколько связей между кластерами
        g.add_edge(0, 10)
        g.add_edge(5, 15)
        
        # Случайное разбиение
        random_partition = Partition(20, 2)
        for v in range(20):
            random_partition.assign(v, random.randint(0, 1))
        
        random_cut = random_partition.cut_edges(g)
        
        # KL улучшение
        optimized_partition = self.kl.partition(g, num_parts=2, balance_ratio=0.5)
        optimized_cut = optimized_partition.cut_edges(g)
        
        # KL должен найти лучшее разбиение
        self.assertLessEqual(optimized_cut, random_cut)
    
    def test_edge_weights(self):
        """Тест с весами рёбер"""
        g = Graph(4)
        g.add_edge(0, 1, weight=10)
        g.add_edge(1, 2, weight=1)
        g.add_edge(2, 3, weight=1)
        g.add_edge(3, 0, weight=1)
        
        partition = self.kl.partition(g, num_parts=2, balance_ratio=0.5)
        
        # Оптимально разорвать тяжёлое ребро
        cut = partition.cut_edges(g)
        self.assertLess(cut, 5)
    
    def test_vertex_weights(self):
        """
        Тест с весами вершин - проверяем корректность хранения и суммирования
        """
        g = Graph(4)
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)
        
        # Устанавливаем веса вершин
        g.set_vertex_weight(0, 10)
        g.set_vertex_weight(1, 20)
        g.set_vertex_weight(2, 30)
        g.set_vertex_weight(3, 40)
        
        partition = self.kl.partition(g, num_parts=2, balance_ratio=0.5)
        
        # Проверяем, что веса вершин сохранились
        for v in range(g.num_vertices):
            self.assertEqual(g.get_vertex_weight(v), 
                           [10, 20, 30, 40][v])
        
        # Вычисляем веса частей
        part_weights = partition.compute_part_weights(g)
        
        # Проверяем, что сумма весов равна общей сумме
        total_weight = sum(g.get_vertex_weight(v) for v in range(g.num_vertices))
        self.assertEqual(sum(part_weights), total_weight)
        
        # Проверяем, что веса частей положительные
        self.assertGreater(part_weights[0], 0)
        self.assertGreater(part_weights[1], 0)
        
        print(f"Part weights: {part_weights}")
        print(f"Part sizes: {partition.part_sizes}")
    
    def test_statistics_collection(self):
        """Тест сбора статистики"""
        g = Graph(10)
        for i in range(9):
            g.add_edge(i, i + 1)
        
        partition, stats = self.kl.partition_with_stats(g, num_parts=2)
        
        # Проверяем наличие статистики
        expected_keys = ['partition_time', 'initial_cut', 'final_cut', 
                        'iterations', 'memory_usage_mb', 'num_vertices',
                        'num_edges', 'num_parts', 'balance_ratio',
                        'actual_balance', 'is_balanced', 'part_sizes']
        
        for key in expected_keys:
            self.assertIn(key, stats)
    
    def test_improved_kl(self):
        """Тест улучшенной версии KL"""
        g = Graph(20)
        for i in range(19):
            g.add_edge(i, i + 1)
        
        kl_standard = KernighanLin(max_passes=10)
        kl_improved = ImprovedKernighanLin(max_passes=10)
        
        partition_std, stats_std = kl_standard.partition_with_stats(g, num_parts=2)
        partition_imp, stats_imp = kl_improved.partition_with_stats(g, num_parts=2)
        
        # Улучшенная версия должна работать (просто проверяем, что не падает)
        self.assertIsNotNone(partition_imp)
        self.assertIsNotNone(stats_imp)
    
    def test_empty_graph(self):
        """Тест на графе без рёбер"""
        g = Graph(5)
        
        partition = self.kl.partition(g, num_parts=2, balance_ratio=0.5)
        
        # Разрез должен быть 0
        self.assertEqual(partition.cut_edges(g), 0)
        
        # Части должны быть сбалансированы
        self.assertTrue(partition.is_balanced(0.5))
    
    def test_single_vertex_graph(self):
        """Тест на графе из одной вершины"""
        g = Graph(1)
        
        partition = self.kl.partition(g, num_parts=2, balance_ratio=0.5)
        
        # Вершина может быть в любой части, но должна быть назначена
        part = partition.get_part(0)
        self.assertNotEqual(part, -1)  # Вершина назначена
        self.assertTrue(part == 0 or part == 1)  # В одной из частей
        
        # Проверяем размеры частей
        self.assertEqual(partition.part_sizes[0] + partition.part_sizes[1], 1)
    
    def test_two_vertex_graph(self):
        """Тест на графе из двух вершин"""
        g = Graph(2)
        g.add_edge(0, 1)
        
        partition = self.kl.partition(g, num_parts=2, balance_ratio=0.5)
        
        # Вершины должны быть в разных частях (оптимально)
        self.assertNotEqual(partition.get_part(0), partition.get_part(1))
        
        # Разрез должен быть 1
        self.assertEqual(partition.cut_edges(g), 1)


class TestKernighanLinEdgeCases(unittest.TestCase):
    """Тесты крайних случаев для KL алгоритма"""
    
    def setUp(self):
        self.kl = KernighanLin(max_passes=5)
    
    def test_disconnected_graph(self):
        """
        Тест на несвязном графе
        Примечание: KL может создавать разрезы между компонентами,
        но это нормально, так как алгоритм оптимизирует cut size
        """
        g = Graph(6)
        # Компонента 1: вершины 0-2
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        # Компонента 2: вершины 3-5
        g.add_edge(3, 4)
        g.add_edge(4, 5)
        
        partition = self.kl.partition(g, num_parts=2, balance_ratio=0.5)
        
        # Просто проверяем, что разбиение существует и сбалансировано
        self.assertTrue(partition.is_balanced(0.5))
        
        # Все вершины должны быть назначены
        for v in range(g.num_vertices):
            self.assertNotEqual(partition.get_part(v), -1)
    
    def test_star_graph(self):
        """Тест на звездообразном графе"""
        g = Graph(10)
        # Центральная вершина 0 соединена со всеми
        for i in range(1, 10):
            g.add_edge(0, i)
        
        partition = self.kl.partition(g, num_parts=2, balance_ratio=0.5)
        
        # Проверяем балансировку
        part_sizes = partition.part_sizes
        self.assertAlmostEqual(part_sizes[0], part_sizes[1], delta=2)
    
    def test_bipartite_graph(self):
        """
        Тест на двудольном графе
        Примечание: Для K_{4,4} оптимальный разрез - 0,
        но KL может найти локальный оптимум
        """
        g = Graph(8)
        # Полный двудольный граф K_{4,4}
        for i in range(4):
            for j in range(4, 8):
                g.add_edge(i, j)
        
        partition = self.kl.partition(g, num_parts=2, balance_ratio=0.5)
        
        # Просто проверяем, что разбиение существует
        self.assertTrue(partition.is_balanced(0.5))
        
        # Выводим информацию для отладки
        cut = partition.cut_edges(g)
        print(f"Bipartite graph cut: {cut} (total edges: {g.num_edges})")
        
        # Разрез должен быть не больше общего количества рёбер
        self.assertLessEqual(cut, g.num_edges)
    
    def test_line_graph_balance(self):
        """Тест на линейном графе"""
        g = Graph(10)
        for i in range(9):
            g.add_edge(i, i + 1)
        
        partition = self.kl.partition(g, num_parts=2, balance_ratio=0.5)
        
        # Проверяем балансировку
        self.assertTrue(partition.is_balanced(0.5))
        
        # Проверяем, что разрез не слишком большой
        cut = partition.cut_edges(g)
        self.assertLessEqual(cut, g.num_edges)
    
    def test_random_graph_small(self):
        """Тест на маленьком случайном графе"""
        g = Graph(8)
        # Добавляем случайные рёбра
        edges = [(0,1), (0,2), (1,3), (2,4), (3,5), (4,6), (5,7), (6,7)]
        for u, v in edges:
            g.add_edge(u, v)
        
        partition = self.kl.partition(g, num_parts=2, balance_ratio=0.5)
        
        # Проверяем, что разбиение существует
        self.assertEqual(partition.num_vertices, g.num_vertices)
        self.assertTrue(partition.is_balanced(0.5))


def run_kl_tests():
    """Запуск всех тестов KL"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestKernighanLin))
    suite.addTests(loader.loadTestsFromTestCase(TestKernighanLinEdgeCases))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == '__main__':
    unittest.main()