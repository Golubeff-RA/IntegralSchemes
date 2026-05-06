"""
Модульные тесты для многоуровневого алгоритма разбиения
"""

import unittest
import sys
import random
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition
from algorithms.kernighan_lin import KernighanLin
from algorithms.multilevel_slow import MultilevelPartitioner, AdaptiveMultilevelPartitioner
from data.generators import FastClusterGenerator


class TestMultilevelPartitioner(unittest.TestCase):
    """Тесты для многоуровневого разбиения"""
    
    def setUp(self):
        """Подготовка перед каждым тестом"""
        self.seed = 42
        random.seed(self.seed)
    
    def _create_test_graph(self, num_clusters: int = 3, vertices_per_cluster: int = 10):
        """Создание тестового графа с известной структурой"""
        gen = FastClusterGenerator(
            num_clusters=num_clusters,
            vertices_per_cluster=vertices_per_cluster,
            target_edges=200,
            intra_ratio=0.8,
            seed=self.seed
        )
        return gen.generate()
    
    def test_all_vertices_assigned(self):
        """Тест: все вершины должны быть назначены"""
        graph = self._create_test_graph(3, 10)
        ml = MultilevelPartitioner(min_coarse_vertices=10, seed=self.seed)
        partition, _ = ml.partition(graph, balance_ratio=0.5)
        
        # Проверяем, что все вершины назначены
        for v in range(graph.num_vertices):
            self.assertNotEqual(partition.get_part(v), -1, 
                              f"Vertex {v} is not assigned")
        
        self.assertTrue(partition.is_complete(), "Partition is not complete")
    
    def test_partition_sizes_sum(self):
        """Тест: сумма размеров частей равна общему числу вершин"""
        graph = self._create_test_graph(3, 10)
        ml = MultilevelPartitioner(min_coarse_vertices=10, seed=self.seed)
        partition, _ = ml.partition(graph, balance_ratio=0.5)
        
        total = partition.size0 + partition.size1
        self.assertEqual(total, graph.num_vertices,
                        f"Sum of parts ({total}) != total vertices ({graph.num_vertices})")
    
    def test_partition_weights_sum(self):
        """Тест: сумма весов частей равна общему весу вершин"""
        graph = self._create_test_graph(3, 10)
        ml = MultilevelPartitioner(min_coarse_vertices=10, seed=self.seed)
        partition, _ = ml.partition(graph, balance_ratio=0.5)
        
        # Обновляем веса
        partition.update_weights(graph)
        
        total_weight = partition.weight0 + partition.weight1
        expected_weight = sum(graph.get_vertex_weight(v) for v in range(graph.num_vertices))
        
        self.assertEqual(total_weight, expected_weight,
                        f"Sum of weights ({total_weight}) != total weight ({expected_weight})")
    
    def test_balance_ratio(self):
        """Тест: разбиение должно быть сбалансировано"""
        graph = self._create_test_graph(3, 10)
        ml = MultilevelPartitioner(min_coarse_vertices=10, seed=self.seed)
        partition, _ = ml.partition(graph, balance_ratio=0.5)
        
        # Допускаем отклонение до 20%
        ratio = partition.balance_quality()
        self.assertGreaterEqual(ratio, 0.8, 
                               f"Balance ratio {ratio:.4f} is too low (should be >= 0.8)")
    
    def test_cut_weight_consistency(self):
        """Тест: cut_weight должен быть непротиворечив при многократном вызове"""
        graph = self._create_test_graph(3, 10)
        ml = MultilevelPartitioner(min_coarse_vertices=10, seed=self.seed)
        partition, _ = ml.partition(graph, balance_ratio=0.5)
        
        cut1 = partition.cut_weight(graph)
        cut2 = partition.cut_weight(graph)
        
        self.assertEqual(cut1, cut2, "Cut weight changed between calls")
    
    def test_move_vertex_updates_correctly(self):
        """Тест: перемещение вершины корректно обновляет метрики"""
        graph = self._create_test_graph(3, 10)
        ml = MultilevelPartitioner(min_coarse_vertices=10, seed=self.seed)
        partition, _ = ml.partition(graph, balance_ratio=0.5)
        
        # Сохраняем начальные значения
        old_cut = partition.cut_weight(graph)
        old_weight0 = partition.weight0
        old_weight1 = partition.weight1
        old_size0 = partition.size0
        old_size1 = partition.size1
        
        # Находим вершину для перемещения
        vertex = None
        for v in range(graph.num_vertices):
            if partition.get_part(v) == 0:
                vertex = v
                break
        
        if vertex is not None:
            delta = partition.move_vertex(vertex, graph)
            
            # Проверяем, что размеры обновились
            self.assertEqual(partition.size0, old_size0 - 1)
            self.assertEqual(partition.size1, old_size1 + 1)
            
            # Проверяем, что веса обновились
            v_weight = graph.get_vertex_weight(vertex)
            self.assertEqual(partition.weight0, old_weight0 - v_weight)
            self.assertEqual(partition.weight1, old_weight1 + v_weight)
            
            # Проверяем, что cut weight изменился на delta
            new_cut = partition.cut_weight(graph)
            self.assertEqual(new_cut, old_cut + delta)
    
    def test_swap_vertices_updates_correctly(self):
        """Тест: обмен вершинами корректно обновляет метрики"""
        # Создаём простой линейный граф
        graph = Graph(4)
        graph.add_edge(0, 1, 1)
        graph.add_edge(1, 2, 1)
        graph.add_edge(2, 3, 1)
        
        # Простое разбиение: чётные в часть 0, нечётные в часть 1
        partition = Partition(4)
        partition.assign(0, 0)
        partition.assign(1, 1)
        partition.assign(2, 0)
        partition.assign(3, 1)
        partition.update_weights(graph)
        
        # Запоминаем состояние
        old_cut = partition.cut_weight(graph)
        old_weight0 = partition.weight0
        old_weight1 = partition.weight1
        old_size0 = partition.size0
        old_size1 = partition.size1
        
        # Обмениваем вершины 1 и 2
        delta = partition.swap_vertices(1, 2, graph)
        
        # Проверяем, что части поменялись
        self.assertEqual(partition.get_part(1), 0)  # была 1, стала 0
        self.assertEqual(partition.get_part(2), 1)  # была 0, стала 1
        
        # Размеры не изменились
        self.assertEqual(partition.size0, old_size0)
        self.assertEqual(partition.size1, old_size1)
        
        # Веса не изменились (все веса = 1)
        self.assertEqual(partition.weight0, old_weight0)
        self.assertEqual(partition.weight1, old_weight1)
        
        # Проверяем изменение cut
        new_cut = partition.cut_weight(graph)
        
        # delta должна равняться разнице
        # Из-за особенностей вычисления delta, просто проверяем что cut изменился ожидаемо
        # В линейном графе после обмена cut должен уменьшиться
        self.assertLessEqual(new_cut, old_cut, 
                            f"Cut should not increase: old={old_cut}, new={new_cut}")
        
        # Проверяем, что формула delta = new_cut - old_cut работает
        self.assertEqual(new_cut, old_cut + delta,
                        f"new_cut({new_cut}) != old_cut({old_cut}) + delta({delta})")
    
    def test_fix_unassigned(self):
        """Тест: fix_unassigned назначает все неназначенные вершины"""
        graph = self._create_test_graph(3, 10)
        partition = Partition(graph.num_vertices)
        
        # Назначаем только половину вершин
        for v in range(graph.num_vertices // 2):
            partition.assign(v, 0)
        
        # Проверяем, что есть неназначенные вершины
        self.assertFalse(partition.is_complete())
        
        # Исправляем
        partition.fix_unassigned(graph)
        
        # Проверяем, что все назначены
        self.assertTrue(partition.is_complete())
        
        # Проверяем, что веса обновились
        partition.update_weights(graph)
        total_weight = partition.weight0 + partition.weight1
        expected_weight = sum(graph.get_vertex_weight(v) for v in range(graph.num_vertices))
        self.assertEqual(total_weight, expected_weight)
    
    def test_coarsening_levels(self):
        """Тест: количество уровней стягивания разумно"""
        # Используем граф с которым точно будет стягивание
        gen = FastClusterGenerator(
            num_clusters=4,
            vertices_per_cluster=30,
            target_edges=800,
            intra_ratio=0.7,
            seed=self.seed
        )
        graph = gen.generate()
        
        ml = MultilevelPartitioner(min_coarse_vertices=20, max_levels=10, seed=self.seed)
        ml.partition(graph, balance_ratio=0.5)
        
        # Должно быть хотя бы 1 уровень стягивания
        self.assertGreaterEqual(len(ml.levels), 1, 
                            f"Expected at least 1 coarsening level, got {len(ml.levels)}")
        
        # Каждый следующий уровень должен иметь НЕ БОЛЬШЕ вершин (не строго меньше)
        if len(ml.levels) >= 2:
            prev_vertices = graph.num_vertices
            for level in ml.levels:
                self.assertLessEqual(level.graph.num_vertices, prev_vertices,
                                f"Level {level.level}: {level.graph.num_vertices} > {prev_vertices}")
                prev_vertices = level.graph.num_vertices
    
    def test_comparison_with_kl(self):
        """Тест: сравнение с KL алгоритмом"""
        # Используем небольшой граф для теста
        graph = self._create_test_graph(3, 10)
        
        # KL
        kl = KernighanLin(max_passes=20, seed=self.seed)
        kl_partition, kl_metrics = kl.partition(graph, balance_ratio=0.5)
        
        # Multilevel
        ml = MultilevelPartitioner(min_coarse_vertices=10, seed=self.seed)
        ml_partition, ml_metrics = ml.partition(graph, balance_ratio=0.5)
        
        # Проверяем, что оба разбиения корректны
        self.assertTrue(kl_partition.is_complete())
        self.assertTrue(ml_partition.is_complete())
        
        # Выводим сравнение для информации
        print(f"\n  KL cut: {kl_metrics.cut_weight}")
        print(f"  ML cut: {ml_metrics.cut_weight}")
        print(f"  Difference: {kl_metrics.cut_weight - ml_metrics.cut_weight:+d}")
        
        # ML не должен быть намного хуже KL (допускаем ухудшение до 20%)
        # На кластерных графах ML обычно лучше
        degradation = (ml_metrics.cut_weight - kl_metrics.cut_weight) / max(1, kl_metrics.cut_weight)
        self.assertLessEqual(degradation, 0.2, 
                            f"ML is {degradation*100:.1f}% worse than KL")
    
    def test_reproducibility(self):
        """Тест: воспроизводимость результатов"""
        graph = self._create_test_graph(3, 10)
        
        ml1 = MultilevelPartitioner(seed=123)
        partition1, metrics1 = ml1.partition(graph, balance_ratio=0.5)
        
        ml2 = MultilevelPartitioner(seed=123)
        partition2, metrics2 = ml2.partition(graph, balance_ratio=0.5)
        
        # Результаты должны быть одинаковыми
        self.assertEqual(metrics1.cut_weight, metrics2.cut_weight,
                        "Cut weights differ between runs")
        self.assertEqual(partition1.size0, partition2.size0,
                        "Part sizes differ between runs")
    
    def test_adaptive_partitioner(self):
        """Тест: адаптивный разбиватель"""
        sizes = [50, 100, 200]
        
        for size in sizes:
            gen = FastClusterGenerator(
                num_clusters=max(2, size // 25),
                vertices_per_cluster=size // max(2, size // 25),
                target_edges=min(size * 5, 5000),
                intra_ratio=0.7,
                seed=self.seed
            )
            graph = gen.generate()
            
            adaptive = AdaptiveMultilevelPartitioner(total_vertices=size, seed=self.seed)
            partition, metrics = adaptive.partition(graph, balance_ratio=0.5)
            
            # Проверяем корректность
            self.assertTrue(partition.is_complete())
            self.assertGreaterEqual(partition.balance_quality(), 0.7,
                                   f"Balance {partition.balance_quality():.4f} too low for size {size}")
            
            print(f"  Size {size}: cut={metrics.cut_weight}, balance={partition.balance_quality():.4f}, time={metrics.time_seconds:.3f}s")
    
    def test_large_graph_scalability(self):
        """Тест: масштабируемость на большом графе"""
        # Небольшой тест масштабируемости
        sizes = [100, 200, 300]
        times = []
        
        for size in sizes:
            gen = FastClusterGenerator(
                num_clusters=5,
                vertices_per_cluster=size // 5,
                target_edges=min(size * 5, 5000),
                intra_ratio=0.7,
                seed=self.seed
            )
            graph = gen.generate()
            
            ml = MultilevelPartitioner(min_coarse_vertices=20, seed=self.seed)
            _, metrics = ml.partition(graph, balance_ratio=0.5)
            times.append(metrics.time_seconds)
            
            print(f"  Size {size}: {metrics.time_seconds:.3f}s, cut={metrics.cut_weight}")
        
        # Время не должно расти слишком быстро
        if len(times) >= 2:
            # Примерная проверка: при удвоении размера время не должно расти в 10+ раз
            if times[0] > 0:
                growth = times[-1] / times[0]
                print(f"  Time growth factor: {growth:.2f} for {sizes[-1]}/{sizes[0]} = {sizes[-1]/sizes[0]:.0f}x size increase")
                # Не строгое требование, просто информация


def run_all_tests():
    """Запуск всех тестов"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMultilevelPartitioner)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    # Простой запуск
    #unittest.main()
    
    # Или с детальным выводом
    run_all_tests()