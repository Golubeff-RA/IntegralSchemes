"""
Многоуровневое разбиение графа - УЛУЧШЕННАЯ ВЕРСИЯ
"""

import time
import random
from typing import List, Tuple, Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.graph import Graph
from core.partition import Partition
from ..base_partitioner import BasePartitioner
from ..kernighan_lin import KernighanLin


class MultilevelPartitioner(BasePartitioner):
    """
    Многоуровневый алгоритм разбиения графа
    """
    
    def __init__(self, min_coarse_vertices: int = 20, max_levels: int = 5, num_trials: int = 5):
        super().__init__(name="MultilevelPartitioner")
        self.min_coarse_vertices = min_coarse_vertices
        self.max_levels = max_levels
        self.num_trials = num_trials  # Количество попыток
        self.coarse_graphs_history = []
    
    def partition(self, graph: Graph, num_parts: int = 2, balance_ratio: float = 0.5) -> Partition:
        """Многоуровневое разбиение графа"""
        
        print(f"\n{'='*60}")
        print(f"MULTILEVEL PARTITIONING (k={num_parts})")
        print(f"{'='*60}")
        print(f"Graph: {graph.num_vertices} vertices, {graph.num_edges} edges")
        
        # Этап 1: Стягивание (один раз, используется для всех попыток)
        print(f"\n📉 PHASE 1: COARSENING")
        levels = self._coarsen(graph)
        
        if not levels:
            print("  No coarsening, using KL directly")
            kl = KernighanLin(max_passes=20)
            return kl.partition(graph, num_parts, balance_ratio)
        
        # Этап 2-3: Множественные попытки разбиения и проекции
        print(f"\n🎲 PHASE 2-3: PARTITIONING & UNCOARSENING ({self.num_trials} trials)")
        
        best_partition = None
        best_cut = float('inf')
        
        for trial in range(self.num_trials):
            print(f"\n  Trial {trial + 1}/{self.num_trials}")
            
            # Разбиение самого грубого графа
            coarse_graph, vertex_map = levels[-1]
            partition = self._partition_coarse(coarse_graph, num_parts, balance_ratio, trial)
            
            # Проекция обратно
            for level in reversed(levels[:-1]):
                prev_graph, prev_map = level
                partition = self._project(partition, prev_map, prev_graph, num_parts, balance_ratio)
            
            # Финальная проверка
            partition.fix_unassigned(graph)
            current_cut = partition.cut_edges(graph)
            
            print(f"    Cut size: {current_cut}")
            
            if current_cut < best_cut:
                best_cut = current_cut
                best_partition = partition
                print(f"    *** NEW BEST! ***")
        
        # Финальная статистика
        best_partition.fix_unassigned(graph)
        self._statistics['final_cut'] = best_cut
        self._statistics['partition_time'] = time.time() - self._statistics.get('start_time', time.time())
        
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"✂️  Final cut size: {best_cut}")
        print(f"📊 Final part sizes: {best_partition.part_sizes}")
        print(f"📈 Best of {self.num_trials} trials")
        
        return best_partition
    
    def _coarsen(self, graph: Graph) -> List[Tuple[Graph, Dict[int, List[int]]]]:
        """
        Стягивание графа - возвращает список (граф, маппинг)
        """
        levels = []
        current = graph
        self.coarse_graphs_history = [current]
        
        for level in range(self.max_levels):
            if current.num_vertices <= self.min_coarse_vertices:
                print(f"  Level {level}: {current.num_vertices} vertices - stopping")
                break
            
            print(f"  Level {level}: {current.num_vertices} vertices -> ", end="")
            
            # Находим пары для стягивания (используем тяжелые рёбра)
            matching = self._find_heavy_matching(current)
            
            if not matching or len(matching) < current.num_vertices // 10:
                print(f"no matching found")
                break
            
            # Стягиваем граф
            coarse, vertex_map = self._contract(current, matching)
            levels.append((current, vertex_map))
            current = coarse
            self.coarse_graphs_history.append(current)
            print(f"{current.num_vertices} vertices")
        
        # Добавляем последний уровень
        levels.append((current, {v: [v] for v in range(current.num_vertices)}))
        
        return levels
    
    def _find_heavy_matching(self, graph: Graph) -> List[Tuple[int, int]]:
        """
        Находит паросочетание, предпочитая тяжёлые рёбра
        """
        used = set()
        matching = []
        
        # Получаем все рёбра и сортируем по весу
        edges = list(graph.edges())
        edges.sort(key=lambda x: x[2], reverse=True)
        
        for u, v, w in edges:
            if u not in used and v not in used:
                matching.append((u, v))
                used.add(u)
                used.add(v)
        
        return matching
    
    def _contract(self, graph: Graph, matching: List[Tuple[int, int]]) -> Tuple[Graph, Dict[int, List[int]]]:
        """
        Стягивает пары вершин в одну
        """
        vertex_map = {}
        next_id = 0
        
        # Сначала обрабатываем пары
        for u, v in matching:
            vertex_map[u] = next_id
            vertex_map[v] = next_id
            next_id += 1
        
        # Обрабатываем одиночные вершины
        for v in range(graph.num_vertices):
            if v not in vertex_map:
                vertex_map[v] = next_id
                next_id += 1
        
        # Создаём обратный маппинг
        reverse_map = {new_id: [] for new_id in range(next_id)}
        for old, new in vertex_map.items():
            reverse_map[new].append(old)
        
        # Создаём новый граф
        new_graph = Graph(next_id)
        
        # Копируем веса вершин (суммируем)
        for new_id, old_vertices in reverse_map.items():
            total_weight = sum(graph.get_vertex_weight(v) for v in old_vertices)
            new_graph.set_vertex_weight(new_id, total_weight)
        
        # Добавляем рёбра между новыми вершинами
        edge_weights = {}
        for u, v, w in graph.edges():
            nu = vertex_map[u]
            nv = vertex_map[v]
            
            if nu != nv:
                key = (min(nu, nv), max(nu, nv))
                edge_weights[key] = edge_weights.get(key, 0) + w
        
        for (nu, nv), w in edge_weights.items():
            new_graph.add_edge(nu, nv, w)
        
        return new_graph, reverse_map
    
    def _partition_coarse(self, graph: Graph, num_parts: int, balance_ratio: float, trial: int) -> Partition:
        """
        Разбиение грубого графа с разными seed
        """
        # Разный seed для разных попыток
        random.seed(trial * 12345)
        
        # Случайное сбалансированное разбиение
        partition = Partition(graph.num_vertices, num_parts)
        
        target = graph.num_vertices // num_parts
        vertices = list(range(graph.num_vertices))
        random.shuffle(vertices)
        
        part_sizes = [0] * num_parts
        for v in vertices:
            # Выбираем часть с наименьшим размером
            min_part = min(range(num_parts), key=lambda p: part_sizes[p])
            partition.assign(v, min_part)
            part_sizes[min_part] += 1
        
        # Улучшаем KL
        kl = KernighanLin(max_passes=20)
        partition = kl.partition(graph, num_parts, balance_ratio)
        
        return partition
    
    def _project(self, partition: Partition, reverse_map: Dict[int, List[int]],
                 graph: Graph, num_parts: int, balance_ratio: float) -> Partition:
        """
        Проекция разбиения на более детальный уровень
        """
        # Определяем количество вершин в детальном графе
        num_fine = sum(len(vertices) for vertices in reverse_map.values())
        
        # Создаём новое разбиение
        fine_partition = Partition(num_fine, num_parts)
        
        # Каждая вершина получает часть своей грубой вершины
        for coarse_v, fine_vertices in reverse_map.items():
            part = partition.get_part(coarse_v)
            if part != -1:
                for v in fine_vertices:
                    fine_partition.assign(v, part)
        
        # Назначаем оставшиеся вершины
        fine_partition.fix_unassigned(graph)
        
        # Улучшаем KL (многократно)
        best_partition = fine_partition
        best_cut = fine_partition.cut_edges(graph)
        
        for _ in range(3):
            kl = KernighanLin(max_passes=15)
            test_partition = kl.partition(graph, num_parts, balance_ratio)
            test_cut = test_partition.cut_edges(graph)
            
            if test_cut < best_cut:
                best_cut = test_cut
                best_partition = test_partition
        
        return best_partition
    
    def get_coarsening_history(self):
        """Получение истории стягивания для визуализации"""
        return self.coarse_graphs_history