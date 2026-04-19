"""
Управление запуском алгоритмов в отдельных потоках
"""

import threading
import time
from typing import Optional, Callable
from dataclasses import dataclass
from queue import Queue

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition
from algorithms import KernighanLin, MultilevelPartitioner


@dataclass
class AlgorithmResult:
    """Результат выполнения алгоритма"""
    algorithm_name: str
    partition: Partition
    elapsed_time: float
    stats: dict
    success: bool = True
    error: Optional[str] = None


class AlgorithmRunner:
    """
    Запуск алгоритмов в отдельном потоке с возможностью отмены
    """
    
    def __init__(self):
        self.current_thread: Optional[threading.Thread] = None
        self.should_cancel = False
        self.result_queue = Queue()
    
    def run_kl(self, graph: Graph, num_parts: int, balance_ratio: float,
              callback: Callable[[AlgorithmResult], None]):
        """Запуск KL алгоритма"""
        self._run_algorithm(
            "Kernighan-Lin",
            lambda: self._run_kl_impl(graph, num_parts, balance_ratio),
            callback
        )
    
    def run_multilevel(self, graph: Graph, num_parts: int, balance_ratio: float,
                      callback: Callable[[AlgorithmResult], None]):
        """Запуск многоуровневого алгоритма"""
        self._run_algorithm(
            "Multilevel",
            lambda: self._run_multilevel_impl(graph, num_parts, balance_ratio),
            callback
        )
    
    def run_compare(self, graph: Graph, num_parts: int, balance_ratio: float,
                   callback: Callable[[AlgorithmResult], None]):
        """Сравнение алгоритмов"""
        self._run_algorithm(
            "Comparison",
            lambda: self._run_compare_impl(graph, num_parts, balance_ratio),
            callback
        )
    
    def _run_kl_impl(self, graph: Graph, num_parts: int, balance_ratio: float) -> AlgorithmResult:
        """Реализация KL"""
        start = time.time()
        kl = KernighanLin(max_passes=10)
        partition = kl.partition(graph, num_parts, balance_ratio)
        elapsed = time.time() - start
        
        return AlgorithmResult(
            algorithm_name="KL",
            partition=partition,
            elapsed_time=elapsed,
            stats=kl.get_statistics()
        )
    
    def _run_multilevel_impl(self, graph: Graph, num_parts: int, balance_ratio: float) -> AlgorithmResult:
        """Реализация многоуровневого алгоритма"""
        start = time.time()
        ml = MultilevelPartitioner(min_coarse_vertices=10)
        partition, stats = ml.partition_with_stats(graph, num_parts, balance_ratio)
        elapsed = time.time() - start
        
        return AlgorithmResult(
            algorithm_name="Multilevel",
            partition=partition,
            elapsed_time=elapsed,
            stats=stats
        )
    
    def _run_compare_impl(self, graph: Graph, num_parts: int, balance_ratio: float) -> AlgorithmResult:
        """Сравнение алгоритмов"""
        # Запускаем KL
        start_kl = time.time()
        kl = KernighanLin(max_passes=10)
        partition_kl = kl.partition(graph, num_parts, balance_ratio)
        time_kl = time.time() - start_kl
        cut_kl = partition_kl.cut_edges(graph)
        
        # Запускаем Multilevel
        start_ml = time.time()
        ml = MultilevelPartitioner(min_coarse_vertices=10)
        partition_ml, stats_ml = ml.partition_with_stats(graph, num_parts, balance_ratio)
        time_ml = time.time() - start_ml
        cut_ml = partition_ml.cut_edges(graph)
        
        # Выбираем лучшее
        if cut_ml < cut_kl:
            partition = partition_ml
            stats = {
                **stats_ml,
                'comparison': {'kl_cut': cut_kl, 'kl_time': time_kl}
            }
        else:
            partition = partition_kl
            stats = {
                **kl.get_statistics(),
                'comparison': {'ml_cut': cut_ml, 'ml_time': time_ml}
            }
        
        return AlgorithmResult(
            algorithm_name="Comparison (Best)",
            partition=partition,
            elapsed_time=min(time_kl, time_ml),
            stats=stats
        )
    
    def _run_algorithm(self, name: str, impl_func, callback: Callable):
        """Общий метод запуска алгоритма"""
        if self.current_thread and self.current_thread.is_alive():
            # Отменяем предыдущий запуск
            self.should_cancel = True
            self.current_thread.join(timeout=1)
        
        self.should_cancel = False
        
        def target():
            try:
                if not self.should_cancel:
                    result = impl_func()
                    self.result_queue.put(result)
            except Exception as e:
                self.result_queue.put(AlgorithmResult(
                    algorithm_name=name,
                    partition=None,
                    elapsed_time=0,
                    stats={},
                    success=False,
                    error=str(e)
                ))
        
        self.current_thread = threading.Thread(target=target, daemon=True)
        self.current_thread.start()
        
        # Проверяем результат в отдельном потоке
        def check_result():
            try:
                result = self.result_queue.get(timeout=0.1)
                callback(result)
            except:
                self.root.after(100, check_result)
        
        self.root.after(100, check_result)
    
    def cancel(self):
        """Отмена текущего запуска"""
        self.should_cancel = True