"""
Базовый класс для всех алгоритмов разбиения графов

Поддерживает:
- Замер времени выполнения
- Отслеживание использования памяти
- Сбор метрик производительности
"""

import time
import tracemalloc
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition


@dataclass
class PerformanceMetrics:
    """Метрики производительности алгоритма"""
    
    # Время (в секундах)
    time_seconds: float = 0.0
    
    # Память (в МБ)
    memory_mb: float = 0.0
    
    # Дополнительные метрики (зависят от алгоритма)
    extra: Dict[str, Any] = field(default_factory=dict)
    
    # Качество разбиения (заполняется после выполнения)
    cut_weight: int = 0
    balance: float = 0.0
    
    def __str__(self) -> str:
        s = f"⏱️  Time: {self.time_seconds:.4f}s | 💾 Memory: {self.memory_mb:.2f}MB"
        if self.cut_weight:
            s += f" | ✂️  Cut: {self.cut_weight}"
        if self.balance:
            s += f" | ⚖️  Balance: {self.balance:.4f}"
        return s
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            'time_seconds': self.time_seconds,
            'memory_mb': self.memory_mb,
            'cut_weight': self.cut_weight,
            'balance': self.balance,
            **self.extra
        }


class BasePartitioner(ABC):
    """
    Базовый класс для всех алгоритмов разбиения
    
    Пример использования:
    
        class MyPartitioner(BasePartitioner):
            def _partition_impl(self, graph, num_parts, balance_ratio):
                # реализация алгоритма
                return partition
        
        partitioner = MyPartitioner()
        partition, metrics = partitioner.partition(graph)
    """
    
    def __init__(self, name: str = "BasePartitioner"):
        """
        Args:
            name: имя алгоритма (для логирования)
        """
        self.name = name
        self._last_metrics: Optional[PerformanceMetrics] = None
    
    @abstractmethod
    def _partition_impl(self, graph: Graph, balance_ratio: float = 0.5) -> Partition:
        """
        Реализация алгоритма разбиения (должна быть переопределена)
        
        Args:
            graph: граф для разбиения (уже должен быть подготовлен)
            balance_ratio: целевая доля вершин в одной части (0.5 = равные части)
        
        Returns:
            Partition: разбиение графа на 2 части
        """
        pass
    
    def partition(self, graph: Graph, balance_ratio: float = 0.5) -> Tuple[Partition, PerformanceMetrics]:
        """
        Выполнить разбиение графа с замером производительности
        
        Args:
            graph: граф для разбиения
            balance_ratio: целевая доля вершин в одной части
        
        Returns:
            Tuple[Partition, PerformanceMetrics]: (разбиение, метрики)
        """
        # Запускаем отслеживание памяти
        tracemalloc.start()
        
        # Замер времени
        start_time = time.perf_counter()
        
        try:
            # Выполняем алгоритм
            partition = self._partition_impl(graph, balance_ratio)
            
            # Убеждаемся, что все вершины назначены
            if not partition.is_complete():
                partition.fix_unassigned(graph)
            
        finally:
            # Замер времени
            end_time = time.perf_counter()
            
            # Замер памяти
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        
        # Собираем метрики
        metrics = PerformanceMetrics(
            time_seconds=end_time - start_time,
            memory_mb=peak / 1024 / 1024,
            cut_weight=partition.cut_weight(graph),
            balance=partition.balance_quality()
        )
        
        self._last_metrics = metrics
        return partition, metrics
    
    def get_last_metrics(self) -> Optional[PerformanceMetrics]:
        """Получить метрики последнего выполнения"""
        return self._last_metrics
    
    def print_metrics(self) -> None:
        """Вывести метрики последнего выполнения"""
        if self._last_metrics:
            print(f"\n📊 [{self.name}] {self._last_metrics}")
        else:
            print(f"No metrics available for {self.name}")
    
    @contextmanager
    def measure(self):
        """Контекстный менеджер для замера производительности (без разбиения)"""
        tracemalloc.start()
        start_time = time.perf_counter()
        
        yield
        
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return PerformanceMetrics(
            time_seconds=end_time - start_time,
            memory_mb=peak / 1024 / 1024
        )
    
    def __repr__(self) -> str:
        return f"{self.name}()"


class PartitionerWithStats(BasePartitioner):
    """
    Расширенный базовый класс с дополнительной статистикой
    
    Добавляет:
    - Количество итераций
    - Время по этапам
    - Историю улучшений
    """
    
    def __init__(self, name: str = "PartitionerWithStats"):
        super().__init__(name)
        self._iterations: int = 0
        self._stage_times: Dict[str, float] = {}
        self._improvement_history: list = []
    
    def _reset_stats(self) -> None:
        """Сброс статистики"""
        self._iterations = 0
        self._stage_times = {}
        self._improvement_history = []
    
    def _record_iteration(self, cut_weight: int) -> None:
        """Запись итерации"""
        self._iterations += 1
        self._improvement_history.append(cut_weight)
    
    @contextmanager
    def _stage(self, name: str):
        """Контекстный менеджер для замера этапа"""
        start = time.perf_counter()
        yield
        self._stage_times[name] = time.perf_counter() - start
    
    def partition(self, graph: Graph, balance_ratio: float = 0.5) -> Tuple[Partition, PerformanceMetrics]:
        """Выполнить разбиение с расширенной статистикой"""
        self._reset_stats()
        
        partition, metrics = super().partition(graph, balance_ratio)
        
        # Добавляем дополнительную статистику
        metrics.extra.update({
            'iterations': self._iterations,
            'stage_times': self._stage_times.copy(),
            'improvement_history': self._improvement_history.copy()
        })
        
        return partition, metrics