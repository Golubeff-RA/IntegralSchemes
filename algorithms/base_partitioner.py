"""
Абстрактный базовый класс для всех алгоритмов разбиения графов
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import time
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition


class BasePartitioner(ABC):
    """
    Абстрактный класс для алгоритмов разбиения
    
    Все алгоритмы разбиения должны наследовать этот класс и реализовать метод partition()
    """
    
    def __init__(self, name: str = "BasePartitioner"):
        """
        Инициализация алгоритма разбиения
        
        Args:
            name: имя алгоритма (для логирования)
        """
        self.name = name
        self._statistics = {
            'partition_time': 0,
            'initial_cut': 0,
            'final_cut': 0,
            'iterations': 0,
            'memory_usage_mb': 0
        }
    
    @abstractmethod
    def partition(self, graph: Graph, num_parts: int = 2, 
                  balance_ratio: float = 0.5) -> Partition:
        """
        Разбиение графа на части
        
        Args:
            graph: граф для разбиения
            num_parts: количество частей
            balance_ratio: допустимый дисбаланс (0.5 = строго равные части)
        
        Returns:
            Partition: объект разбиения
        """
        pass
    
    def partition_with_stats(self, graph: Graph, num_parts: int = 2,
                            balance_ratio: float = 0.5) -> Tuple[Partition, Dict[str, Any]]:
        """
        Разбиение графа со сбором статистики
        
        Args:
            graph: граф для разбиения
            num_parts: количество частей
            balance_ratio: допустимый дисбаланс
        
        Returns:
            Tuple[Partition, Dict]: (разбиение, статистика)
        """
        import psutil
        import os
        
        # Замеряем память до
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        
        start_time = time.time()
        
        partition = self.partition(graph, num_parts, balance_ratio)
        
        end_time = time.time()
        mem_after = process.memory_info().rss / 1024 / 1024
        
        # Собираем статистику
        self._statistics['partition_time'] = end_time - start_time
        self._statistics['memory_usage_mb'] = mem_after - mem_before
        self._statistics['final_cut'] = partition.cut_edges(graph)
        
        # Дополнительная статистика
        stats = {
            **self._statistics,
            'num_vertices': graph.num_vertices,
            'num_edges': graph.num_edges,
            'num_parts': num_parts,
            'balance_ratio': balance_ratio,
            'actual_balance': partition.balance_quality(),
            'is_balanced': partition.is_balanced(balance_ratio),
            'part_sizes': partition.part_sizes.tolist()
        }
        
        return partition, stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики последнего запуска"""
        return self._statistics.copy()
    
    def reset_statistics(self):
        """Сброс статистики"""
        self._statistics = {
            'partition_time': 0,
            'initial_cut': 0,
            'final_cut': 0,
            'iterations': 0,
            'memory_usage_mb': 0
        }
    
    def __repr__(self) -> str:
        return f"{self.name}()"