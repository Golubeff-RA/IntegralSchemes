"""
Абстрактный базовый класс для всех генераторов графов
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np
import sys
from pathlib import Path

# Добавляем путь к корневой директории
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.graph import Graph


class BaseGraphGenerator(ABC):
    """
    Абстрактный базовый класс для генерации графов
    
    Все генераторы должны наследовать этот класс и реализовать метод generate()
    """
    
    def __init__(self, name: str = "BaseGenerator"):
        """
        Инициализация генератора
        
        Args:
            name: имя генератора (для логирования)
        """
        self.name = name
        self._last_generated_graph = None
        self._generation_params = {}
    
    @abstractmethod
    def generate(self) -> Graph:
        """
        Генерация графа
        
        Returns:
            Graph: сгенерированный граф
        """
        pass
    
    def generate_with_stats(self) -> tuple[Graph, Dict[str, Any]]:
        """
        Генерация графа со статистикой
        
        Returns:
            tuple: (граф, словарь со статистикой генерации)
        """
        import time
        start_time = time.time()
        
        graph = self.generate()
        
        generation_time = time.time() - start_time
        
        stats = {
            'generator': self.name,
            'num_vertices': graph.num_vertices,
            'num_edges': graph.num_edges,
            'density': 2 * graph.num_edges / (graph.num_vertices * (graph.num_vertices - 1)) if graph.num_vertices > 1 else 0,
            'generation_time': generation_time,
            'params': self._generation_params.copy()
        }
        
        # Добавляем статистику по степеням вершин
        degrees = [graph.get_degree(v) for v in range(graph.num_vertices)]
        stats['max_degree'] = max(degrees) if degrees else 0
        stats['min_degree'] = min(degrees) if degrees else 0
        stats['avg_degree'] = np.mean(degrees) if degrees else 0
        
        self._last_generated_graph = graph
        return graph, stats
    
    def save_to_file(self, graph: Graph, filename: str) -> None:
        """
        Сохранение сгенерированного графа в файл
        
        Args:
            graph: граф для сохранения
            filename: путь к файлу
        """
        graph.save_to_file(filename)
        print(f"Graph saved to {filename}")
    
    def get_last_graph(self) -> Optional[Graph]:
        """Получение последнего сгенерированного графа"""
        return self._last_generated_graph
    
    def __repr__(self) -> str:
        return f"{self.name}()"