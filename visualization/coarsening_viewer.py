"""
Визуализация этапов стягивания графа
"""

import tkinter as tk
from tkinter import ttk
from typing import List, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.coarse_graph import CoarseGraph
from .graph_canvas import GraphCanvas


class CoarseningViewer(tk.Toplevel):
    """
    Окно для просмотра этапов стягивания графа
    """
    
    def __init__(self, parent, coarse_graphs: List[CoarseGraph], original_graph: Graph):
        super().__init__(parent)
        
        self.title("Coarsening Stages Viewer")
        self.geometry("1000x700")
        
        self.coarse_graphs = coarse_graphs
        self.original_graph = original_graph
        self.current_level = 0
        
        self._create_widgets()
        self._show_level(0)
    
    def _create_widgets(self):
        """Создание виджетов"""
        # Верхняя панель с управлением
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Coarsening Level:").pack(side=tk.LEFT, padx=5)
        
        self.level_var = tk.IntVar(value=0)
        self.level_spinbox = ttk.Spinbox(
            control_frame, from_=0, to=len(self.coarse_graphs),
            textvariable=self.level_var, width=10, command=self._on_level_change
        )
        self.level_spinbox.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="◀ Previous", 
                  command=self._prev_level).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next ▶", 
                  command=self._next_level).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Show Original", 
                  command=lambda: self._show_original()).pack(side=tk.RIGHT, padx=5)
        
        # Информационная панель
        info_frame = ttk.LabelFrame(self, text="Stage Info", padding=5)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.info_label = ttk.Label(info_frame, text="")
        self.info_label.pack()
        
        # Канвас для отображения графа
        self.canvas = GraphCanvas(self, width=900, height=550)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Привязка событий
        self.level_var.trace('w', lambda *args: self._on_level_change())
    
    def _show_level(self, level: int):
        """Показать уровень стягивания"""
        if level < 0 or level > len(self.coarse_graphs):
            return
        
        self.current_level = level
        self.level_var.set(level)
        
        if level == len(self.coarse_graphs):
            # Показываем исходный граф
            graph = self.original_graph
            info = f"Original Graph: {graph.num_vertices} vertices, {graph.num_edges} edges"
        else:
            # Показываем грубый граф
            coarse = self.coarse_graphs[level]
            graph = coarse.to_graph()
            original_count = coarse.get_original_vertex_count()
            info = f"Level {level + 1}: {graph.num_vertices} vertices, {graph.num_edges} edges | " \
                   f"Represents {original_count} original vertices | " \
                   f"Compression: {graph.num_vertices / original_count:.3f}"
        
        self.info_label.config(text=info)
        self.canvas.set_graph(graph, None)
        self.canvas.randomize_layout()
    
    def _show_original(self):
        """Показать исходный граф"""
        self._show_level(len(self.coarse_graphs))
    
    def _prev_level(self):
        """Предыдущий уровень"""
        if self.current_level > 0:
            self._show_level(self.current_level - 1)
    
    def _next_level(self):
        """Следующий уровень"""
        if self.current_level < len(self.coarse_graphs):
            self._show_level(self.current_level + 1)
    
    def _on_level_change(self):
        """Обработка изменения уровня"""
        self._show_level(self.level_var.get())