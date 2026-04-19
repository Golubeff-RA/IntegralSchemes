"""
Панель отображения метрик качества разбиения
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition
from metrics.partition_metrics import PartitionMetrics


class MetricsPanel(ttk.LabelFrame):
    """
    Панель для отображения метрик
    """
    
    def __init__(self, parent):
        super().__init__(parent, text="Metrics", padding=10)
        
        self.current_metrics: Optional[Dict[str, Any]] = None
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Создание виджетов"""
        
        # Основные метрики
        metrics_frame = ttk.Frame(self)
        metrics_frame.pack(fill="x", pady=5)
        
        # Cut Size
        cut_frame = ttk.Frame(metrics_frame)
        cut_frame.pack(fill="x", pady=2)
        ttk.Label(cut_frame, text="✂️ Cut Size:", font=("Arial", 10, "bold")).pack(side="left")
        self.cut_value = ttk.Label(cut_frame, text="0", font=("Arial", 10, "bold"), foreground="#e74c3c")
        self.cut_value.pack(side="right")
        
        # Cut Ratio
        ratio_frame = ttk.Frame(metrics_frame)
        ratio_frame.pack(fill="x", pady=2)
        ttk.Label(ratio_frame, text="📊 Cut Ratio:").pack(side="left")
        self.ratio_value = ttk.Label(ratio_frame, text="0%")
        self.ratio_value.pack(side="right")
        
        # Balance
        balance_frame = ttk.Frame(metrics_frame)
        balance_frame.pack(fill="x", pady=2)
        ttk.Label(balance_frame, text="⚖️ Balance Quality:").pack(side="left")
        self.balance_value = ttk.Label(balance_frame, text="0.00")
        self.balance_value.pack(side="right")
        
        # Time
        time_frame = ttk.Frame(metrics_frame)
        time_frame.pack(fill="x", pady=2)
        ttk.Label(time_frame, text="⏱️ Time:").pack(side="left")
        self.time_value = ttk.Label(time_frame, text="0.000 s")
        self.time_value.pack(side="right")
        
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=5)
        
        # Размеры частей
        parts_frame = ttk.LabelFrame(self, text="Part Sizes", padding=5)
        parts_frame.pack(fill="x", pady=5)
        
        self.parts_container = ttk.Frame(parts_frame)
        self.parts_container.pack(fill="x")
        
        self.part_labels = []
        
        ttk.Separator(self, orient="horizontal").pack(fill="x", pady=5)
        
        # Многоуровневые метрики
        self.multilevel_frame = ttk.LabelFrame(self, text="Multilevel Metrics", padding=5)
        self.multilevel_frame.pack(fill="x", pady=5)
        
        # Compression
        comp_frame = ttk.Frame(self.multilevel_frame)
        comp_frame.pack(fill="x", pady=2)
        ttk.Label(comp_frame, text="📉 Compression Ratio:").pack(side="left")
        self.compression_value = ttk.Label(comp_frame, text="N/A")
        self.compression_value.pack(side="right")
        
        # Levels
        levels_frame = ttk.Frame(self.multilevel_frame)
        levels_frame.pack(fill="x", pady=2)
        ttk.Label(levels_frame, text="📊 Coarsening Levels:").pack(side="left")
        self.levels_value = ttk.Label(levels_frame, text="N/A")
        self.levels_value.pack(side="right")
        
        # Stage times
        stages_frame = ttk.LabelFrame(self.multilevel_frame, text="Stage Times", padding=5)
        stages_frame.pack(fill="x", pady=5)
        
        # Coarsening
        coarsen_frame = ttk.Frame(stages_frame)
        coarsen_frame.pack(fill="x", pady=1)
        ttk.Label(coarsen_frame, text="  Coarsening:").pack(side="left")
        self.coarsening_time = ttk.Label(coarsen_frame, text="N/A")
        self.coarsening_time.pack(side="right")
        
        # Initial partition
        initial_frame = ttk.Frame(stages_frame)
        initial_frame.pack(fill="x", pady=1)
        ttk.Label(initial_frame, text="  Initial Partition:").pack(side="left")
        self.initial_time = ttk.Label(initial_frame, text="N/A")
        self.initial_time.pack(side="right")
        
        # Uncoarsening
        uncoarsen_frame = ttk.Frame(stages_frame)
        uncoarsen_frame.pack(fill="x", pady=1)
        ttk.Label(uncoarsen_frame, text="  Uncoarsening:").pack(side="left")
        self.uncoarsening_time = ttk.Label(uncoarsen_frame, text="N/A")
        self.uncoarsening_time.pack(side="right")
    
    def update_metrics(self, graph: Graph, partition: Partition, 
                      time_seconds: float = 0, algorithm_stats: Dict = None):
        """Обновление отображаемых метрик"""
        metrics = PartitionMetrics(graph, partition)
        all_metrics = metrics.get_all_metrics()
        
        # Обновляем основные метрики
        self.cut_value.config(text=str(all_metrics['cut_size']))
        self.ratio_value.config(text=f"{all_metrics['cut_ratio']*100:.2f}%")
        self.balance_value.config(text=f"{all_metrics['balance_quality']:.4f}")
        self.time_value.config(text=f"{time_seconds:.4f} s")
        
        # Обновляем размеры частей
        for widget in self.part_labels:
            widget.destroy()
        self.part_labels.clear()
        
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6"]
        
        for i, size in enumerate(all_metrics['part_sizes']):
            frame = ttk.Frame(self.parts_container)
            frame.pack(fill="x", pady=1)
            
            color = colors[i % len(colors)]
            
            ttk.Label(frame, text=f"Part {i}:", foreground=color, width=8).pack(side="left")
            ttk.Label(frame, text=f"{size} vertices").pack(side="right")
            self.part_labels.append(frame)
        
        # Обновляем метрики многоуровневого алгоритма
        if algorithm_stats:
            if 'compression_ratio' in algorithm_stats:
                self.compression_value.config(text=f"{algorithm_stats['compression_ratio']:.4f}")
            else:
                self.compression_value.config(text="N/A")
                
            if 'coarsening_levels' in algorithm_stats:
                self.levels_value.config(text=str(algorithm_stats['coarsening_levels']))
            else:
                self.levels_value.config(text="N/A")
                
            if 'coarsening_time' in algorithm_stats:
                self.coarsening_time.config(text=f"{algorithm_stats['coarsening_time']:.4f}s")
            else:
                self.coarsening_time.config(text="N/A")
                
            if 'initial_partition_time' in algorithm_stats:
                self.initial_time.config(text=f"{algorithm_stats['initial_partition_time']:.4f}s")
            else:
                self.initial_time.config(text="N/A")
                
            if 'uncoarsening_time' in algorithm_stats:
                self.uncoarsening_time.config(text=f"{algorithm_stats['uncoarsening_time']:.4f}s")
            else:
                self.uncoarsening_time.config(text="N/A")
        else:
            self.compression_value.config(text="N/A")
            self.levels_value.config(text="N/A")
            self.coarsening_time.config(text="N/A")
            self.initial_time.config(text="N/A")
            self.uncoarsening_time.config(text="N/A")
    
    def clear(self):
        """Очистка метрик"""
        self.cut_value.config(text="0")
        self.ratio_value.config(text="0%")
        self.balance_value.config(text="0.00")
        self.time_value.config(text="0.000 s")
        
        for widget in self.part_labels:
            widget.destroy()
        self.part_labels.clear()
        
        self.compression_value.config(text="N/A")
        self.levels_value.config(text="N/A")
        self.coarsening_time.config(text="N/A")
        self.initial_time.config(text="N/A")
        self.uncoarsening_time.config(text="N/A")