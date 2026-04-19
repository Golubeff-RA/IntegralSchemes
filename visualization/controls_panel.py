"""
Панель управления для настройки тестов и запуска алгоритмов
"""

import tkinter as tk
from tkinter import ttk, filedialog
from typing import Optional, Callable

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph


class ControlsPanel(ttk.LabelFrame):
    """
    Панель управления
    """
    
    def __init__(self, parent, on_load_graph: Callable, on_run_algorithm: Callable):
        super().__init__(parent, text="Controls", padding=10)
        
        self.on_load_graph = on_load_graph
        self.on_run_algorithm = on_run_algorithm
        
        # Дополнительные колбэки
        self.on_reset_view = None
        self.on_randomize_layout = None
        self.on_show_coarsening = None
        
        # Переменные
        self._balance_ratio = tk.DoubleVar(value=0.5)
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Создание виджетов"""
        
        # === Информация о графе ===
        info_frame = ttk.Frame(self)
        info_frame.pack(fill="x", pady=5)
        
        ttk.Label(info_frame, text="Graph:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        self.graph_info_label = ttk.Label(info_frame, text="No graph loaded", foreground="#888888")
        self.graph_info_label.pack(side=tk.LEFT, padx=5)
        
        # === Загрузка графа ===
        load_frame = ttk.LabelFrame(self, text="Load Graph", padding=5)
        load_frame.pack(fill="x", pady=5)
        
        ttk.Button(load_frame, text="📂 Load from File", 
                  command=self._load_from_file).pack(fill="x", pady=2)
        
        # === Генерация графа ===
        gen_frame = ttk.LabelFrame(self, text="Generate Graph", padding=5)
        gen_frame.pack(fill="x", pady=5)
        
        # Тип графа
        ttk.Label(gen_frame, text="Type:").grid(row=0, column=0, sticky="w", pady=2)
        self.graph_type = ttk.Combobox(gen_frame, values=["Cluster", "Barabasi-Albert", "Erdos-Renyi"])
        self.graph_type.grid(row=0, column=1, sticky="ew", pady=2)
        self.graph_type.current(0)
        
        # Количество вершин
        ttk.Label(gen_frame, text="Vertices:").grid(row=1, column=0, sticky="w", pady=2)
        self.num_vertices = tk.IntVar(value=50)
        ttk.Spinbox(gen_frame, from_=10, to=500, textvariable=self.num_vertices,
                   width=10).grid(row=1, column=1, sticky="w", pady=2)
        
        # Количество кластеров
        ttk.Label(gen_frame, text="Clusters:").grid(row=2, column=0, sticky="w", pady=2)
        self.num_clusters = tk.IntVar(value=3)
        ttk.Spinbox(gen_frame, from_=2, to=20, textvariable=self.num_clusters,
                   width=10).grid(row=2, column=1, sticky="w", pady=2)
        
        # Плотность
        ttk.Label(gen_frame, text="Density:").grid(row=3, column=0, sticky="w", pady=2)
        self.density = tk.DoubleVar(value=0.1)
        
        density_frame = ttk.Frame(gen_frame)
        density_frame.grid(row=3, column=1, sticky="ew", pady=2)
        
        density_scale = ttk.Scale(density_frame, from_=0.01, to=0.5, variable=self.density,
                                  orient="horizontal")
        density_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.density_label = ttk.Label(density_frame, text="0.10", width=5)
        self.density_label.pack(side=tk.RIGHT, padx=5)
        
        def update_density_label(*args):
            self.density_label.config(text=f"{self.density.get():.2f}")
        self.density.trace('w', update_density_label)
        
        ttk.Button(gen_frame, text="✨ Generate", 
                  command=self._generate_graph).grid(row=4, column=0, columnspan=2, pady=5)
        
        gen_frame.columnconfigure(1, weight=1)
        
        # === Алгоритм ===
        algo_frame = ttk.LabelFrame(self, text="Algorithm Settings", padding=5)
        algo_frame.pack(fill="x", pady=5)
        
        ttk.Label(algo_frame, text="Algorithm:").grid(row=0, column=0, sticky="w", pady=2)
        self.algorithm = ttk.Combobox(algo_frame, values=["Kernighan-Lin", "Multilevel", "Compare Both"])
        self.algorithm.grid(row=0, column=1, sticky="ew", pady=2)
        self.algorithm.current(0)
        
        ttk.Label(algo_frame, text="Parts (k):").grid(row=1, column=0, sticky="w", pady=2)
        self.num_parts = tk.IntVar(value=2)
        ttk.Spinbox(algo_frame, from_=2, to=10, textvariable=self.num_parts,
                   width=10).grid(row=1, column=1, sticky="w", pady=2)
        
        ttk.Label(algo_frame, text="Balance:").grid(row=2, column=0, sticky="w", pady=2)
        
        balance_frame = ttk.Frame(algo_frame)
        balance_frame.grid(row=2, column=1, sticky="ew", pady=2)
        
        balance_scale = ttk.Scale(balance_frame, from_=0.3, to=0.7, variable=self._balance_ratio,
                                  orient="horizontal")
        balance_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.balance_label = ttk.Label(balance_frame, text="0.50", width=5)
        self.balance_label.pack(side=tk.RIGHT, padx=5)
        
        def update_balance_label(*args):
            self.balance_label.config(text=f"{self._balance_ratio.get():.2f}")
        self._balance_ratio.trace('w', update_balance_label)
        
        ttk.Button(algo_frame, text="▶ Run Algorithm", 
                  command=self._run_algorithm).grid(row=3, column=0, columnspan=2, pady=10)
        
        # Кнопка показа этапов стягивания
        ttk.Button(algo_frame, text="📊 Show Coarsening Stages", 
                  command=self._show_coarsening).grid(row=4, column=0, columnspan=2, pady=5)
        
        algo_frame.columnconfigure(1, weight=1)
        
        # === Вид ===
        view_frame = ttk.LabelFrame(self, text="View Settings", padding=5)
        view_frame.pack(fill="x", pady=5)
        
        self.show_labels = tk.BooleanVar(value=True)
        ttk.Checkbutton(view_frame, text="Show vertex labels", 
                       variable=self.show_labels, command=self._update_view).pack(anchor="w")
        
        self.show_edge_weights = tk.BooleanVar(value=False)
        ttk.Checkbutton(view_frame, text="Show edge weights", 
                       variable=self.show_edge_weights, command=self._update_view).pack(anchor="w")
        
        self.highlight_cuts = tk.BooleanVar(value=True)
        ttk.Checkbutton(view_frame, text="Highlight cut edges", 
                       variable=self.highlight_cuts, command=self._update_view).pack(anchor="w")
        
        ttk.Button(view_frame, text="🔄 Reset View", 
                  command=self._reset_view).pack(fill="x", pady=2)
        
        ttk.Button(view_frame, text="🎲 Randomize Layout", 
                  command=self._randomize_layout).pack(fill="x", pady=2)
    
    def _load_from_file(self):
        """Загрузка графа из файла"""
        filename = filedialog.askopenfilename(
            title="Select graph file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            try:
                graph = Graph.load_from_file(filename)
                self.on_load_graph(graph, None)
                self.graph_info_label.config(text=f"V: {graph.num_vertices} | E: {graph.num_edges}")
            except Exception as e:
                if hasattr(self, 'on_error'):
                    self.on_error(f"Failed to load graph: {str(e)}")
    
    def _generate_graph(self):
        """Генерация тестового графа"""
        try:
            from data.generators import ClusterGraphGenerator, BarabasiAlbertGenerator, ErdosRenyiGenerator
            
            n = self.num_vertices.get()
            graph_type = self.graph_type.get()
            
            if graph_type == "Cluster":
                gen = ClusterGraphGenerator(
                    num_clusters=self.num_clusters.get(),
                    cluster_size=n // self.num_clusters.get(),
                    intra_prob=self.density.get() * 5,
                    inter_prob=self.density.get(),
                    seed=42
                )
            elif graph_type == "Barabasi-Albert":
                gen = BarabasiAlbertGenerator(n=n, m0=5, m=int(n * self.density.get()), seed=42)
            else:
                gen = ErdosRenyiGenerator(n=n, p=self.density.get(), seed=42)
            
            graph = gen.generate()
            self.on_load_graph(graph, None)
            self.graph_info_label.config(text=f"V: {graph.num_vertices} | E: {graph.num_edges}")
            
        except Exception as e:
            if hasattr(self, 'on_error'):
                self.on_error(f"Failed to generate graph: {str(e)}")
    
    def _run_algorithm(self):
        """Запуск алгоритма"""
        algo = self.algorithm.get()
        num_parts = self.num_parts.get()
        balance = self._balance_ratio.get()
        self.on_run_algorithm(algo, num_parts, balance)
    
    def _show_coarsening(self):
        """Показать этапы стягивания"""
        if self.on_show_coarsening:
            self.on_show_coarsening()
    
    def _reset_view(self):
        if self.on_reset_view:
            self.on_reset_view()
    
    def _randomize_layout(self):
        if self.on_randomize_layout:
            self.on_randomize_layout()
    
    def _update_view(self):
        """Обновление настроек отображения"""
        if hasattr(self, 'on_update_view'):
            self.on_update_view()
    
    def get_view_settings(self) -> dict:
        return {
            'show_labels': self.show_labels.get(),
            'show_edge_weights': self.show_edge_weights.get(),
            'highlight_cuts': self.highlight_cuts.get()
        }