"""
Главное окно GUI приложения для визуализации разбиения графов
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
from typing import Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition
from algorithms import KernighanLin, MultilevelPartitioner
from metrics.partition_metrics import PartitionMetrics
from .graph_canvas import GraphCanvas
from .controls_panel import ControlsPanel
from .metrics_panel import MetricsPanel
from .coarsening_viewer import CoarseningViewer


class GraphPartitioningGUI:
    """
    Главное окно приложения
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Graph Partitioning Visualizer")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)
        
        # Тёмная тема
        self._setup_theme()
        
        # Состояние
        self.current_graph: Optional[Graph] = None
        self.current_partition: Optional[Partition] = None
        self.current_stats: Optional[dict] = None
        self.current_ml: Optional[MultilevelPartitioner] = None
        self.algorithm_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Для non-blocking сообщений
        self.message_queue = []
        self._process_message_queue()
        
        # Создание интерфейса
        self._create_layout()
        
        # Привязка событий
        self._bind_events()
        
        # Статус бар
        self.status_var = tk.StringVar(value="Ready. Load or generate a graph to begin.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                               relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Создаём тестовый граф
        self._create_demo_graph()
    
    def _setup_theme(self):
        """Настройка тёмной темы"""
        style = ttk.Style()
        
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')
        
        bg_color = "#1e1e1e"
        fg_color = "#d4d4d4"
        select_color = "#264f78"
        
        style.configure(".", background=bg_color, foreground=fg_color)
        style.configure("TLabel", background=bg_color, foreground=fg_color)
        style.configure("TFrame", background=bg_color)
        style.configure("TLabelframe", background=bg_color, foreground=fg_color, 
                       bordercolor="#3c3c3c")
        style.configure("TLabelframe.Label", background=bg_color, foreground=fg_color)
        style.configure("TButton", background="#3c3c3c", foreground=fg_color,
                       bordercolor="#3c3c3c", focuscolor="none")
        style.map("TButton", 
                 background=[("active", "#505050"), ("pressed", "#2d2d2d")])
        style.configure("TEntry", fieldbackground="#2d2d2d", foreground=fg_color,
                       insertcolor=fg_color)
        style.configure("TCombobox", fieldbackground="#2d2d2d", foreground=fg_color,
                       selectbackground=select_color)
        style.configure("TSpinbox", fieldbackground="#2d2d2d", foreground=fg_color,
                       selectbackground=select_color)
        style.configure("TScale", background=bg_color, troughcolor="#2d2d2d",
                       slidercolor="#3c3c3c")
        
        self.root.configure(bg=bg_color)
    
    def _create_layout(self):
        """Создание layout интерфейса"""
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Левая панель
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Панель управления
        self.controls = ControlsPanel(left_frame, self._on_load_graph, self._on_run_algorithm)
        self.controls.pack(fill=tk.X, padx=5, pady=5)
        
        # Привязываем дополнительные обработчики
        self.controls.on_reset_view = self._on_reset_view
        self.controls.on_randomize_layout = self._on_randomize_layout
        self.controls.on_show_coarsening = self._on_show_coarsening
        
        # Панель метрик
        self.metrics = MetricsPanel(left_frame)
        self.metrics.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Правая панель
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)
        
        # Канвас для графа
        self.canvas = GraphCanvas(right_frame, width=900, height=700)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Привязываем обработчики канваса
        self.canvas.on_node_selected = self._on_node_selected
        self.canvas.on_node_right_click = self._on_node_right_click
    
    def _bind_events(self):
        """Привязка глобальных событий"""
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Control-o>", lambda e: self.controls._load_from_file())
        self.root.bind("<Control-g>", lambda e: self.controls._generate_graph())
        self.root.bind("<Control-r>", lambda e: self.controls._run_algorithm())
        self.root.bind("<Control-q>", lambda e: self._on_close())
    
    def _show_message(self, msg_type: str, title: str, message: str, duration: int = 3000):
        """Non-blocking сообщение"""
        self.message_queue.append({
            'type': msg_type,
            'title': title,
            'message': message,
            'duration': duration,
            'time': time.time()
        })
    
    def _process_message_queue(self):
        """Обработка очереди сообщений"""
        current_time = time.time()
        new_queue = []
        
        for msg in self.message_queue:
            if current_time - msg['time'] < msg['duration']:
                new_queue.append(msg)
            else:
                # Показываем сообщение в статус баре вместо messagebox
                self.status_var.set(f"{msg['title']}: {msg['message']}")
        
        self.message_queue = new_queue
        self.root.after(500, self._process_message_queue)
    
    def _create_demo_graph(self):
        """Создание демонстрационного графа"""
        try:
            from data.generators import ClusterGraphGenerator
            
            gen = ClusterGraphGenerator(
                num_clusters=3,
                cluster_size=10,
                intra_prob=0.7,
                inter_prob=0.05,
                seed=42
            )
            
            graph = gen.generate()
            self._on_load_graph(graph, None)
            self._show_message("info", "Demo", f"Demo graph loaded: {graph.num_vertices} vertices, {graph.num_edges} edges", 2000)
        except Exception as e:
            # Если генератор не работает, создаём простой граф вручную
            graph = Graph(15)
            edges = [(0,1), (0,2), (1,3), (2,4), (3,5), (4,6), (5,7), (6,8), 
                     (7,9), (8,10), (9,11), (10,12), (11,13), (12,14), (0,14),
                     (1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14)]
            for u, v in edges:
                graph.add_edge(u, v)
            self._on_load_graph(graph, None)
    
    def _on_load_graph(self, graph: Graph, partition: Optional[Partition] = None):
        """Обработка загрузки графа"""
        self.current_graph = graph
        self.current_partition = partition
        self.current_stats = None
        self.current_ml = None
        
        self.canvas.set_graph(graph, partition)
        
        if partition:
            self.metrics.update_metrics(graph, partition, 0)
        
        if hasattr(self.controls, 'graph_info_label'):
            self.controls.graph_info_label.config(
                text=f"V: {graph.num_vertices} | E: {graph.num_edges}"
            )
        
        self.status_var.set(f"Loaded graph: {graph.num_vertices} vertices, {graph.num_edges} edges")
    
    def _on_run_algorithm(self, algorithm_name: str, num_parts: int, balance_ratio: float):
        """Запуск алгоритма"""
        if self.current_graph is None:
            self._show_message("warning", "Warning", "Please load or generate a graph first.", 2000)
            return
        
        if self.is_running:
            self._show_message("warning", "Warning", "Please wait for the current algorithm to finish.", 2000)
            return
        
        self.is_running = True
        self.status_var.set(f"Running {algorithm_name} with k={num_parts}...")
        self.root.config(cursor="watch")
        
        # Запускаем в отдельном потоке
        self.algorithm_thread = threading.Thread(
            target=self._run_algorithm_thread,
            args=(algorithm_name, num_parts, balance_ratio),
            daemon=True
        )
        self.algorithm_thread.start()
        
        self._check_algorithm_thread()
    
    def _run_algorithm_thread(self, algorithm_name: str, num_parts: int, balance_ratio: float):
        """Выполнение алгоритма в потоке"""
        start_time = time.time()
        
        try:
            if algorithm_name == "Kernighan-Lin":
                algorithm = KernighanLin(max_passes=10, max_iterations=100)
                partition = algorithm.partition(self.current_graph, num_parts, balance_ratio)
                stats = algorithm.get_statistics()
                stats['partition_time'] = time.time() - start_time
                algo_stats = None
                self.current_ml = None
                
            elif algorithm_name == "Multilevel":
                self.current_ml = MultilevelPartitioner(
                    min_coarse_vertices=10,
                    max_levels=10
                )
                partition, stats = self.current_ml.partition_with_stats(
                    self.current_graph, num_parts, balance_ratio
                )
                algo_stats = stats
                
            else:  # Compare Both
                kl = KernighanLin(max_passes=10)
                partition_kl = kl.partition(self.current_graph, num_parts, balance_ratio)
                cut_kl = partition_kl.cut_edges(self.current_graph)
                
                self.current_ml = MultilevelPartitioner(min_coarse_vertices=10)
                partition_ml, stats_ml = self.current_ml.partition_with_stats(
                    self.current_graph, num_parts, balance_ratio
                )
                cut_ml = partition_ml.cut_edges(self.current_graph)
                
                if cut_ml < cut_kl:
                    partition = partition_ml
                    stats = stats_ml
                    algo_stats = stats_ml
                else:
                    partition = partition_kl
                    stats = {'final_cut': cut_kl, 'partition_time': time.time() - start_time}
                    algo_stats = None
            
            elapsed = time.time() - start_time
            
            self.current_partition = partition
            self.current_stats = stats
            
            self.root.after(0, self._on_algorithm_complete, partition, elapsed, algo_stats)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self._show_message("error", "Error", f"Algorithm failed: {str(e)}", 5000))
            self.root.after(0, lambda: setattr(self, 'is_running', False))
            self.root.after(0, lambda: self.root.config(cursor=""))
    
    def _check_algorithm_thread(self):
        """Проверка завершения потока"""
        if self.algorithm_thread and self.algorithm_thread.is_alive():
            self.root.after(100, self._check_algorithm_thread)
        else:
            self.is_running = False
            self.root.config(cursor="")
    
    def _on_algorithm_complete(self, partition: Partition, elapsed: float, algo_stats: dict):
        """Обработка завершения алгоритма"""
        self.current_partition = partition
        self.canvas.set_partition(partition)
        self.metrics.update_metrics(self.current_graph, partition, elapsed, algo_stats)
        
        cut_size = partition.cut_edges(self.current_graph)
        balance = partition.balance_quality()
        self.status_var.set(f"✓ Algorithm completed | Cut size: {cut_size} | Balance: {balance:.3f} | Time: {elapsed:.3f}s")
        
        view_settings = self.controls.get_view_settings()
        self.canvas.show_vertex_labels = view_settings['show_labels']
        self.canvas.show_edge_labels = view_settings['show_edge_weights']
        self.canvas.highlight_cut_edges = view_settings['highlight_cuts']
        self.canvas.redraw()
    
    def _on_node_selected(self, node_id: int):
        """Обработка выбора вершины"""
        if self.current_graph:
            degree = self.current_graph.get_degree(node_id)
            part = -1
            if self.current_partition:
                part = self.current_partition.get_part(node_id)
            weight = self.current_graph.get_vertex_weight(node_id)
            self.status_var.set(f"Vertex {node_id} | Degree: {degree} | Weight: {weight} | Part: {part if part >= 0 else 'unassigned'}")
    
    def _on_node_right_click(self, node_id: int, x: int, y: int):
        """Обработка правого клика по вершине"""
        menu = tk.Menu(self.root, tearoff=0, bg="#2d2d2d", fg="#d4d4d4")
        menu.add_command(label=f"Info: Vertex {node_id}", command=lambda: self._show_vertex_info_in_status(node_id))
        menu.add_separator()
        
        if self.current_partition and self.current_partition.num_parts > 1:
            current_part = self.current_partition.get_part(node_id)
            if current_part >= 0:
                for part in range(self.current_partition.num_parts):
                    if part != current_part:
                        menu.add_command(
                            label=f"Move to Part {part}",
                            command=lambda p=part: self._move_vertex(node_id, p)
                        )
        
        menu.post(x, y)
    
    def _show_vertex_info_in_status(self, node_id: int):
        """Показ информации о вершине в статус баре"""
        if not self.current_graph:
            return
        
        degree = self.current_graph.get_degree(node_id)
        weight = self.current_graph.get_vertex_weight(node_id)
        part = self.current_partition.get_part(node_id) if self.current_partition else -1
        
        neighbors = list(self.current_graph.get_neighbors(node_id).items())[:5]
        neighbors_str = ", ".join(f"{n}(w={w})" for n, w in neighbors)
        if len(self.current_graph.get_neighbors(node_id)) > 5:
            neighbors_str += "..."
        
        self.status_var.set(f"Vertex {node_id} | Weight: {weight} | Degree: {degree} | Part: {part} | Neighbors: {neighbors_str}")
    
    def _move_vertex(self, node_id: int, new_part: int):
        """Перемещение вершины в другую часть"""
        if self.current_partition:
            delta = self.current_partition.move_vertex(node_id, new_part, self.current_graph)
            self.canvas.set_partition(self.current_partition)
            self.metrics.update_metrics(self.current_graph, self.current_partition, 0, self.current_stats)
            
            if delta < 0:
                self.status_var.set(f"Moved vertex {node_id} to part {new_part} | Cut decreased by {-delta}")
            elif delta > 0:
                self.status_var.set(f"Moved vertex {node_id} to part {new_part} | Cut increased by {delta}")
            else:
                self.status_var.set(f"Moved vertex {node_id} to part {new_part} | Cut unchanged")
    
    def _on_reset_view(self):
        """Сброс вида канваса"""
        self.canvas.reset_view()
        self.status_var.set("View reset")
    
    def _on_randomize_layout(self):
        """Случайное расположение вершин"""
        self.canvas.randomize_layout()
        self.status_var.set("Layout randomized")
    
    def _on_show_coarsening(self):
        """Показать окно с этапами стягивания"""
        if self.current_ml and self.current_graph:
            coarse_graphs = self.current_ml.get_coarsening_history()
            if coarse_graphs:
                CoarseningViewer(self.root, coarse_graphs, self.current_graph)
                self.status_var.set(f"Showing {len(coarse_graphs)} coarsening stages")
            else:
                self.status_var.set("No coarsening stages available - graph may be too small")
        else:
            self.status_var.set("Run Multilevel algorithm first to see coarsening stages")
    
    def _on_close(self):
        """Обработка закрытия окна"""
        self.root.destroy()
    
    def run(self):
        """Запуск приложения"""
        self.root.mainloop()