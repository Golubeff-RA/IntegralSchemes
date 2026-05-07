#!/usr/bin/env python3
"""
Полнофункциональный GUI для визуализации и сравнения алгоритмов разбиения графов
"""

import sys
import threading
import time
import random
from math import cos, sin, pi
from tkinter import *
from typing import Dict, List
from tkinter import ttk, messagebox, scrolledtext, filedialog

sys.path.append('.')

from core.graph import Graph
from core.coarse_graph import CoarseGraph
from core.partition import Partition
from algorithms.kernighan_lin import KernighanLin
from algorithms.multilevel import FastMultilevelPartitioner
from algorithms.multilevel_slow import MultilevelPartitioner
from data.generators import FastClusterGenerator, ClusterGraphGenerator, BarabasiAlbertGenerator


class GraphVisualizer:
    """Визуализация графа с возможностью перетаскивания вершин"""
    
    def __init__(self, parent, width=800, height=600):
        self.canvas = Canvas(parent, width=width, height=height, bg='#1e1e1e', highlightthickness=0)
        self.canvas.pack(fill=BOTH, expand=True)
        
        self.width = width
        self.height = height
        self.graph = None
        self.partition = None
        self.nodes = []  # [x, y, id, part, color, weight]
        self.node_radius = 18
        self.selected_node = None
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.drag_start = None
        
        self.part_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#95a5a6']
        self.show_weights = True
        
        self._bind_events()
    
    def _bind_events(self):
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Control-Button-1>", self.on_pan_start)
        self.canvas.bind("<Control-B1-Motion>", self.on_pan)
    
    def set_graph(self, graph: Graph, partition: Partition = None):
        self.graph = graph
        self.partition = partition
        self.compute_layout()
        self.redraw()
    
    def compute_layout(self):
        """Force-directed layout для красивого расположения"""
        n = self.graph.num_vertices
        if n == 0:
            return
        
        # Случайные начальные позиции
        random.seed(42)
        pos = [[random.uniform(100, self.width - 100), random.uniform(100, self.height - 100)] for _ in range(n)]
        
        # Несколько итераций силового алгоритма
        for _ in range(50):
            forces = [[0, 0] for _ in range(n)]
            
            # Отталкивание между всеми вершинами
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i][0] - pos[j][0]
                    dy = pos[i][1] - pos[j][1]
                    dist = max((dx*dx + dy*dy) ** 0.5, 1)
                    force = 500 / (dist * dist)
                    fx = force * dx / dist
                    fy = force * dy / dist
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy
            
            # Притяжение по рёбрам
            for u, v, w in self.graph.edges():
                dx = pos[u][0] - pos[v][0]
                dy = pos[u][1] - pos[v][1]
                dist = max((dx*dx + dy*dy) ** 0.5, 1)
                force = 0.05 * w * dist
                fx = force * dx / dist
                fy = force * dy / dist
                forces[u][0] -= fx
                forces[u][1] -= fy
                forces[v][0] += fx
                forces[v][1] += fy
            
            # Обновление позиций
            for i in range(n):
                pos[i][0] += forces[i][0] * 0.1
                pos[i][1] += forces[i][1] * 0.1
                # Границы
                pos[i][0] = max(50, min(self.width - 50, pos[i][0]))
                pos[i][1] = max(50, min(self.height - 50, pos[i][1]))
        
        # Сохраняем позиции
        self.nodes = []
        for i in range(n):
            part = self.partition.get_part(i) if self.partition else -1
            color = self.part_colors[part % len(self.part_colors)] if part >= 0 else '#7f8c8d'
            weight = self.graph.get_vertex_weight(i)
            self.nodes.append([pos[i][0], pos[i][1], i, part, color, weight])
    
    def redraw(self):
        self.canvas.delete("all")
        if self.graph is None:
            return
        
        # Рисуем рёбра
        for u, v, w in self.graph.edges():
            if u < len(self.nodes) and v < len(self.nodes):
                x1, y1 = self.transform(self.nodes[u][0], self.nodes[u][1])
                x2, y2 = self.transform(self.nodes[v][0], self.nodes[v][1])
                
                # Определяем цвет ребра
                if self.partition and self.partition.get_part(u) != self.partition.get_part(v):
                    color = '#e74c3c'  # Красный - разрезанное
                    width = 2
                else:
                    color = '#555555'
                    width = 1
                
                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)
                
                # Показываем вес ребра
                if self.show_weights and w != 1:
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    self.canvas.create_text(mx, my, text=str(w), fill='#888888', font=('Arial', 9))
        
        # Рисуем вершины
        for x, y, vid, part, color, weight in self.nodes:
            sx, sy = self.transform(x, y)
            r = self.node_radius * self.zoom
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill=color, outline='white', width=2)
            self.canvas.create_text(sx, sy, text=str(vid), fill='white', font=('Arial', int(11 * self.zoom), 'bold'))
            
            # Показываем вес вершины
            if self.show_weights and weight != 1:
                self.canvas.create_text(sx, sy + r + 5, text=f"w={weight}", fill='#aaaaaa', font=('Arial', int(8 * self.zoom)))
    
    def transform(self, x, y):
        return (x * self.zoom + self.offset_x, y * self.zoom + self.offset_y)
    
    def on_click(self, event):
        x, y = event.x, event.y
        best_dist = self.node_radius * self.zoom + 5
        self.selected_node = None
        
        for i, (nx, ny, vid, part, color, weight) in enumerate(self.nodes):
            sx, sy = self.transform(nx, ny)
            dist = ((sx - x) ** 2 + (sy - y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                self.selected_node = i
        
        if self.selected_node is not None:
            self.canvas.config(cursor='hand2')
            self.drag_start = (event.x, event.y)
    
    def on_drag(self, event):
        if self.selected_node is not None:
            dx = (event.x - self.drag_start[0]) / self.zoom
            dy = (event.y - self.drag_start[1]) / self.zoom
            self.nodes[self.selected_node][0] += dx
            self.nodes[self.selected_node][1] += dy
            self.drag_start = (event.x, event.y)
            self.redraw()
    
    def on_release(self, event):
        self.selected_node = None
        self.canvas.config(cursor='')
    
    def on_right_click(self, event):
        if self.partition and self.selected_node is not None:
            menu = Menu(self.canvas, tearoff=0)
            menu.add_command(label="Move to Part 0", command=lambda: self.move_vertex_to(0))
            menu.add_command(label="Move to Part 1", command=lambda: self.move_vertex_to(1))
            menu.post(event.x_root, event.y_root)
    
    def move_vertex_to(self, part):
        if self.selected_node is not None and self.partition:
            vid = self.nodes[self.selected_node][2]
            self.partition.move_vertex_to(vid, part, self.graph)
            self.nodes[self.selected_node][3] = part
            self.nodes[self.selected_node][4] = self.part_colors[part % len(self.part_colors)]
            self.redraw()
    
    def on_mousewheel(self, event):
        if event.delta > 0 or event.num == 4:
            self.zoom *= 1.1
        else:
            self.zoom *= 0.9
        self.zoom = max(0.2, min(3.0, self.zoom))
        self.redraw()
    
    def on_pan_start(self, event):
        self.pan_start = (event.x, event.y)
    
    def on_pan(self, event):
        if hasattr(self, 'pan_start'):
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.pan_start = (event.x, event.y)
            self.redraw()
    
    def reset_view(self):
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.compute_layout()
        self.redraw()
    
    def randomize_layout(self):
        for node in self.nodes:
            node[0] = random.uniform(50, self.width - 50)
            node[1] = random.uniform(50, self.height - 50)
        self.redraw()
    
    def toggle_weights(self):
        self.show_weights = not self.show_weights
        self.redraw()
    
    def set_partition(self, partition):
        self.partition = partition
        for i, node in enumerate(self.nodes):
            part = partition.get_part(node[2])
            node[3] = part
            node[4] = self.part_colors[part % len(self.part_colors)] if part >= 0 else '#7f8c8d'
        self.redraw()


class GraphPartitioningGUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Graph Partitioning Visualizer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        self.current_graph = None
        self.current_partition = None
        self.current_ml = None
        self.levels = None
        
        self._create_widgets()
        self._setup_logging()
        self._create_menu()
    def _update_stage_display(self):
        """Обновление отображения текущего уровня"""
        idx = self.stage_index.get()
        
        if idx == len(self.display_levels):
            self._show_original_stage()
            return
        
        if idx < 0 or idx >= len(self.display_levels):
            return
        
        level = self.display_levels[idx]
        graph = level.graph
        coarse_graph = level.coarse_graph
        reverse_map = level.reverse_map
        
        self.stage_info.config(text=f"Level {idx}: {graph.num_vertices} vertices, {graph.num_edges} edges | "
                                    f"Compression: {level.compression_ratio:.3f} | "
                                    f"Next level: {coarse_graph.num_vertices} vertices")
        
        self._draw_coarsening_stage(graph, coarse_graph, reverse_map)

    def _show_original_stage(self):
        """Показать исходный граф"""
        if not self.display_levels:
            return
        
        original_graph = self.display_levels[0].graph
        self.stage_info.config(text=f"Original Graph: {original_graph.num_vertices} vertices, {original_graph.num_edges} edges")
        self._draw_graph_simple(original_graph)

    def _prev_stage(self):
        idx = self.stage_index.get()
        if idx > 0:
            self.stage_index.set(idx - 1)
            self._update_stage_display()

    def _next_stage(self):
        idx = self.stage_index.get()
        if idx < len(self.display_levels):
            self.stage_index.set(idx + 1)
            self._update_stage_display()
    def _create_menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Graph", command=self.load_graph)
        file_menu.add_command(label="Save Graph", command=self.save_graph)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        view_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Reset View", command=self.reset_view)
        view_menu.add_command(label="Randomize Layout", command=self.randomize_layout)
        view_menu.add_command(label="Toggle Weights", command=self.toggle_weights)
    
    def _create_widgets(self):
        # Главный панель
        main_paned = PanedWindow(self.root, orient=HORIZONTAL, bg='#2c3e50')
        main_paned.pack(fill=BOTH, expand=True)
        
        # Левая панель - управление
        left_frame = Frame(main_paned, bg='#2c3e50', width=400)
        main_paned.add(left_frame, width=400)
        
        # Правая панель - визуализация и логи
        right_paned = PanedWindow(main_paned, orient=VERTICAL, bg='#2c3e50')
        main_paned.add(right_paned)
        
        # Верхняя часть правой панели - визуализация
        self.visualizer = GraphVisualizer(right_paned, width=850, height=550)
        right_paned.add(self.visualizer.canvas)
        
        # Нижняя часть правой панели - логи
        log_frame = LabelFrame(right_paned, text="Console Log", bg='#2c3e50', fg='white', font=('Arial', 10, 'bold'))
        right_paned.add(log_frame)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, bg='#1e1e1e', fg='#00ff00', font=('Courier', 9), height=10)
        self.log_text.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        self._create_controls(left_frame)
        self._create_metrics(left_frame)
    
    def _create_controls(self, parent):
        # === Параметры графа ===
        graph_frame = LabelFrame(parent, text="Graph Parameters", bg='#2c3e50', fg='white', font=('Arial', 10, 'bold'))
        graph_frame.pack(fill=X, padx=10, pady=5)
        
        # Основные параметры
        row = 0
        Label(graph_frame, text="Number of vertices:", bg='#2c3e50', fg='white').grid(row=row, column=0, sticky=W, padx=5, pady=3)
        self.vertices_var = IntVar(value=100)
        Spinbox(graph_frame, from_=10, to=5000, textvariable=self.vertices_var, width=12, bg='#34495e', fg='white').grid(row=row, column=1, padx=5, pady=3)
        
        row += 1
        Label(graph_frame, text="Graph type:", bg='#2c3e50', fg='white').grid(row=row, column=0, sticky=W, padx=5, pady=3)
        self.graph_type_var = StringVar(value="Cluster")
        ttk.Combobox(graph_frame, textvariable=self.graph_type_var, values=["Cluster", "Barabasi-Albert", "Erdos-Renyi"], width=15).grid(row=row, column=1, padx=5, pady=3)
        
        row += 1
        Label(graph_frame, text="Number of clusters:", bg='#2c3e50', fg='white').grid(row=row, column=0, sticky=W, padx=5, pady=3)
        self.clusters_var = IntVar(value=5)
        Spinbox(graph_frame, from_=2, to=50, textvariable=self.clusters_var, width=12, bg='#34495e', fg='white').grid(row=row, column=1, padx=5, pady=3)
        
        # Intra-cluster probability с отображением значения
        row += 1
        Label(graph_frame, text="Intra-cluster prob:", bg='#2c3e50', fg='white').grid(row=row, column=0, sticky=W, padx=5, pady=3)
        self.intra_prob_var = DoubleVar(value=0.8)
        self.intra_prob_label = Label(graph_frame, text="0.80", bg='#2c3e50', fg='#3498db', width=6)
        self.intra_prob_label.grid(row=row, column=1, sticky=E, padx=5, pady=3)
        
        row += 1
        Scale(graph_frame, from_=0.0, to=1.0, resolution=0.01, variable=self.intra_prob_var, 
            orient=HORIZONTAL, bg='#2c3e50', troughcolor='#34495e', sliderlength=20,
            command=lambda x: self.intra_prob_label.config(text=f"{float(x):.2f}")).grid(row=row, column=0, columnspan=2, sticky=EW, padx=5, pady=3)
        
        # Inter-cluster probability с отображением значения
        row += 1
        Label(graph_frame, text="Inter-cluster prob:", bg='#2c3e50', fg='white').grid(row=row, column=0, sticky=W, padx=5, pady=3)
        self.inter_prob_var = DoubleVar(value=0.05)
        self.inter_prob_label = Label(graph_frame, text="0.05", bg='#2c3e50', fg='#3498db', width=6)
        self.inter_prob_label.grid(row=row, column=1, sticky=E, padx=5, pady=3)
        
        row += 1
        Scale(graph_frame, from_=0.0, to=0.3, resolution=0.005, variable=self.inter_prob_var, 
            orient=HORIZONTAL, bg='#2c3e50', troughcolor='#34495e', sliderlength=20,
            command=lambda x: self.inter_prob_label.config(text=f"{float(x):.3f}")).grid(row=row, column=0, columnspan=2, sticky=EW, padx=5, pady=3)
        
        # Vertex weights
        row += 1
        Label(graph_frame, text="Vertex weight range:", bg='#2c3e50', fg='white').grid(row=row, column=0, sticky=W, padx=5, pady=3)
        weight_frame = Frame(graph_frame, bg='#2c3e50')
        weight_frame.grid(row=row, column=1, sticky=W, padx=5, pady=3)
        self.vw_min_var = IntVar(value=1)
        Spinbox(weight_frame, from_=1, to=100, textvariable=self.vw_min_var, width=5, bg='#34495e', fg='white').pack(side=LEFT)
        Label(weight_frame, text="-", bg='#2c3e50', fg='white').pack(side=LEFT, padx=2)
        self.vw_max_var = IntVar(value=10)
        Spinbox(weight_frame, from_=1, to=100, textvariable=self.vw_max_var, width=5, bg='#34495e', fg='white').pack(side=LEFT)
        
        # Edge weights
        row += 1
        Label(graph_frame, text="Edge weight range:", bg='#2c3e50', fg='white').grid(row=row, column=0, sticky=W, padx=5, pady=3)
        weight_frame = Frame(graph_frame, bg='#2c3e50')
        weight_frame.grid(row=row, column=1, sticky=W, padx=5, pady=3)
        self.ew_min_var = IntVar(value=1)
        Spinbox(weight_frame, from_=1, to=100, textvariable=self.ew_min_var, width=5, bg='#34495e', fg='white').pack(side=LEFT)
        Label(weight_frame, text="-", bg='#2c3e50', fg='white').pack(side=LEFT, padx=2)
        self.ew_max_var = IntVar(value=5)
        Spinbox(weight_frame, from_=1, to=100, textvariable=self.ew_max_var, width=5, bg='#34495e', fg='white').pack(side=LEFT)
        
        row += 1
        Button(graph_frame, text="Generate Graph", command=self.generate_graph, bg='#3498db', fg='white', font=('Arial', 10, 'bold')).grid(row=row+1, column=0, columnspan=2, pady=10, sticky=EW)
        
        graph_frame.columnconfigure(1, weight=1)
        
        # === Алгоритм ===
        algo_frame = LabelFrame(parent, text="Algorithm Settings", bg='#2c3e50', fg='white', font=('Arial', 10, 'bold'))
        algo_frame.pack(fill=X, padx=10, pady=5)
        
        algo_row = 0
        Label(algo_frame, text="Algorithm:", bg='#2c3e50', fg='white').grid(row=algo_row, column=0, sticky=W, padx=5, pady=3)
        self.algo_var = StringVar(value="Multilevel")
        ttk.Combobox(algo_frame, textvariable=self.algo_var, values=["Kernighan-Lin", "Multilevel", "Compare"], width=15).grid(row=algo_row, column=1, padx=5, pady=3)
        
        algo_row += 1
        Label(algo_frame, text="Max passes (KL):", bg='#2c3e50', fg='white').grid(row=algo_row, column=0, sticky=W, padx=5, pady=3)
        self.max_passes_var = IntVar(value=20)
        Spinbox(algo_frame, from_=1, to=100, textvariable=self.max_passes_var, width=12, bg='#34495e', fg='white').grid(row=algo_row, column=1, padx=5, pady=3)
        
        # Balance ratio с отображением значения
        algo_row += 1
        Label(algo_frame, text="Balance ratio:", bg='#2c3e50', fg='white').grid(row=algo_row, column=0, sticky=W, padx=5, pady=3)
        self.balance_var = DoubleVar(value=0.5)
        self.balance_label = Label(algo_frame, text="0.50", bg='#2c3e50', fg='#3498db', width=6)
        self.balance_label.grid(row=algo_row, column=1, sticky=E, padx=5, pady=3)
        
        algo_row += 1
        Scale(algo_frame, from_=0.3, to=0.7, resolution=0.01, variable=self.balance_var, 
            orient=HORIZONTAL, bg='#2c3e50', troughcolor='#34495e', sliderlength=20,
            command=lambda x: self.balance_label.config(text=f"{float(x):.2f}")).grid(row=algo_row, column=0, columnspan=2, sticky=EW, padx=5, pady=3)
        
        algo_row += 1
        Button(algo_frame, text="Run Algorithm", command=self.run_algorithm, bg='#e74c3c', fg='white', font=('Arial', 10, 'bold')).grid(row=algo_row, column=0, columnspan=2, pady=10, sticky=EW)
        
        algo_frame.columnconfigure(1, weight=1)
            
        # === View Controls ===
        view_frame = LabelFrame(parent, text="View Controls", bg='#2c3e50', fg='white', font=('Arial', 10, 'bold'))
        view_frame.pack(fill=X, padx=10, pady=5)
        
        Button(view_frame, text="Reset View", command=self.reset_view, bg='#95a5a6', fg='white').pack(fill=X, padx=10, pady=2)
        Button(view_frame, text="Randomize Layout", command=self.randomize_layout, bg='#95a5a6', fg='white').pack(fill=X, padx=10, pady=2)
        Button(view_frame, text="Show Coarsening Stages", command=self.show_stages, bg='#9b59b6', fg='white').pack(fill=X, padx=10, pady=2)
        
        # === File Operations ===
        file_frame = LabelFrame(parent, text="File Operations", bg='#2c3e50', fg='white', font=('Arial', 10, 'bold'))
        file_frame.pack(fill=X, padx=10, pady=5)
        
        Button(file_frame, text="Load Graph from File", command=self.load_graph, bg='#2ecc71', fg='white').pack(fill=X, padx=10, pady=2)
        Button(file_frame, text="Save Graph to File", command=self.save_graph, bg='#2ecc71', fg='white').pack(fill=X, padx=10, pady=2)
    
    def _create_metrics(self, parent):
        metrics_frame = LabelFrame(parent, text="Metrics", bg='#2c3e50', fg='white', font=('Arial', 10, 'bold'))
        metrics_frame.pack(fill=BOTH, expand=True, padx=10, pady=5)
        
        self.metrics_text = Text(metrics_frame, bg='#34495e', fg='white', font=('Courier', 10), wrap=WORD)
        self.metrics_text.pack(fill=BOTH, expand=True, padx=5, pady=5)
    
    def _setup_logging(self):
        sys.stdout = LogRedirector(self.log_text)
        print("=" * 60)
        print("GRAPH PARTITIONING VISUALIZER")
        print("=" * 60)
        print("Ready. Configure and generate a graph to start.\n")
    
    def generate_graph(self):
        n = self.vertices_var.get()
        graph_type = self.graph_type_var.get()
        clusters = self.clusters_var.get()
        intra_prob = self.intra_prob_var.get()
        inter_prob = self.inter_prob_var.get()
        vw_min = self.vw_min_var.get()
        vw_max = self.vw_max_var.get()
        ew_min = self.ew_min_var.get()
        ew_max = self.ew_max_var.get()
        
        print(f"\n--- Generating {graph_type} graph ---")
        print(f"  Vertices: {n}")
        print(f"  Clusters: {clusters}")
        print(f"  Intra probability: {intra_prob}")
        print(f"  Inter probability: {inter_prob}")
        print(f"  Vertex weights: [{vw_min}, {vw_max}]")
        print(f"  Edge weights: [{ew_min}, {ew_max}]")
        
        try:
            if graph_type == "Cluster":
                vertices_per_cluster = n // clusters
                gen = FastClusterGenerator(
                    num_clusters=clusters,
                    vertices_per_cluster=vertices_per_cluster,
                    target_edges=min(n * 6, 500000),
                    intra_ratio=intra_prob,
                    weight_range=(ew_min, ew_max),
                    vertex_weight_range=(vw_min, vw_max),
                    seed=42
                )
                self.current_graph = gen.generate()
            elif graph_type == "Barabasi-Albert":
                from data.generators import BarabasiAlbertGenerator
                gen = BarabasiAlbertGenerator(
                    n=n,
                    m0=min(10, n // 2),
                    m=max(2, n // 100),
                    weight_range=(ew_min, ew_max),
                    vertex_weight_range=(vw_min, vw_max),
                    seed=42
                )
                self.current_graph = gen.generate()
            else:  # Erdos-Renyi
                from data.generators import ErdosRenyiGenerator
                p = intra_prob * 0.1  # Для разреженности
                gen = ErdosRenyiGenerator(n=n, p=p, seed=42)
                self.current_graph = gen.generate()
                # Добавляем веса
                for v in range(n):
                    w = random.randint(vw_min, vw_max)
                    self.current_graph.set_vertex_weight(v, w)
                for u, v, _ in self.current_graph.edges():
                    w = random.randint(ew_min, ew_max)
                    self.current_graph.add_edge(u, v, w)
            
            self.current_partition = None
            self.levels = None
            self.current_ml = None
            self.visualizer.set_graph(self.current_graph)
            
            print(f"✓ Graph generated: {self.current_graph.num_vertices} vertices, {self.current_graph.num_edges} edges")
            self._update_metrics()
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    def run_algorithm(self):
        if self.current_graph is None:
            print("Please generate a graph first")
            return
        
        algorithm = self.algo_var.get()
        max_passes = self.max_passes_var.get()
        balance = self.balance_var.get()
        
        print(f"\n=== Running {algorithm} ===")
        print(f"  Max passes: {max_passes}")
        print(f"  Balance ratio: {balance}")
        
        def run():
            try:
                if algorithm == "Kernighan-Lin":
                    part = KernighanLin(max_passes=max_passes, seed=42)
                    self.current_partition, metrics = part.partition(self.current_graph, balance)
                    self.levels = []
                    self.current_ml = None
                    
                elif algorithm == "Multilevel":
                    # Используем MultilevelPartitioner
                    from algorithms.multilevel_slow import MultilevelPartitioner
                    self.current_ml = MultilevelPartitioner(
                        min_coarse_vertices=20,
                        max_levels=10,
                        refinement_passes=2,
                        seed=42
                    )
                    self.current_partition, metrics = self.current_ml.partition(self.current_graph, balance)
                    # Сохраняем уровни
                    self.levels = self.current_ml.levels if hasattr(self.current_ml, 'levels') else []
                    print(f"  ✓ Saved {len(self.levels)} coarsening levels")
                    
                else:  # Compare
                    print("\n--- Running Kernighan-Lin ---")
                    kl = KernighanLin(max_passes=max_passes, seed=42)
                    p_kl, m_kl = kl.partition(self.current_graph, balance)
                    
                    print("\n--- Running Multilevel ---")
                    from algorithms.multilevel import MultilevelPartitioner
                    self.current_ml = MultilevelPartitioner(
                        min_coarse_vertices=20,
                        max_levels=10,
                        refinement_passes=2,
                        seed=42
                    )
                    p_ml, m_ml = self.current_ml.partition(self.current_graph, balance)
                    self.levels = self.current_ml.levels if hasattr(self.current_ml, 'levels') else []
                    print(f"  ✓ Saved {len(self.levels)} coarsening levels")
                    
                    # Выбираем лучшее
                    if m_ml.cut_weight < m_kl.cut_weight:
                        self.current_partition = p_ml
                        metrics = m_ml
                        print(f"\n✓ Multilevel is better! (cut: {m_ml.cut_weight} vs {m_kl.cut_weight})")
                    else:
                        self.current_partition = p_kl
                        metrics = m_kl
                        print(f"\n✓ KL is better! (cut: {m_kl.cut_weight} vs {m_ml.cut_weight})")
                
                self.root.after(0, self._on_algorithm_done, metrics)
            except Exception as e:
                self.root.after(0, lambda: print(f"✗ Error: {e}"))
                import traceback
                traceback.print_exc()
        
        threading.Thread(target=run, daemon=True).start()

    def _on_algorithm_done(self, metrics):
        self.visualizer.set_partition(self.current_partition)
        self._update_metrics(metrics)
        
        print(f"\n✓ Algorithm completed!")
        print(f"  Cut weight: {metrics.cut_weight}")
        print(f"  Time: {metrics.time_seconds:.4f}s")
        print(f"  Memory: {metrics.memory_mb:.2f}MB")
        
        # Проверяем наличие уровней
        if hasattr(self, 'levels') and self.levels:
            print(f"  Coarsening levels available: {len(self.levels)}")
            self.stages_btn.config(state=NORMAL)
        else:
            print(f"  No coarsening levels available")
            self.stages_btn.config(state=DISABLED)
        
        # Обновляем информацию в метриках
        self._update_metrics(metrics)

    def show_stages(self):
        """Показать окно с этапами стягивания"""
        if hasattr(self, 'levels') and self.levels and len(self.levels) > 0:
            print(f"\n=== Showing {len(self.levels)} coarsening stages ===")
            self._show_stages_window()
        else:
            print("\n⚠️ No coarsening stages available.")
            print("   Reasons:")
            print("   1. You haven't run Multilevel algorithm yet")
            print("   2. Graph is too small (need 50+ vertices for effective coarsening)")
            print("   3. Graph is too dense (coarsening didn't reduce size)")
            messagebox.showinfo("Info", 
                "No coarsening stages available.\n\n"
                "Possible reasons:\n"
                "• You haven't run Multilevel algorithm\n"
                "• Graph is too small (need 50+ vertices)\n"
                "• Graph is too dense\n\n"
                "Try generating a larger graph (100+ vertices) and running Multilevel algorithm.")
        
    def _update_metrics(self, metrics=None):
        self.metrics_text.delete(1.0, END)
        
        if self.current_graph is None:
            self.metrics_text.insert(END, "No graph loaded")
            return
        
        self.metrics_text.insert(END, f"Graph Statistics:\n")
        self.metrics_text.insert(END, f"  Vertices: {self.current_graph.num_vertices}\n")
        self.metrics_text.insert(END, f"  Edges: {self.current_graph.num_edges}\n\n")
        
        if self.current_partition:
            cut = self.current_partition.cut_weight(self.current_graph)
            balance = self.current_partition.balance_quality()
            weight_balance = self.current_partition.weight_balance_quality()
            
            self.metrics_text.insert(END, f"Partition Metrics:\n")
            self.metrics_text.insert(END, f"  Cut weight: {cut}\n")
            self.metrics_text.insert(END, f"  Balance (count): {balance:.4f}\n")
            self.metrics_text.insert(END, f"  Balance (weight): {weight_balance:.4f}\n")
            self.metrics_text.insert(END, f"  Part 0: {self.current_partition.size0} vertices (weight={self.current_partition.weight0})\n")
            self.metrics_text.insert(END, f"  Part 1: {self.current_partition.size1} vertices (weight={self.current_partition.weight1})\n\n")
            
            if metrics:
                self.metrics_text.insert(END, f"Performance:\n")
                self.metrics_text.insert(END, f"  Time: {metrics.time_seconds:.4f}s\n")
                self.metrics_text.insert(END, f"  Memory: {metrics.memory_mb:.2f}MB\n")
        else:
            self.metrics_text.insert(END, "Not partitioned yet")
    
    def _show_stages_window(self):
        """Окно с наглядной визуализацией этапов стягивания"""
        if not self.levels:
            messagebox.showinfo("Info", "No coarsening stages available. Run Multilevel algorithm on a graph with 50+ vertices first.")
            return
        
        window = Toplevel(self.root)
        window.title("Coarsening Stages Visualizer")
        window.geometry("1200x800")
        window.configure(bg='#2c3e50')
        
        # Верхняя панель с управлением
        control_frame = Frame(window, bg='#2c3e50')
        control_frame.pack(fill=X, padx=10, pady=10)
        
        Label(control_frame, text="Coarsening Level:", bg='#2c3e50', fg='white', font=('Arial', 12, 'bold')).pack(side=LEFT, padx=5)
        
        self.stage_index = IntVar(value=0)
        self.stage_spinbox = Spinbox(control_frame, from_=0, to=len(self.levels), textvariable=self.stage_index, 
                                    width=5, command=self._update_stage_display, bg='#34495e', fg='white')
        self.stage_spinbox.pack(side=LEFT, padx=5)
        
        Button(control_frame, text="◀ Previous", command=self._prev_stage, bg='#3498db', fg='white').pack(side=LEFT, padx=5)
        Button(control_frame, text="Next ▶", command=self._next_stage, bg='#3498db', fg='white').pack(side=LEFT, padx=5)
        Button(control_frame, text="Show Original", command=self._show_original_stage, bg='#2ecc71', fg='white').pack(side=LEFT, padx=20)
        
        # Информационная панель
        self.stage_info = Label(window, text="", bg='#2c3e50', fg='#3498db', font=('Arial', 10))
        self.stage_info.pack(pady=5)
        
        # Основной канвас для визуализации
        self.stage_canvas = Canvas(window, bg='#1e1e1e', highlightthickness=0)
        self.stage_canvas.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Легенда
        legend_frame = Frame(window, bg='#2c3e50')
        legend_frame.pack(fill=X, padx=10, pady=5)
        
        Label(legend_frame, text="Legend:", bg='#2c3e50', fg='white', font=('Arial', 10, 'bold')).pack(side=LEFT, padx=10)
        
        # Цвета для разных частей
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        for i, col in enumerate(colors):
            Label(legend_frame, text=f"Part {i}", bg=col, fg='white', padx=10, pady=2).pack(side=LEFT, padx=5)
        
        Label(legend_frame, text="→", bg='#2c3e50', fg='#e74c3c', font=('Arial', 12, 'bold')).pack(side=LEFT, padx=10)
        Label(legend_frame, text="Contracted edge", bg='#2c3e50', fg='#e74c3c').pack(side=LEFT)
        
        # Сохраняем уровни и текущий отображаемый граф
        self.display_levels = self.levels
        self.current_display_graph = None
        self.current_display_partition = None
        
        # Показываем первый уровень
        self.stage_index.set(0)
        self._update_stage_display()
        
        # Привязываем события изменения размера окна
        window.bind('<Configure>', lambda e: self._update_stage_display() if e.widget == window else None)

    def _update_stage_display(self):
        """Обновление отображения текущего уровня"""
        idx = self.stage_index.get()
        
        if idx == len(self.display_levels):
            self._show_original_stage()
            return
        
        if idx < 0 or idx >= len(self.display_levels):
            return
        
        level = self.display_levels[idx]
        graph = level.graph
        coarse_graph = level.coarse_graph
        reverse_map = level.reverse_map
        
        self.stage_info.config(text=f"Level {idx}: {graph.num_vertices} vertices, {graph.num_edges} edges | "
                                    f"Compression: {level.compression_ratio:.3f}")
        
        # Визуализируем граф с показом того, какие вершины будут стянуты
        self._draw_coarsening_stage(graph, coarse_graph, reverse_map)

    def _show_original_stage(self):
        """Показать исходный граф"""
        if not self.display_levels:
            return
        
        original_graph = self.display_levels[0].graph
        self.stage_info.config(text=f"Original Graph: {original_graph.num_vertices} vertices, {original_graph.num_edges} edges")
        self._draw_graph_simple(original_graph)

    def _prev_stage(self):
        """Предыдущий уровень"""
        idx = self.stage_index.get()
        if idx > 0:
            self.stage_index.set(idx - 1)
            self._update_stage_display()

    def _next_stage(self):
        """Следующий уровень"""
        idx = self.stage_index.get()
        if idx < len(self.display_levels):
            self.stage_index.set(idx + 1)
            self._update_stage_display()

    def _draw_graph_simple(self, graph: Graph):
        """
        Простая визуализация графа без группировки
        """
        self.stage_canvas.delete("all")
        w = self.stage_canvas.winfo_width()
        h = self.stage_canvas.winfo_height()
        if w <= 1:
            w = 1000
        if h <= 1:
            h = 600
        
        n = graph.num_vertices
        if n == 0:
            return
        
        center_x, center_y = w/2, h/2
        radius = min(w, h) * 0.4
        
        positions = []
        for i in range(n):
            angle = 2 * pi * i / n
            x = center_x + radius * cos(angle)
            y = center_y + radius * sin(angle)
            positions.append((x, y))
        
        # Рисуем рёбра
        for u, v, wgt in graph.edges():
            x1, y1 = positions[u]
            x2, y2 = positions[v]
            self.stage_canvas.create_line(x1, y1, x2, y2, fill='#555555', width=1)
            
            if wgt != 1:
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                self.stage_canvas.create_text(mx, my, text=str(wgt), fill='#888888', font=('Arial', 8))
        
        # Рисуем вершины
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        for i in range(n):
            x, y = positions[i]
            weight = graph.get_vertex_weight(i)
            color = colors[i % len(colors)]
            r = self.visualizer.node_radius * (0.7 + weight / 20)
            r = min(r, self.visualizer.node_radius * 1.2)
            
            self.stage_canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline='white', width=2)
            self.stage_canvas.create_text(x, y, text=str(i), fill='white', font=('Arial', 10, 'bold'))
            
            if weight != 1:
                self.stage_canvas.create_text(x, y + r + 5, text=f"w={weight}", fill='#aaaaaa', font=('Arial', 8))
    
    def load_graph(self):
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if filename:
            try:
                self.current_graph = Graph.load_from_file(filename)
                self.current_partition = None
                self.levels = None
                self.visualizer.set_graph(self.current_graph)
                print(f"Loaded graph from {filename}")
                print(f"  Vertices: {self.current_graph.num_vertices}, Edges: {self.current_graph.num_edges}")
                self._update_metrics()
            except Exception as e:
                print(f"Error loading graph: {e}")
    
    def save_graph(self):
        if self.current_graph is None:
            print("No graph to save")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if filename:
            self.current_graph.save_to_file(filename)
            print(f"Graph saved to {filename}")
    
    def reset_view(self):
        self.visualizer.reset_view()
    
    def randomize_layout(self):
        self.visualizer.randomize_layout()
    
    def toggle_weights(self):
        self.visualizer.toggle_weights()
    
    def run(self):
        self.root.mainloop()


class LogRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
    
    def write(self, text):
        self.text_widget.insert(END, text)
        self.text_widget.see(END)
    
    def flush(self):
        pass


if __name__ == "__main__":
    app = GraphPartitioningGUI()
    app.run()