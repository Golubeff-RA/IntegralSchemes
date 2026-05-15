"""Главное окно приложения"""

import threading
import random
from tkinter import Tk, Frame, LEFT, RIGHT, Y, BOTH

from core.graph import Graph
from algorithms.kernighan_lin import KernighanLin
from algorithms.multilevel_slow import MultilevelPartitioner   # <-- изменён импорт
from data.generators import (
    ClusterGraphGenerator, 
    FastClusterGenerator,
    BarabasiAlbertGenerator,
)

from .canvas import GraphCanvas
from .controls import ControlsPanel
from .metrics_panel import MetricsPanel
from .log_panel import LogPanel
from .stages_window import StagesWindow


class GraphPartitioningGUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Graph Partitioning Visualizer")
        self.root.geometry("1450x950")
        self.root.configure(bg='#2c3e50')
        
        self.graph = None
        self.partition = None
        self.levels = []
        self.last_metrics = None
        self.last_algorithm_stats = None
        
        self._create_layout()
    
    def _create_layout(self):
        # Left panel
        left = Frame(self.root, bg='#2c3e50', width=400)
        left.pack(side=LEFT, fill=Y, padx=5, pady=5)
        
        # Right panel
        right = Frame(self.root, bg='#2c3e50')
        right.pack(side=RIGHT, fill=BOTH, expand=True, padx=5, pady=5)
        
        # Canvas
        self.canvas = GraphCanvas(right, width=950, height=750)
        
        # Controls
        self.controls = ControlsPanel(
            left, 
            self._on_generate, 
            self._on_run,
            self._randomize_layout,
            self._apply_force_layout
        )
        self.controls.pack(fill='x')
        
        # Metrics
        self.metrics = MetricsPanel(left)
        
        # Log
        self.log = LogPanel(left)
        
        # Connect stages button
        self.controls.stages_btn.config(command=self._show_stages)
    
    def _randomize_layout(self):
        self.canvas.randomize_layout()
    
    def _apply_force_layout(self):
        self.canvas.apply_force_layout()
    
    def _on_generate(self, params):
        gen_type = params['type']
        n = params['vertices']
        
        print(f"\n--- Generating {gen_type} graph: {n} vertices ---")
        
        try:
            if gen_type == "Cluster":
                gen = ClusterGraphGenerator(
                    num_clusters=params['clusters'],
                    cluster_size=n // params['clusters'],
                    target_edges=min(n*7, 500000),
                    intra_ratio=params['intra_prob'],
                    weight_range=(params['ew_min'], params['ew_max']),
                    vertex_weight_range=(params['vw_min'], params['vw_max']),
                    seed=42
                )
            
            elif gen_type == "FastCluster":
                gen = FastClusterGenerator(
                    num_clusters=params['clusters'],
                    vertices_per_cluster=n // params['clusters'],
                    target_edges=params['target_edges'],
                    intra_ratio=params['intra_ratio'],
                    weight_range=(params['ew_min'], params['ew_max']),
                    vertex_weight_range=(params['vw_min'], params['vw_max']),
                    seed=42
                )
            
            elif gen_type == "Barabasi-Albert":
                gen = BarabasiAlbertGenerator(
                    n=n,
                    m0=params['m0'],
                    m=params['m'],
                    weight_range=(params['ew_min'], params['ew_max']),
                    vertex_weight_range=(params['vw_min'], params['vw_max']),
                    seed=42
                )
            
            self.graph = gen.generate()
            self.partition = None
            self.levels = []
            self.last_metrics = None
            self.last_algorithm_stats = None
            self.canvas.set_graph(self.graph)
            self.controls.set_stages_enabled(False)
            self.metrics.update(self.graph)
            
            print(f"✓ Generated: {self.graph.num_vertices} vertices, {self.graph.num_edges} edges")
            
        except Exception as e:
            print(f"Error generating graph: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_run(self, params):
        if self.graph is None:
            print("Please generate a graph first")
            return
        
        algo = params['algorithm']
        max_passes = params['max_passes']
        coarsen_to = params['min_coarse']     # параметр min_coarse теперь coarsen_to
        balance = params['balance']
        
        print(f"\n=== Running {algo} algorithm ===")
        print(f"  Balance: {balance}")
        
        def task():
            try:
                if algo == "KL":
                    part = KernighanLin(max_passes=max_passes, seed=42)
                    p, m = part.partition(self.graph, balance)
                    self.levels = []
                    self.last_algorithm_stats = None
                else:
                    # Новый MultilevelPartitioner
                    ml = MultilevelPartitioner(
                        coarsen_to=coarsen_to,
                        max_passes=5,       # можно тоже вынести в параметры
                        seed=42
                    )
                    p, m = ml.partition(self.graph, balance)
                    
                    # Получаем историю стягивания
                    self.levels = ml.get_coarsening_history()
                    self.last_algorithm_stats = {
                        'coarsening_levels': len(self.levels),
                        'compression_ratio': self.levels[-1].compression_ratio if self.levels else 1.0
                    }
                    print(f"  Coarsening levels: {len(self.levels)}")
                
                self.partition = p
                self.last_metrics = m
                self.root.after(0, self._on_complete)
            except Exception as e:
                self.root.after(0, lambda: print(f"Error: {e}"))
                import traceback
                traceback.print_exc()
        
        threading.Thread(target=task, daemon=True).start()
    
    def _on_complete(self):
        self.canvas.set_partition(self.partition)
        self.metrics.update(self.graph, self.partition, self.last_metrics, self.last_algorithm_stats)
        self.controls.set_stages_enabled(len(self.levels) > 0)
        
        print(f"\n✓ Completed!")
        print(f"  Cut weight: {self.last_metrics.cut_weight}")
        print(f"  Time: {self.last_metrics.time_seconds:.4f}s")
        print(f"  Memory: {self.last_metrics.memory_mb:.2f}MB")
    
    def _show_stages(self):
        if self.levels:
            StagesWindow(self.root, self.levels)
        else:
            print("No coarsening stages available. Run Multilevel on a graph with 50+ vertices.")
    
    def run(self):
        self.root.mainloop()