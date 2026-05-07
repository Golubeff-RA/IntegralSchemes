"""Панель метрик с полной информацией о разбиении"""

from tkinter import LabelFrame, Text, Frame, Label, LEFT, RIGHT, TOP, BOTTOM, X, Y, BOTH, EW
from tkinter import ttk


class MetricsPanel(LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Metrics", bg='#2c3e50', fg='white', font=('Arial', 10, 'bold'))
        self.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Создаём текстовое поле с прокруткой
        self.text = Text(self, bg='#34495e', fg='white', font=('Courier', 10), 
                         wrap='word', height=15, relief='flat')
        
        scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        
        self.text.pack(side=LEFT, fill=BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=RIGHT, fill=Y, padx=(0, 5), pady=5)
    
    def _insert_header(self, text):
        """Вставка заголовка"""
        self.text.insert('end', f"\n{'='*50}\n")
        self.text.insert('end', f"{text}\n", 'header')
        self.text.insert('end', f"{'='*50}\n")
        self.text.tag_config('header', foreground='#3498db', font=('Arial', 10, 'bold'))
    
    def _insert_subheader(self, text):
        """Вставка подзаголовка"""
        self.text.insert('end', f"\n{text}\n", 'subheader')
        self.text.tag_config('subheader', foreground='#2ecc71', font=('Arial', 9, 'bold'))
    
    def _insert_pair(self, label, value, color=None):
        """Вставка пары ключ-значение"""
        self.text.insert('end', f"{label}: ", 'label')
        self.text.insert('end', f"{value}\n", 'value')
        self.text.tag_config('label', foreground='#95a5a6')
        if color:
            self.text.tag_config('value', foreground=color)
        else:
            self.text.tag_config('value', foreground='#ecf0f1')
    
    def update(self, graph, partition=None, metrics=None, algorithm_stats=None):
        """Обновление всех метрик"""
        self.text.delete(1.0, 'end')
        
        # === Граф ===
        self._insert_header("GRAPH STATISTICS")
        self._insert_pair("Vertices", str(graph.num_vertices))
        self._insert_pair("Edges", str(graph.num_edges))
        
        # Плотность графа
        if graph.num_vertices > 1:
            density = 2 * graph.num_edges / (graph.num_vertices * (graph.num_vertices - 1))
            self._insert_pair("Density", f"{density:.6f}")
        
        # Статистика степеней
        degrees = [graph.get_degree(v) for v in range(graph.num_vertices)]
        if degrees:
            self._insert_pair("Max degree", str(max(degrees)))
            self._insert_pair("Min degree", str(min(degrees)))
            self._insert_pair("Avg degree", f"{sum(degrees)/len(degrees):.2f}")
        
        self.text.insert('end', "\n")
        
        # === Веса ===
        self._insert_header("WEIGHTS INFORMATION")
        
        v_weights = [graph.get_vertex_weight(v) for v in range(graph.num_vertices)]
        if v_weights:
            self._insert_pair("Vertex weights", f"[{min(v_weights)} ... {max(v_weights)}]")
            self._insert_pair("Total vertex weight", str(sum(v_weights)))
        
        e_weights = [w for _, _, w in graph.edges()]
        if e_weights:
            self._insert_pair("Edge weights", f"[{min(e_weights)} ... {max(e_weights)}]")
            self._insert_pair("Total edge weight", str(sum(e_weights)))
        
        self.text.insert('end', "\n")
        
        # === Разбиение ===
        if partition:
            self._insert_header("PARTITION METRICS")
            
            cut = partition.cut_weight(graph)
            cut_edges = len(partition.cut_edges(graph))
            self._insert_pair("Cut weight", str(cut), "#e74c3c")
            self._insert_pair("Cut edges count", str(cut_edges), "#e74c3c")
            
            if graph.num_edges > 0 and e_weights:
                cut_ratio = cut / sum(e_weights) * 100
                self._insert_pair("Cut ratio", f"{cut_ratio:.2f}%", "#e74c3c")
            
            self.text.insert('end', "\n")
            
            balance_count = partition.balance_quality()
            balance_weight = partition.weight_balance_quality()
            
            self._insert_pair("Balance (count)", f"{balance_count:.4f}")
            self._insert_pair("Balance (weight)", f"{balance_weight:.4f}")
            
            self.text.insert('end', "\n")
            self._insert_subheader("Part Sizes")
            self._insert_pair("  Part 0", f"{partition.size0} vertices", "#3498db")
            self._insert_pair("  Part 1", f"{partition.size1} vertices", "#e74c3c")
            
            self._insert_subheader("Part Weights")
            self._insert_pair("  Part 0", f"{partition.weight0}", "#3498db")
            self._insert_pair("  Part 1", f"{partition.weight1}", "#e74c3c")
            
            self.text.insert('end', "\n")
            
            # Детали разреза
            cut_edges_list = partition.cut_edges(graph)
            if cut_edges_list:
                self._insert_subheader("Cut Edges Details")
                for i, (u, v, w) in enumerate(cut_edges_list[:10]):
                    self._insert_pair(f"  Edge {u}-{v}", f"weight={w}", "#e74c3c")
                if len(cut_edges_list) > 10:
                    self.text.insert('end', f"  ... and {len(cut_edges_list) - 10} more\n")
        
        self.text.insert('end', "\n")
        
        # === Производительность ===
        if metrics:
            self._insert_header("PERFORMANCE")
            self._insert_pair("Execution time", f"{metrics.time_seconds:.4f} seconds")
            self._insert_pair("Memory usage", f"{metrics.memory_mb:.2f} MB")
        
        # === Многоуровневые метрики ===
        if algorithm_stats:
            self._insert_header("MULTILEVEL METRICS")
            
            if 'coarsening_levels' in algorithm_stats:
                self._insert_pair("Coarsening levels", str(algorithm_stats['coarsening_levels']))
            
            if 'compression_ratio' in algorithm_stats:
                self._insert_pair("Final compression", f"{algorithm_stats['compression_ratio']:.4f}")
            
            if 'coarsening_time' in algorithm_stats:
                self._insert_pair("Coarsening time", f"{algorithm_stats['coarsening_time']:.4f}s")
            
            if 'initial_partition_time' in algorithm_stats:
                self._insert_pair("Initial partition time", f"{algorithm_stats['initial_partition_time']:.4f}s")
            
            if 'uncoarsening_time' in algorithm_stats:
                self._insert_pair("Uncoarsening time", f"{algorithm_stats['uncoarsening_time']:.4f}s")
        
        # Прокручиваем в начало
        self.text.see(1.0)
    
    def clear(self):
        """Очистка панели"""
        self.text.delete(1.0, 'end')