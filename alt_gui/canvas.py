"""Визуализация графа с force-directed layout и возможностью перемещения вершин"""

import math
import random
from tkinter import Canvas

from core.graph import Graph
from core.partition import Partition


class GraphCanvas:
    """Канвас для отображения и взаимодействия с графом"""
    
    COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#95a5a6']
    BG_COLOR = '#1e1e1e'
    EDGE_COLOR = '#555555'
    CUT_EDGE_COLOR = '#e74c3c'
    
    def __init__(self, parent, width=900, height=700):
        self.canvas = Canvas(parent, width=width, height=height, bg=self.BG_COLOR, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        
        self.width = width
        self.height = height
        self.graph = None
        self.partition = None
        self.nodes = []  # [x, y, id, part, color]
        self.node_radius = 16
        self.selected = None
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Force-directed layout parameters
        self.repulsion = 5000
        self.attraction = 0.05
        self.damping = 0.9
        self.iterations = 100
        
        self._bind_events()
        self.randomize_layout()  # Начальный случайный layout
    
    def _bind_events(self):
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Control-Button-1>", self._on_pan_start)
        self.canvas.bind("<Control-B1-Motion>", self._on_pan)
    
    def set_graph(self, graph: Graph, partition: Partition = None):
        self.graph = graph
        self.partition = partition
        self._compute_force_layout()
        self.redraw()
    
    def _compute_force_layout(self):
        """Force-directed layout для красивого расположения вершин"""
        n = self.graph.num_vertices
        if n == 0:
            return
        
        # Если нет узлов, создаём случайные позиции
        if not self.nodes or len(self.nodes) != n:
            self.nodes = []
            for i in range(n):
                x = random.uniform(100, self.width - 100)
                y = random.uniform(100, self.height - 100)
                part = self.partition.get_part(i) if self.partition else -1
                color = self.COLORS[part % len(self.COLORS)] if part >= 0 else '#7f8c8d'
                self.nodes.append([x, y, i, part, color, 0.0, 0.0])  # [x, y, id, part, color, vx, vy]
        
        # Силовой алгоритм
        for _ in range(self.iterations):
            forces = [[0.0, 0.0] for _ in range(n)]
            
            # Отталкивание между всеми парами вершин
            for i in range(n):
                for j in range(i + 1, n):
                    dx = self.nodes[i][0] - self.nodes[j][0]
                    dy = self.nodes[i][1] - self.nodes[j][1]
                    dist = max(math.sqrt(dx*dx + dy*dy), 10)
                    force = self.repulsion / (dist * dist)
                    fx = force * dx / dist
                    fy = force * dy / dist
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy
            
            # Притяжение по рёбрам
            for u, v, w in self.graph.edges():
                dx = self.nodes[u][0] - self.nodes[v][0]
                dy = self.nodes[u][1] - self.nodes[v][1]
                dist = max(math.sqrt(dx*dx + dy*dy), 1)
                force = self.attraction * w * dist
                fx = force * dx / dist
                fy = force * dy / dist
                forces[u][0] -= fx
                forces[u][1] -= fy
                forces[v][0] += fx
                forces[v][1] += fy
            
            # Обновление позиций
            for i in range(n):
                self.nodes[i][5] = (self.nodes[i][5] + forces[i][0]) * self.damping
                self.nodes[i][6] = (self.nodes[i][6] + forces[i][1]) * self.damping
                
                self.nodes[i][0] += self.nodes[i][5]
                self.nodes[i][1] += self.nodes[i][6]
                
                # Границы
                margin = 50
                self.nodes[i][0] = max(margin, min(self.width - margin, self.nodes[i][0]))
                self.nodes[i][1] = max(margin, min(self.height - margin, self.nodes[i][1]))
    
    def randomize_layout(self):
        """Случайное расположение вершин"""
        if not self.graph:
            return
        
        n = self.graph.num_vertices
        self.nodes = []
        for i in range(n):
            x = random.uniform(100, self.width - 100)
            y = random.uniform(100, self.height - 100)
            part = self.partition.get_part(i) if self.partition else -1
            color = self.COLORS[part % len(self.COLORS)] if part >= 0 else '#7f8c8d'
            self.nodes.append([x, y, i, part, color, 0.0, 0.0])
        
        self.redraw()
    
    def apply_force_layout(self):
        """Применить force-directed layout"""
        if self.graph:
            self._compute_force_layout()
            self.redraw()
    
    def redraw(self):
        self.canvas.delete("all")
        if not self.graph:
            return
        
        self._draw_edges()
        self._draw_vertices()
    
    def _draw_edges(self):
        if (self.graph == None or self.graph.num_vertices > 100):
            return
        for u, v, w in self.graph.edges():
            if u >= len(self.nodes) or v >= len(self.nodes):
                continue
            
            x1, y1 = self._transform(self.nodes[u][0], self.nodes[u][1])
            x2, y2 = self._transform(self.nodes[v][0], self.nodes[v][1])
            
            if self.partition and self.partition.get_part(u) != self.partition.get_part(v):
                color, width = self.CUT_EDGE_COLOR, 2
            else:
                color, width = self.EDGE_COLOR, 1
            
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)
            
            if w != 1:
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                self.canvas.create_text(mx, my, text=str(w), fill='#888888', font=('Arial', 8))
    
    def _draw_vertices(self):
        if (self.graph == None or self.graph.num_vertices > 100):
            return
        for x, y, vid, part, color, vx, vy in self.nodes:
            sx, sy = self._transform(x, y)
            r = self.node_radius * (0.7 + self.graph.get_vertex_weight(vid) / 20)
            r = min(r, self.node_radius * 1.3)
            r = r * self.zoom
            
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill=color, outline='white', width=1)
            self.canvas.create_text(sx, sy, text=str(vid), fill='white', font=('Arial', int(10 * self.zoom), 'bold'))
            
            # Показываем вес вершины
            weight = self.graph.get_vertex_weight(vid)
            if weight != 1:
                self.canvas.create_text(sx, sy + r + 4, text=f"w={weight}", fill='#aaaaaa', font=('Arial', int(8 * self.zoom)))
    
    def _transform(self, x, y):
        return (x * self.zoom + self.offset_x, y * self.zoom + self.offset_y)
    
    def _on_click(self, event):
        best = None
        best_dist = self.node_radius * self.zoom + 5
        
        for i, (x, y, vid, part, color, vx, vy) in enumerate(self.nodes):
            sx, sy = self._transform(x, y)
            dist = ((sx - event.x) ** 2 + (sy - event.y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = i
        
        self.selected = best
        if best is not None:
            self.drag_start = (event.x, event.y)
    
    def _on_drag(self, event):
        if self.selected is not None:
            dx = (event.x - self.drag_start[0]) / self.zoom
            dy = (event.y - self.drag_start[1]) / self.zoom
            self.nodes[self.selected][0] += dx
            self.nodes[self.selected][1] += dy
            self.drag_start = (event.x, event.y)
            self.redraw()
    
    def _on_release(self, event):
        self.selected = None
    
    def _on_wheel(self, event):
        if event.delta > 0 or event.num == 4:
            self.zoom *= 1.1
        else:
            self.zoom *= 0.9
        self.zoom = max(0.2, min(3.0, self.zoom))
        self.redraw()
    
    def _on_pan_start(self, event):
        self.pan_start = (event.x, event.y)
    
    def _on_pan(self, event):
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
        self._compute_force_layout()
        self.redraw()
    
    def set_partition(self, partition: Partition):
        self.partition = partition
        for i, (x, y, vid, part, color, vx, vy) in enumerate(self.nodes):
            new_part = partition.get_part(vid)
            self.nodes[i][3] = new_part
            self.nodes[i][4] = self.COLORS[new_part % len(self.COLORS)] if new_part >= 0 else '#7f8c8d'
        self.redraw()