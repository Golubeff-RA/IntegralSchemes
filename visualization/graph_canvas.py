"""
Визуализация графа на канвасе с force-directed layout
"""

import tkinter as tk
import math
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition


@dataclass
class GraphNode:
    """Визуальное представление вершины графа"""
    id: int
    x: float
    y: float
    vx: float = 0  # velocity for force-directed layout
    vy: float = 0
    part: int = -1
    radius: int = 15
    color: str = "#3498db"
    selected: bool = False


class GraphCanvas(tk.Canvas):
    """
    Канвас для отображения графа с force-directed layout
    """
    
    PART_COLORS = [
        "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#95a5a6", "#34495e", "#16a085",
    ]
    DEFAULT_COLOR = "#7f8c8d"
    SELECTED_COLOR = "#f1c40f"
    EDGE_COLOR = "#bdc3c7"
    CUT_EDGE_COLOR = "#e74c3c"
    
    def __init__(self, parent, width: int = 800, height: int = 600, bg: str = "#2c3e50"):
        super().__init__(parent, width=width, height=height, bg=bg, highlightthickness=0)
        
        self.width = width
        self.height = height
        
        self.graph: Optional[Graph] = None
        self.partition: Optional[Partition] = None
        self.nodes: List[GraphNode] = []
        
        # Настройки отображения
        self.show_vertex_labels = True
        self.show_edge_labels = False
        self.highlight_cut_edges = True
        self.animation_running = False
        
        # Интерактивность
        self.selected_node: Optional[GraphNode] = None
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # Масштабирование
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # Force-directed layout parameters
        self.repulsion = 5000  # Сила отталкивания
        self.attraction = 0.01  # Сила притяжения
        self.damping = 0.9  # Затухание
        self.max_iterations = 100
        
        self._bind_events()
    
    def _bind_events(self):
        """Привязка событий"""
        self.bind("<Button-1>", self._on_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Button-3>", self._on_right_click)
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Control-Button-1>", self._on_pan_start)
        self.bind("<Control-B1-Motion>", self._on_pan)
    
    def set_graph(self, graph: Graph, partition: Optional[Partition] = None):
        """Установка графа"""
        self.graph = graph
        self.partition = partition
        self._compute_force_directed_layout()
        self.redraw()
    
    def _compute_force_directed_layout(self):
        """Force-directed layout алгоритм"""
        if not self.graph or self.graph.num_vertices == 0:
            return
        
        n = self.graph.num_vertices
        
        # Инициализация случайными позициями
        self.nodes = []
        for i in range(n):
            part = -1
            if self.partition:
                part = self.partition.get_part(i)
            color = self.PART_COLORS[part % len(self.PART_COLORS)] if part >= 0 else self.DEFAULT_COLOR
            
            self.nodes.append(GraphNode(
                id=i,
                x=random.uniform(self.width * 0.2, self.width * 0.8),
                y=random.uniform(self.height * 0.2, self.height * 0.8),
                part=part,
                color=color
            ))
        
        # Силовой алгоритм
        for iteration in range(self.max_iterations):
            forces = [(0.0, 0.0) for _ in range(n)]
            
            # Силы отталкивания между всеми парами вершин
            for i in range(n):
                for j in range(i + 1, n):
                    dx = self.nodes[i].x - self.nodes[j].x
                    dy = self.nodes[i].y - self.nodes[j].y
                    dist = math.sqrt(dx*dx + dy*dy) + 0.1
                    
                    force = self.repulsion / (dist * dist)
                    fx = force * dx / dist
                    fy = force * dy / dist
                    
                    forces[i] = (forces[i][0] + fx, forces[i][1] + fy)
                    forces[j] = (forces[j][0] - fx, forces[j][1] - fy)
            
            # Силы притяжения по рёбрам
            for u, v, w in self.graph.edges():
                dx = self.nodes[u].x - self.nodes[v].x
                dy = self.nodes[u].y - self.nodes[v].y
                dist = math.sqrt(dx*dx + dy*dy) + 0.1
                
                force = self.attraction * w * dist
                fx = force * dx / dist
                fy = force * dy / dist
                
                forces[u] = (forces[u][0] - fx, forces[u][1] - fy)
                forces[v] = (forces[v][0] + fx, forces[v][1] + fy)
            
            # Обновление позиций
            for i in range(n):
                self.nodes[i].vx = (self.nodes[i].vx + forces[i][0]) * self.damping
                self.nodes[i].vy = (self.nodes[i].vy + forces[i][1]) * self.damping
                
                self.nodes[i].x += self.nodes[i].vx
                self.nodes[i].y += self.nodes[i].vy
                
                # Ограничение в пределах канваса с отступами
                margin = 50
                self.nodes[i].x = max(margin, min(self.width - margin, self.nodes[i].x))
                self.nodes[i].y = max(margin, min(self.height - margin, self.nodes[i].y))
        
        # Центрирование
        self._center_layout()
    
    def _center_layout(self):
        """Центрирование layout"""
        if not self.nodes:
            return
        
        min_x = min(n.x for n in self.nodes)
        max_x = max(n.x for n in self.nodes)
        min_y = min(n.y for n in self.nodes)
        max_y = max(n.y for n in self.nodes)
        
        width = max_x - min_x
        height = max_y - min_y
        
        scale_x = (self.width - 100) / width if width > 0 else 1
        scale_y = (self.height - 100) / height if height > 0 else 1
        scale = min(scale_x, scale_y)
        
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        for node in self.nodes:
            node.x = (node.x - center_x) * scale + self.width / 2
            node.y = (node.y - center_y) * scale + self.height / 2
    
    def redraw(self):
        """Перерисовка"""
        self.delete("all")
        
        if not self.graph:
            self.create_text(self.width/2, self.height/2, 
                           text="No graph loaded", fill="white", font=("Arial", 16))
            return
        
        self._draw_edges()
        
        for node in self.nodes:
            self._draw_node(node)
        
        self._draw_legend()
    
    def _draw_edges(self):
        """Рисование рёбер"""
        for u, v, w in self.graph.edges():
            if u < len(self.nodes) and v < len(self.nodes):
                x1, y1 = self._transform_coords(self.nodes[u].x, self.nodes[u].y)
                x2, y2 = self._transform_coords(self.nodes[v].x, self.nodes[v].y)
                
                is_cut = False
                if self.partition and self.highlight_cut_edges:
                    if self.partition.get_part(u) != self.partition.get_part(v):
                        is_cut = True
                
                color = self.CUT_EDGE_COLOR if is_cut else self.EDGE_COLOR
                width = 2 if is_cut else 1
                
                self.create_line(x1, y1, x2, y2, fill=color, width=width, tags="edge")
                
                if self.show_edge_labels and w != 1:
                    mx = (x1 + x2) / 2
                    my = (y1 + y2) / 2
                    self.create_text(mx, my, text=str(w), fill="#ecf0f1", 
                                   font=("Arial", 8), tags="edge_label")
    
    def _draw_node(self, node: GraphNode):
        """Рисование вершины"""
        x, y = self._transform_coords(node.x, node.y)
        radius = node.radius * self.zoom
        
        if node.selected:
            color = self.SELECTED_COLOR
        else:
            color = node.color
        
        # Тень
        self.create_oval(x - radius - 2, y - radius - 2,
                        x + radius + 2, y + radius + 2,
                        fill="#000000", outline="", tags="node_shadow")
        
        # Основной круг
        self.create_oval(x - radius, y - radius,
                        x + radius, y + radius,
                        fill=color, outline="#ecf0f1", width=2, tags="node")
        
        if self.show_vertex_labels:
            self.create_text(x, y, text=str(node.id), fill="white",
                           font=("Arial", int(10 * self.zoom), "bold"), tags="node_label")
    
    def _draw_legend(self):
        """Легенда"""
        legend_x = 10
        legend_y = 10
        
        self.create_rectangle(legend_x, legend_y, legend_x + 180, legend_y + 150,
                             fill="#34495e", outline="#ecf0f1", tags="legend")
        self.create_text(legend_x + 90, legend_y + 15, text="Legend", 
                        fill="white", font=("Arial", 10, "bold"), tags="legend")
        
        y_offset = 35
        for i in range(min(6, len(self.PART_COLORS))):
            color = self.PART_COLORS[i]
            self.create_rectangle(legend_x + 10, legend_y + y_offset,
                                 legend_x + 25, legend_y + y_offset + 12,
                                 fill=color, outline="white", tags="legend")
            self.create_text(legend_x + 35, legend_y + y_offset + 6, 
                           text=f"Part {i}", fill="white", 
                           anchor="w", font=("Arial", 9), tags="legend")
            y_offset += 18
        
        self.create_line(legend_x + 10, legend_y + y_offset + 5,
                        legend_x + 25, legend_y + y_offset + 5,
                        fill=self.CUT_EDGE_COLOR, width=2, tags="legend")
        self.create_text(legend_x + 35, legend_y + y_offset + 6,
                        text="Cut edges", fill="white",
                        anchor="w", font=("Arial", 9), tags="legend")
    
    def _transform_coords(self, x: float, y: float) -> Tuple[float, float]:
        """Преобразование координат"""
        return (x * self.zoom + self.offset_x, y * self.zoom + self.offset_y)
    
    def _inverse_transform(self, x: float, y: float) -> Tuple[float, float]:
        return ((x - self.offset_x) / self.zoom, (y - self.offset_y) / self.zoom)
    
    def _find_node_at(self, x: float, y: float) -> Optional[GraphNode]:
        """Поиск вершины под курсором"""
        canvas_x, canvas_y = self._inverse_transform(x, y)
        
        for node in self.nodes:
            dx = node.x - canvas_x
            dy = node.y - canvas_y
            distance = math.sqrt(dx*dx + dy*dy)
            if distance < node.radius * self.zoom:
                return node
        return None
    
    def _on_click(self, event):
        """Обработка клика"""
        node = self._find_node_at(event.x, event.y)
        
        if node:
            if self.selected_node:
                self.selected_node.selected = False
            self.selected_node = node
            node.selected = True
            self.redraw()
            
            if hasattr(self, 'on_node_selected'):
                self.on_node_selected(node.id)
        else:
            if self.selected_node:
                self.selected_node.selected = False
                self.selected_node = None
                self.redraw()
            
            self.dragging = True
            self.drag_start_x = event.x
            self.drag_start_y = event.y
    
    def _on_drag(self, event):
        """Перетаскивание"""
        if self.dragging and self.selected_node:
            dx = event.x - self.drag_start_x
            dy = event.y - self.drag_start_y
            self.selected_node.x += dx / self.zoom
            self.selected_node.y += dy / self.zoom
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.redraw()
    
    def _on_release(self, event):
        self.dragging = False
    
    def _on_right_click(self, event):
        node = self._find_node_at(event.x, event.y)
        if node and hasattr(self, 'on_node_right_click'):
            self.on_node_right_click(node.id, event.x_root, event.y_root)
    
    def _on_mousewheel(self, event):
        if event.delta > 0 or (hasattr(event, 'num') and event.num == 4):
            self.zoom *= 1.1
        else:
            self.zoom *= 0.9
        self.zoom = max(0.2, min(3.0, self.zoom))
        self.redraw()
    
    def _on_pan_start(self, event):
        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
    
    def _on_pan(self, event):
        if hasattr(self, 'panning') and self.panning:
            self.offset_x += event.x - self.pan_start_x
            self.offset_y += event.y - self.pan_start_y
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.redraw()
    
    def reset_view(self):
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._center_layout()
        self.redraw()
    
    def set_partition(self, partition: Partition):
        """Обновление разбиения"""
        self.partition = partition
        for node in self.nodes:
            part = partition.get_part(node.id)
            node.part = part
            node.color = self.PART_COLORS[part % len(self.PART_COLORS)] if part >= 0 else self.DEFAULT_COLOR
        self.redraw()
    
    def randomize_layout(self):
        """Случайное расположение вершин"""
        for node in self.nodes:
            node.x = random.uniform(50, self.width - 50)
            node.y = random.uniform(50, self.height - 50)
        self.redraw()