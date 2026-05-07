"""Окно визуализации этапов стягивания с сохранением позиций"""

import math
import random
from tkinter import Toplevel, Frame, Label, Button, Canvas
from tkinter import LEFT, RIGHT, X, BOTH, TOP, BOTTOM, EW


class StagesWindow:
    def __init__(self, parent, levels):
        self.window = Toplevel(parent)
        self.window.title("Coarsening Stages")
        self.window.geometry("1100x800")
        self.window.configure(bg="#2c3e50")

        self.levels = levels
        self.current = 0
        self.node_positions = {}  # Сохраняем позиции для каждого уровня
        self.repulsion = 5000
        self.attraction = 0.05
        self.damping = 0.9
        self.iterations = 50

        self._create_widgets()
        self._show_current()

    def _create_widgets(self):
        # Controls
        frame = Frame(self.window, bg="#2c3e50")
        frame.pack(fill=X, padx=10, pady=10)

        Button(frame, text="◀ Previous", command=self._prev, bg="#3498db", fg="white").pack(side=LEFT, padx=5)
        Button(frame, text="Next ▶", command=self._next, bg="#3498db", fg="white").pack(side=LEFT, padx=5)
        Button(frame, text="Original", command=self._show_original, bg="#2ecc71", fg="white").pack(side=LEFT, padx=20)
        Button(frame, text="Randomize Layout", command=self._randomize_layout, bg="#f39c12", fg="white").pack(
            side=LEFT, padx=5
        )
        Button(frame, text="Apply Force Layout", command=self._apply_force_layout, bg="#9b59b6", fg="white").pack(
            side=LEFT, padx=5
        )

        self.label = Label(frame, text="", bg="#2c3e50", fg="#3498db", font=("Arial", 10))
        self.label.pack(side=LEFT, padx=20)

        # Canvas
        self.canvas = Canvas(self.window, bg="#1e1e1e")
        self.canvas.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Bind resize event
        self.canvas.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        self._draw_current()

    def _compute_force_layout(self, graph, positions):
        """Force-directed layout для заданного графа"""
        n = graph.num_vertices
        if n == 0:
            return positions

        # Инициализация скоростей
        velocities = [[0.0, 0.0] for _ in range(n)]

        for _ in range(self.iterations):
            forces = [[0.0, 0.0] for _ in range(n)]

            # Отталкивание
            for i in range(n):
                for j in range(i + 1, n):
                    dx = positions[i][0] - positions[j][0]
                    dy = positions[i][1] - positions[j][1]
                    dist = max(math.sqrt(dx * dx + dy * dy), 10)
                    force = self.repulsion / (dist * dist)
                    fx = force * dx / dist
                    fy = force * dy / dist
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy

            # Притяжение по рёбрам
            for u, v, w in graph.edges():
                dx = positions[u][0] - positions[v][0]
                dy = positions[u][1] - positions[v][1]
                dist = max(math.sqrt(dx * dx + dy * dy), 1)
                force = self.attraction * w * dist
                fx = force * dx / dist
                fy = force * dy / dist
                forces[u][0] -= fx
                forces[u][1] -= fy
                forces[v][0] += fx
                forces[v][1] += fy

            # Обновление позиций
            for i in range(n):
                velocities[i][0] = (velocities[i][0] + forces[i][0]) * self.damping
                velocities[i][1] = (velocities[i][1] + forces[i][1]) * self.damping

                positions[i][0] += velocities[i][0]
                positions[i][1] += velocities[i][1]

                # Границы
                w = self.canvas.winfo_width()
                h = self.canvas.winfo_height()
                if w <= 1:
                    w, h = 1000, 700
                margin = 50
                positions[i][0] = max(margin, min(w - margin, positions[i][0]))
                positions[i][1] = max(margin, min(h - margin, positions[i][1]))

        return positions

    def _randomize_positions(self, graph):
        """Случайные позиции для вершин графа"""
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1:
            w, h = 1000, 700

        positions = []
        for _ in range(graph.num_vertices):
            x = random.uniform(80, w - 80)
            y = random.uniform(80, h - 80)
            positions.append([x, y])

        return positions

    def _get_positions(self, graph):
        """Получить или создать позиции для графа"""
        key = id(graph)
        if key not in self.node_positions or len(self.node_positions[key]) != graph.num_vertices:
            self.node_positions[key] = self._randomize_positions(graph)
        return self.node_positions[key]

    def _draw_graph(self, graph, is_original=False):
        """Отрисовка графа с текущими позициями"""
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1:
            w, h = 1000, 700

        positions = self._get_positions(graph)

        # Убеждаемся, что позиции в пределах канваса
        for i, (x, y) in enumerate(positions):
            if i < len(positions):
                positions[i][0] = max(50, min(w - 50, x))
                positions[i][1] = max(50, min(h - 50, y))

        # Рисуем рёбра
        for u, v, wgt in graph.edges():
            if u < len(positions) and v < len(positions):
                x1, y1 = positions[u]
                x2, y2 = positions[v]

                # Определяем цвет ребра (красное если будет стянуто)
                color = "#e74c3c" if not is_original else "#555555"
                width = 2 if not is_original else 1

                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

                if wgt != 1:
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    self.canvas.create_text(mx, my, text=str(wgt), fill="#888888", font=("Arial", 8))

        # Рисуем вершины
        for i in range(graph.num_vertices):
            x, y = positions[i]
            weight = graph.get_vertex_weight(i)
            r = 12 * (0.7 + weight / 15)
            r = min(r, 18)

            color = "#3498db" if is_original else "#e74c3c"
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="white", width=2)
            self.canvas.create_text(x, y, text=str(i), fill="white", font=("Arial", 10, "bold"))

            if weight != 1:
                self.canvas.create_text(x, y + r + 4, text=f"w={weight}", fill="#aaaaaa", font=("Arial", 7))

        # Легенда
        if not is_original:
            self.canvas.create_text(
                10, 20, anchor="nw", text="Red edges = will be contracted", fill="#e74c3c", font=("Arial", 9)
            )

    def _draw_with_clusters(self, graph, coarse_graph, reverse_map):
        """Отрисовка с показом кластеров (какие вершины стягиваются)"""
        self.canvas.delete("all")
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1:
            w, h = 1000, 700

        positions = self._get_positions(graph)

        # Определяем кластеры
        clusters = {}
        if isinstance(reverse_map, dict):
            for cv, vertices in reverse_map.items():
                for v in vertices:
                    clusters[v] = cv
        elif isinstance(reverse_map, list):
            for cv, vertices in enumerate(reverse_map):
                for v in vertices:
                    clusters[v] = cv

        # Группируем вершины по кластерам
        cluster_groups = {}
        for v, cv in clusters.items():
            if cv not in cluster_groups:
                cluster_groups[cv] = []
            cluster_groups[cv].append(v)

        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

        # Рисуем рёбра
        for u, v, wgt in graph.edges():
            if u < len(positions) and v < len(positions):
                x1, y1 = positions[u]
                x2, y2 = positions[v]

                if clusters.get(u, -1) == clusters.get(v, -1):
                    color, width = "#e74c3c", 2
                else:
                    color, width = "#555555", 1

                self.canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

        # Рисуем кластеры (пунктирные окружности)
        for cv, vertices in cluster_groups.items():
            if len(vertices) > 1:
                cx = sum(positions[v][0] for v in vertices) / len(vertices)
                cy = sum(positions[v][1] for v in vertices) / len(vertices)
                radius = 0
                for v in vertices:
                    dx = positions[v][0] - cx
                    dy = positions[v][1] - cy
                    dist = math.sqrt(dx * dx + dy * dy)
                    radius = max(radius, dist + 15)

                color = colors[cv % len(colors)]
                # self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                # fill='', outline=color, width=2, dash=(5, 5))

        # Рисуем вершины
        for i in range(graph.num_vertices):
            x, y = positions[i]
            weight = graph.get_vertex_weight(i)
            r = 12 * (0.7 + weight / 15)
            r = min(r, 18)

            cv = clusters.get(i, i)
            color = colors[cv % len(colors)]

            # Находим главную вершину в кластере (с наибольшим весом)
            if cv in cluster_groups:
                vertices_in_cluster = cluster_groups[cv]
                main_vertex = max(vertices_in_cluster, key=lambda v: graph.get_vertex_weight(v))

                self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="white", width=2)
                self.canvas.create_text(x, y, text=str(i), fill="white", font=("Arial", 10, "bold"))

                # Стрелка от меньшей вершины к главной
                if i != main_vertex:
                    mx, my = positions[main_vertex]
                    self.canvas.create_line(x, y, mx, my, fill=color, width=2, arrow="last", arrowshape=(10, 10, 5))
            else:
                self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="white", width=2)
                self.canvas.create_text(x, y, text=str(i), fill="white", font=("Arial", 10, "bold"))

            if weight != 1:
                self.canvas.create_text(x, y + r + 4, text=f"w={weight}", fill="#aaaaaa", font=("Arial", 7))

        # Легенда
        self.canvas.create_text(
            10, 20, anchor="nw", text="Red edges = will be contracted", fill="#e74c3c", font=("Arial", 9)
        )
        self.canvas.create_text(
            10, 40, anchor="nw", text="Dashed circle = vertices contracted together", fill="#ffffff", font=("Arial", 9)
        )
        self.canvas.create_text(
            10,
            60,
            anchor="nw",
            text="Arrow = vertex with smaller weight merges into larger",
            fill="#ffffff",
            font=("Arial", 9),
        )

    def _show_current(self):
        level = self.levels[self.current]
        graph = level.graph
        coarse_graph = level.coarse_graph

        # Получаем reverse_map
        reverse_map = getattr(level, "reverse_map", None)
        if reverse_map is None:
            reverse_map = coarse_graph._coarse_to_original if hasattr(coarse_graph, "_coarse_to_original") else []

        self.label.config(
            text=f"Level {self.current}: {graph.num_vertices} vertices (compression: {level.compression_ratio:.3f})"
        )
        self._draw_with_clusters(graph, coarse_graph, reverse_map)

    def _show_original(self):
        graph = self.levels[0].graph
        self.label.config(text=f"Original Graph: {graph.num_vertices} vertices")
        self._draw_graph(graph, is_original=True)

    def _draw_current(self):
        if self.current == -1:
            self._show_original()
        else:
            self._show_current()

    def _randomize_layout(self):
        """Случайное расположение вершин для текущего графа"""
        if self.current == -1:
            graph = self.levels[0].graph
        else:
            graph = self.levels[self.current].graph

        key = id(graph)
        self.node_positions[key] = self._randomize_positions(graph)
        self._draw_current()

    def _apply_force_layout(self):
        """Применить force-directed layout для текущего графа"""
        if self.current == -1:
            graph = self.levels[0].graph
        else:
            graph = self.levels[self.current].graph

        key = id(graph)
        positions = self._get_positions(graph)
        new_positions = self._compute_force_layout(graph, positions)
        self.node_positions[key] = new_positions
        self._draw_current()

    def _prev(self):
        if self.current == -1:
            self.current = len(self.levels) - 1
        elif self.current > 0:
            self.current -= 1
        else:
            return
        self._show_current()

    def _next(self):
        if self.current == -1:
            return
        elif self.current < len(self.levels) - 1:
            self.current += 1
            self._show_current()
        else:
            self.current = -1
            self._show_original()
