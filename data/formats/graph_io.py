"""
Модуль для чтения и записи графов в различных форматах
Поддерживаемые форматы:
- .txt: простой формат с вершинами и рёбрами (с поддержкой весов)
- .graph: формат DIMACS
- .mtx: Matrix Market формат
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Iterator
from pathlib import Path
import struct

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.graph import Graph


class GraphIO:
    """
    Утилиты для импорта/экспорта графов в различных форматах
    """
    
    @staticmethod
    def read_txt(filename: str, weighted: bool = True, has_vertex_weights: bool = False) -> Graph:
        """
        Чтение графа из простого текстового формата
        
        Формат:
            <num_vertices> <num_edges>
            [vertex_weight1] [vertex_weight2] ... (опционально)
            <vertex1> <vertex2> [weight]
            ...
        
        Args:
            filename: путь к файлу
            weighted: читать ли веса рёбер
            has_vertex_weights: есть ли в файле веса вершин (отдельной строкой после заголовка)
        
        Returns:
            Graph: загруженный граф
        """
        with open(filename, 'r') as f:
            # Читаем заголовок
            line = f.readline().strip()
            while line == '':
                line = f.readline().strip()
            
            parts = line.split()
            n = int(parts[0])
            m = int(parts[1])
            
            graph = Graph(n)
            
            # Читаем веса вершин (если есть)
            if has_vertex_weights:
                weights_line = f.readline().strip()
                while weights_line == '':
                    weights_line = f.readline().strip()
                vertex_weights = list(map(int, weights_line.split()))
                if len(vertex_weights) == n:
                    for v, w in enumerate(vertex_weights):
                        graph.set_vertex_weight(v, w)
            
            # Читаем рёбра
            for _ in range(m):
                line = f.readline().strip()
                while line == '':
                    line = f.readline().strip()
                
                parts = line.split()
                u = int(parts[0]) - 1  # В файле нумерация с 1
                v = int(parts[1]) - 1
                
                if weighted and len(parts) >= 3:
                    w = int(parts[2])
                    graph.add_edge(u, v, w)
                else:
                    graph.add_edge(u, v, 1)
        
        return graph
    
    @staticmethod
    def write_txt(graph: Graph, filename: str, include_vertex_weights: bool = False) -> None:
        """
        Запись графа в текстовый формат
        
        Args:
            graph: граф для сохранения
            filename: путь к файлу
            include_vertex_weights: сохранять ли веса вершин
        """
        with open(filename, 'w') as f:
            f.write(f"{graph.num_vertices} {graph.num_edges}\n")
            
            # Сохраняем веса вершин
            if include_vertex_weights:
                weights = [graph.get_vertex_weight(v) for v in range(graph.num_vertices)]
                f.write(" ".join(map(str, weights)) + "\n")
            
            # Сохраняем рёбра (нумерация с 1 для совместимости)
            for u, v, w in graph.edges():
                if w != 1:
                    f.write(f"{u+1} {v+1} {w}\n")
                else:
                    f.write(f"{u+1} {v+1}\n")
        
        print(f"Graph saved to {filename} (V={graph.num_vertices}, E={graph.num_edges})")
    
    @staticmethod
    def read_dimacs(filename: str) -> Graph:
        """
        Чтение графа в формате DIMACS
        
        Формат:
            p edge <num_vertices> <num_edges>
            e <u> <v> [w]
            n <vertex> <weight>  - вес вершины
        
        Args:
            filename: путь к файлу
        
        Returns:
            Graph: загруженный граф
        """
        graph = None
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('c'):
                    continue
                
                parts = line.split()
                
                if parts[0] == 'p':
                    if parts[1] == 'edge':
                        n = int(parts[2])
                        m = int(parts[3])
                        graph = Graph(n)
                
                elif parts[0] == 'n':
                    # Вес вершины
                    if graph is None:
                        raise ValueError("No problem line found before vertex weights")
                    v = int(parts[1]) - 1
                    w = int(parts[2]) if len(parts) > 2 else 1
                    graph.set_vertex_weight(v, w)
                
                elif parts[0] == 'e':
                    if graph is None:
                        raise ValueError("No problem line found before edges")
                    
                    u = int(parts[1]) - 1
                    v = int(parts[2]) - 1
                    w = int(parts[3]) if len(parts) > 3 else 1
                    graph.add_edge(u, v, w)
        
        if graph is None:
            raise ValueError("No graph data found in file")
        
        return graph
    
    @staticmethod
    def write_dimacs(graph: Graph, filename: str, include_vertex_weights: bool = True) -> None:
        """
        Запись графа в формате DIMACS
        
        Args:
            graph: граф для сохранения
            filename: путь к файлу
            include_vertex_weights: сохранять ли веса вершин
        """
        with open(filename, 'w') as f:
            f.write(f"c Graph with {graph.num_vertices} vertices and {graph.num_edges} edges\n")
            f.write(f"p edge {graph.num_vertices} {graph.num_edges}\n")
            
            # Сохраняем веса вершин
            if include_vertex_weights:
                for v in range(graph.num_vertices):
                    w = graph.get_vertex_weight(v)
                    if w != 1:
                        f.write(f"n {v+1} {w}\n")
            
            # Сохраняем рёбра
            for u, v, w in graph.edges():
                if w != 1:
                    f.write(f"e {u+1} {v+1} {w}\n")
                else:
                    f.write(f"e {u+1} {v+1}\n")
        
        print(f"Graph saved to {filename} in DIMACS format")
    
    @staticmethod
    def read_metis(filename: str) -> Graph:
        """
        Чтение графа в формате METIS
        
        Формат:
            <num_vertices> <num_edges> [format]
            <vertex_weight> [optional]
            <neighbor1> <neighbor2> ... (для каждой вершины)
        
        Args:
            filename: путь к файлу
        
        Returns:
            Graph: загруженный граф
        """
        with open(filename, 'r') as f:
            # Читаем заголовок
            header = f.readline().strip()
            while header.startswith('%'):
                header = f.readline().strip()
            
            parts = header.split()
            n = int(parts[0])
            m = int(parts[1])
            fmt = int(parts[2]) if len(parts) > 2 else 0  # 0=спарсенная, 1=веса вершин
            
            graph = Graph(n)
            
            # Читаем веса вершин (если есть)
            if fmt == 1:
                weights_line = f.readline().strip()
                while weights_line == '':
                    weights_line = f.readline().strip()
                weights = list(map(int, weights_line.split()))
                for v, w in enumerate(weights[:n]):
                    graph.set_vertex_weight(v, w)
            
            # Читаем списки смежности для каждой вершины
            current_vertex = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                numbers = list(map(int, line.split()))
                
                # Первое число может быть весом вершины или степенью
                start_idx = 0
                if fmt == 1 and current_vertex < len(weights):
                    # Уже прочитали веса
                    start_idx = 0
                elif len(numbers) > 1 and numbers[0] == len(numbers) - 1:
                    # Первое число - степень
                    start_idx = 1
                
                for j in range(start_idx, len(numbers)):
                    neighbor = numbers[j] - 1  # METIS использует нумерацию с 1
                    if neighbor != current_vertex and neighbor >= 0:
                        graph.add_edge(current_vertex, neighbor, 1)
                
                current_vertex += 1
                if current_vertex >= n:
                    break
        
        return graph
    
    @staticmethod
    def write_metis(graph: Graph, filename: str, include_vertex_weights: bool = True) -> None:
        """
        Запись графа в формате METIS
        
        Args:
            graph: граф для сохранения
            filename: путь к файлу
            include_vertex_weights: сохранять ли веса вершин
        """
        with open(filename, 'w') as f:
            fmt = 1 if include_vertex_weights else 0
            f.write(f"{graph.num_vertices} {graph.num_edges} {fmt}\n")
            
            # Сохраняем веса вершин
            if include_vertex_weights:
                weights = [str(graph.get_vertex_weight(v)) for v in range(graph.num_vertices)]
                f.write(" ".join(weights) + "\n")
            
            # Сохраняем списки смежности
            for v in range(graph.num_vertices):
                neighbors = [n for n, _ in graph.get_neighbors(v)]
                neighbor_str = " ".join(str(n+1) for n in sorted(neighbors))
                f.write(f"{neighbor_str}\n")
        
        print(f"Graph saved to {filename} in METIS format")
    
    @staticmethod
    def read_matrix_market(filename: str) -> Graph:
        """
        Чтение графа в формате Matrix Market (.mtx)
        
        Формат:
            %%MatrixMarket matrix coordinate real general
            <rows> <cols> <nnz>
            <i> <j> <value>
        
        Args:
            filename: путь к файлу
        
        Returns:
            Graph: загруженный граф
        """
        with open(filename, 'r') as f:
            # Пропускаем комментарии
            line = f.readline()
            while line.startswith('%'):
                line = f.readline()
            
            # Заголовок
            parts = line.split()
            n = int(parts[0])
            m = int(parts[1])
            nnz = int(parts[2])
            
            graph = Graph(n)
            
            # Читаем рёбра
            for _ in range(nnz):
                line = f.readline()
                while line.startswith('%'):
                    line = f.readline()
                
                parts = line.split()
                i = int(parts[0]) - 1
                j = int(parts[1]) - 1
                w = float(parts[2]) if len(parts) > 2 else 1.0
                
                # Преобразуем float в int для весов
                if w != int(w):
                    raise ValueError("Matrix Market with float weights not supported")
                
                graph.add_edge(i, j, int(w))
        
        return graph
    
    @staticmethod
    def write_matrix_market(graph: Graph, filename: str) -> None:
        """
        Запись графа в формате Matrix Market (.mtx)
        
        Args:
            graph: граф для сохранения
            filename: путь к файлу
        """
        with open(filename, 'w') as f:
            f.write("%%MatrixMarket matrix coordinate real general\n")
            f.write(f"{graph.num_vertices} {graph.num_vertices} {graph.num_edges}\n")
            
            for u, v, w in graph.edges():
                f.write(f"{u+1} {v+1} {w}\n")
        
        print(f"Graph saved to {filename} in Matrix Market format")
    
    @staticmethod
    def read_binary(filename: str) -> Graph:
        """
        Чтение графа из бинарного файла (для больших графов)
        
        Формат:
            [n: int64] [m: int64]
            [vertex_weights: n * int32]
            [edges: m * (u: int32, v: int32, w: int32)]
        
        Args:
            filename: путь к файлу
        
        Returns:
            Graph: загруженный граф
        """
        with open(filename, 'rb') as f:
            n = struct.unpack('q', f.read(8))[0]
            m = struct.unpack('q', f.read(8))[0]
            
            graph = Graph(n)
            
            # Читаем веса вершин
            vertex_weights = struct.unpack(f'{n}i', f.read(n * 4))
            for v, w in enumerate(vertex_weights):
                graph.set_vertex_weight(v, w)
            
            # Читаем рёбра
            for _ in range(m):
                u, v, w = struct.unpack('iii', f.read(12))
                graph.add_edge(u, v, w)
        
        return graph
    
    @staticmethod
    def write_binary(graph: Graph, filename: str) -> None:
        """
        Запись графа в бинарный файл (для больших графов)
        
        Args:
            graph: граф для сохранения
            filename: путь к файлу
        """
        with open(filename, 'wb') as f:
            f.write(struct.pack('q', graph.num_vertices))
            f.write(struct.pack('q', graph.num_edges))
            
            # Сохраняем веса вершин
            for v in range(graph.num_vertices):
                f.write(struct.pack('i', graph.get_vertex_weight(v)))
            
            # Сохраняем рёбра
            for u, v, w in graph.edges():
                f.write(struct.pack('iii', u, v, w))
        
        print(f"Graph saved to {filename} in binary format")
    
    @staticmethod
    def generate_test_graphs(output_dir: str = "data/test_cases") -> Dict[str, Graph]:
        """
        Генерация набора тестовых графов для экспериментов
        
        Args:
            output_dir: директория для сохранения
        
        Returns:
            Dict[str, Graph]: словарь с именами и графами
        """
        from ..generators.cluster_generator import FastClusterGenerator
        from ..generators.barabasi_albert import BarabasiAlbertGenerator
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        test_graphs = {}
        
        # 1. Маленький кластерный граф (для визуализации)
        print("Generating small cluster graph...")
        gen_small = FastClusterGenerator(
            num_clusters=3,
            vertices_per_cluster=10,
            target_edges=500,
            intra_ratio=0.8,
            seed=42
        )
        graph_small = gen_small.generate()
        GraphIO.write_txt(graph_small, output_path / "small_cluster.txt", include_vertex_weights=True)
        test_graphs['small_cluster'] = graph_small
        
        # 2. Средний кластерный граф
        print("Generating medium cluster graph...")
        gen_medium = FastClusterGenerator(
            num_clusters=5,
            vertices_per_cluster=50,
            target_edges=5000,
            intra_ratio=0.7,
            seed=123
        )
        graph_medium = gen_medium.generate()
        GraphIO.write_txt(graph_medium, output_path / "medium_cluster.txt", include_vertex_weights=True)
        test_graphs['medium_cluster'] = graph_medium
        
        # 3. Большой кластерный граф
        print("Generating large cluster graph...")
        gen_large = FastClusterGenerator(
            num_clusters=10,
            vertices_per_cluster=100,
            target_edges=50000,
            intra_ratio=0.7,
            seed=456
        )
        graph_large = gen_large.generate()
        GraphIO.write_txt(graph_large, output_path / "large_cluster.txt", include_vertex_weights=True)
        test_graphs['large_cluster'] = graph_large
        
        # 4. Граф Барабаши-Альберт
        print("Generating Barabasi-Albert graph...")
        gen_ba = BarabasiAlbertGenerator(
            n=1000,
            m0=10,
            m=3,
            weight_range=(1, 5),
            vertex_weight_range=(1, 10),
            seed=789
        )
        graph_ba = gen_ba.generate()
        GraphIO.write_txt(graph_ba, output_path / "barabasi_albert.txt", include_vertex_weights=True)
        test_graphs['barabasi_albert'] = graph_ba
        
        print(f"\nAll test graphs saved to {output_dir}/")
        return test_graphs
    
    @staticmethod
    def read_graph_info(filename: str) -> Tuple[int, int, float, Optional[Tuple[int, int]]]:
        """
        Быстрое чтение только информации о графе без загрузки
        
        Args:
            filename: путь к файлу
        
        Returns:
            Tuple[int, int, float, Optional]: (число вершин, число рёбер, плотность, диапазон весов)
        """
        with open(filename, 'r') as f:
            # Пропускаем пустые строки и комментарии
            line = f.readline()
            while line and (line.strip() == '' or line.startswith('%') or line.startswith('c')):
                line = f.readline()
            
            if not line:
                return 0, 0, 0.0, None
            
            parts = line.split()
            n = int(parts[0])
            m = int(parts[1])
            
            density = 2 * m / (n * (n - 1)) if n > 1 else 0
            
            # Пробуем определить диапазон весов
            min_w = float('inf')
            max_w = -float('inf')
            weight_range = None
            
            # Читаем несколько строк для определения весов
            count = 0
            for line in f:
                if count > 1000:  # Ограничиваем
                    break
                line = line.strip()
                if not line or line.startswith('%') or line.startswith('c'):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    w = int(parts[2])
                    min_w = min(min_w, w)
                    max_w = max(max_w, w)
                    weight_range = (min_w, max_w)
                count += 1
        
        return n, m, density, weight_range


class GraphBatchConverter:
    """
    Утилита для пакетного конвертирования графов между форматами
    """
    
    @staticmethod
    def convert_directory(input_dir: str, 
                         output_dir: str,
                         input_format: str = 'txt',
                         output_format: str = 'dimacs') -> None:
        """
        Конвертирование всех графов в директории
        
        Args:
            input_dir: входная директория
            output_dir: выходная директория
            input_format: формат входных файлов ('txt', 'dimacs', 'metis', 'mtx', 'binary')
            output_format: формат выходных файлов
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Расширения файлов
        extensions = {
            'txt': '.txt',
            'dimacs': '.graph',
            'metis': '.metis',
            'mtx': '.mtx',
            'binary': '.bin'
        }
        
        input_ext = extensions.get(input_format, '.txt')
        output_ext = extensions.get(output_format, '.txt')
        
        # Методы чтения/записи
        readers = {
            'txt': GraphIO.read_txt,
            'dimacs': GraphIO.read_dimacs,
            'metis': GraphIO.read_metis,
            'mtx': GraphIO.read_matrix_market,
            'binary': GraphIO.read_binary
        }
        
        writers = {
            'txt': GraphIO.write_txt,
            'dimacs': GraphIO.write_dimacs,
            'metis': GraphIO.write_metis,
            'mtx': GraphIO.write_matrix_market,
            'binary': GraphIO.write_binary
        }
        
        reader = readers.get(input_format)
        writer = writers.get(output_format)
        
        if not reader or not writer:
            raise ValueError(f"Unsupported format: {input_format} or {output_format}")
        
        # Конвертируем каждый файл
        for file_path in input_path.glob(f"*{input_ext}"):
            print(f"Converting {file_path.name}...")
            
            try:
                graph = reader(str(file_path))
                output_file = output_path / f"{file_path.stem}{output_ext}"
                writer(graph, str(output_file))
                print(f"  -> Saved to {output_file}")
            except Exception as e:
                print(f"  -> Error: {e}")
    
    @staticmethod
    def merge_graphs(graphs: List[Graph], output_file: str, format: str = 'txt') -> Graph:
        """
        Объединение нескольких графов в один (несвязный)
        
        Args:
            graphs: список графов
            output_file: выходной файл
            format: формат сохранения
        
        Returns:
            Graph: объединённый граф
        """
        total_vertices = sum(g.num_vertices for g in graphs)
        total_edges = sum(g.num_edges for g in graphs)
        
        merged = Graph(total_vertices)
        
        # Смещение вершин
        offset = 0
        for g in graphs:
            # Копируем веса вершин
            for v in range(g.num_vertices):
                merged.set_vertex_weight(offset + v, g.get_vertex_weight(v))
            
            # Копируем рёбра
            for u, v, w in g.edges():
                merged.add_edge(offset + u, offset + v, w)
            
            offset += g.num_vertices
        
        # Сохраняем
        writers = {
            'txt': GraphIO.write_txt,
            'dimacs': GraphIO.write_dimacs,
            'metis': GraphIO.write_metis,
            'mtx': GraphIO.write_matrix_market,
            'binary': GraphIO.write_binary
        }
        
        writer = writers.get(format, GraphIO.write_txt)
        writer(merged, output_file)
        
        print(f"Merged graph saved to {output_file} (V={total_vertices}, E={total_edges})")
        
        return merged