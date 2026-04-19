"""
Модуль для чтения и записи графов в различных форматах
Поддерживаемые форматы:
- .txt: простой формат с вершинами и рёбрами
- .graph: формат DIMACS
- .mtx: Matrix Market формат
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.graph import Graph


class GraphIO:
    """
    Утилиты для импорта/экспорта графов в различных форматах
    """
    
    @staticmethod
    def read_txt(filename: str, weighted: bool = False) -> Graph:
        """
        Чтение графа из простого текстового формата
        
        Формат:
            <num_vertices> <num_edges>
            <vertex1> <vertex2> [weight]
            ...
        
        Args:
            filename: путь к файлу
            weighted: читать ли веса рёбер
        
        Returns:
            Graph: загруженный граф
        """
        return Graph.load_from_file(filename, weighted)
    
    @staticmethod
    def write_txt(graph: Graph, filename: str) -> None:
        """
        Запись графа в текстовый формат
        
        Args:
            graph: граф для сохранения
            filename: путь к файлу
        """
        graph.save_to_file(filename)
        print(f"Graph saved to {filename} (V={graph.num_vertices}, E={graph.num_edges})")
    
    @staticmethod
    def read_dimacs(filename: str) -> Graph:
        """
        Чтение графа в формате DIMACS
        
        Формат:
            p edge <num_vertices> <num_edges>
            e <u> <v> [w]
        
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
                    # Проблемная строка
                    if parts[1] == 'edge':
                        n = int(parts[2])
                        m = int(parts[3])
                        graph = Graph(n)
                
                elif parts[0] == 'e':
                    # Ребро
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
    def write_dimacs(graph: Graph, filename: str) -> None:
        """
        Запись графа в формате DIMACS
        
        Args:
            graph: граф для сохранения
            filename: путь к файлу
        """
        with open(filename, 'w') as f:
            f.write(f"c Graph with {graph.num_vertices} vertices and {graph.num_edges} edges\n")
            f.write(f"p edge {graph.num_vertices} {graph.num_edges}\n")
            
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
            <vertex_degree> <neighbor1> <neighbor2> ...
        
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
            
            graph = Graph(n)
            
            # Читаем списки смежности
            for v in range(n):
                line = f.readline().strip()
                while line == '':
                    line = f.readline().strip()
                
                neighbors = list(map(int, line.split()))
                if len(neighbors) > 1:
                    # Первое число может быть степенью или весом
                    degree = neighbors[0] if len(neighbors) > 1 else 0
                    start_idx = 1 if degree == len(neighbors) - 1 else 0
                    
                    for j in range(start_idx, len(neighbors)):
                        u = neighbors[j] - 1  # METIS использует нумерацию с 1
                        if u != v and u >= 0:
                            graph.add_edge(v, u)
        
        return graph
    
    @staticmethod
    def write_metis(graph: Graph, filename: str) -> None:
        """
        Запись графа в формате METIS
        
        Args:
            graph: граф для сохранения
            filename: путь к файлу
        """
        with open(filename, 'w') as f:
            f.write(f"{graph.num_vertices} {graph.num_edges} 0\n")
            
            for v in range(graph.num_vertices):
                neighbors = list(graph.get_neighbors(v).keys())
                # METIS формат требует список соседей (нумерация с 1)
                neighbor_str = " ".join(str(n+1) for n in sorted(neighbors))
                f.write(f"{neighbor_str}\n")
        
        print(f"Graph saved to {filename} in METIS format")
    
    @staticmethod
    def generate_test_graphs(output_dir: str = "data/test_cases") -> Dict[str, Graph]:
        """
        Генерация набора тестовых графов для экспериментов
        
        Args:
            output_dir: директория для сохранения
        
        Returns:
            Dict[str, Graph]: словарь с именами и графами
        """
        from ..generators.cluster_generator import ClusterGraphGenerator
        from ..generators.barabasi_albert import BarabasiAlbertGenerator, ErdosRenyiGenerator
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        test_graphs = {}
        
        # 1. Маленький кластерный граф (для визуализации)
        print("Generating small cluster graph...")
        gen_small = ClusterGraphGenerator(num_clusters=3, cluster_size=10, intra_prob=0.9, inter_prob=0.05, seed=42)
        graph_small = gen_small.generate()
        GraphIO.write_txt(graph_small, output_path / "small_cluster.txt")
        test_graphs['small_cluster'] = graph_small
        
        # 2. Средний кластерный граф
        print("Generating medium cluster graph...")
        gen_medium = ClusterGraphGenerator(num_clusters=10, cluster_size=50, intra_prob=0.8, inter_prob=0.03, seed=123)
        graph_medium = gen_medium.generate()
        GraphIO.write_txt(graph_medium, output_path / "medium_cluster.txt")
        test_graphs['medium_cluster'] = graph_medium
        
        # 3. Большой кластерный граф
        print("Generating large cluster graph...")
        gen_large = ClusterGraphGenerator(num_clusters=20, cluster_size=100, intra_prob=0.7, inter_prob=0.02, seed=456)
        graph_large = gen_large.generate()
        GraphIO.write_txt(graph_large, output_path / "large_cluster.txt")
        test_graphs['large_cluster'] = graph_large
        
        # 4. Граф Барабаши-Альберт
        print("Generating Barabasi-Albert graph...")
        gen_ba = BarabasiAlbertGenerator(n=1000, m0=10, m=3, seed=789)
        graph_ba = gen_ba.generate()
        GraphIO.write_txt(graph_ba, output_path / "barabasi_albert.txt")
        test_graphs['barabasi_albert'] = graph_ba
        
        # 5. Случайный граф
        print("Generating Erdos-Renyi graph...")
        gen_er = ErdosRenyiGenerator(n=500, p=0.02, seed=101)
        graph_er = gen_er.generate()
        GraphIO.write_txt(graph_er, output_path / "erdos_renyi.txt")
        test_graphs['erdos_renyi'] = graph_er
        
        print(f"\nAll test graphs saved to {output_dir}/")
        return test_graphs
    
    @staticmethod
    def read_graph_info(filename: str) -> Tuple[int, int, float]:
        """
        Быстрое чтение только информации о графе без загрузки
        
        Args:
            filename: путь к файлу
        
        Returns:
            Tuple[int, int, float]: (число вершин, число рёбер, плотность)
        """
        with open(filename, 'r') as f:
            # Пропускаем пустые строки
            line = f.readline().strip()
            while line == '':
                line = f.readline().strip()
            
            parts = line.split()
            n = int(parts[0])
            m = int(parts[1])
            
            density = 2 * m / (n * (n - 1)) if n > 1 else 0
        
        return n, m, density


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
            input_format: формат входных файлов ('txt', 'dimacs', 'metis')
            output_format: формат выходных файлов
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Находим все файлы с нужным расширением
        extensions = {
            'txt': '.txt',
            'dimacs': '.graph',
            'metis': '.metis'
        }
        
        input_ext = extensions.get(input_format, '.txt')
        output_ext = extensions.get(output_format, '.txt')
        
        # Методы чтения/записи
        readers = {
            'txt': GraphIO.read_txt,
            'dimacs': GraphIO.read_dimacs,
            'metis': GraphIO.read_metis
        }
        
        writers = {
            'txt': GraphIO.write_txt,
            'dimacs': GraphIO.write_dimacs,
            'metis': GraphIO.write_metis
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