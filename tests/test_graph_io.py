# test_graph_io.py
import sys
import tempfile
from pathlib import Path

sys.path.append('.')

from core.graph import Graph
from data.formats.graph_io import GraphIO


def test_all_formats():
    """Тест всех форматов ввода/вывода"""
    
    print("=" * 60)
    print("ТЕСТ ФОРМАТОВ ВВОДА/ВЫВОДА")
    print("=" * 60)
    
    # Создаём тестовый граф
    g = Graph(5)
    edges = [(0, 1, 5), (0, 2, 3), (1, 3, 2), (2, 4, 4), (3, 4, 1)]
    for u, v, w in edges:
        g.add_edge(u, v, w)
    
    # Веса вершин
    for v in range(5):
        g.set_vertex_weight(v, v + 1)
    
    print(f"Исходный граф: {g}")
    print(f"  Веса вершин: {[g.get_vertex_weight(v) for v in range(5)]}")
    print(f"  Рёбра: {list(g.edges())}")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Тест TXT формата
        print("\n--- TXT format ---")
        txt_file = tmp_path / "test.txt"
        GraphIO.write_txt(g, txt_file, include_vertex_weights=True)
        g_txt = GraphIO.read_txt(txt_file, weighted=True, has_vertex_weights=True)
        print(f"  Прочитано: {g_txt}")
        
        # Тест DIMACS формата
        print("\n--- DIMACS format ---")
        dimacs_file = tmp_path / "test.graph"
        GraphIO.write_dimacs(g, dimacs_file, include_vertex_weights=True)
        g_dimacs = GraphIO.read_dimacs(dimacs_file)
        print(f"  Прочитано: {g_dimacs}")
        
        # Тест METIS формата
        print("\n--- METIS format ---")
        metis_file = tmp_path / "test.metis"
        GraphIO.write_metis(g, metis_file, include_vertex_weights=True)
        g_metis = GraphIO.read_metis(metis_file)
        print(f"  Прочитано: {g_metis}")
        
        # Тест бинарного формата
        print("\n--- BINARY format ---")
        bin_file = tmp_path / "test.bin"
        GraphIO.write_binary(g, bin_file)
        g_bin = GraphIO.read_binary(bin_file)
        print(f"  Прочитано: {g_bin}")
        
        # Проверка идентичности
        print("\n--- VERIFICATION ---")
        assert g_txt.num_vertices == g.num_vertices
        assert g_txt.num_edges == g.num_edges
        assert g_dimacs.num_vertices == g.num_vertices
        assert g_metis.num_vertices == g.num_vertices
        assert g_bin.num_vertices == g.num_vertices
        print("✅ Все форматы работают корректно!")
        
        # Тест чтения информации
        print("\n--- READ INFO ---")
        n, m, density, w_range = GraphIO.read_graph_info(txt_file)
        print(f"  Vertices: {n}")
        print(f"  Edges: {m}")
        print(f"  Density: {density:.6f}")
        print(f"  Weight range: {w_range}")


def test_generate_test_graphs():
    """Тест генерации тестовых графов"""
    
    print("\n" + "=" * 60)
    print("ТЕСТ ГЕНЕРАЦИИ ТЕСТОВЫХ ГРАФОВ")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        graphs = GraphIO.generate_test_graphs(tmpdir)
        
        for name, graph in graphs.items():
            print(f"\n{name}:")
            print(f"  Vertices: {graph.num_vertices}")
            print(f"  Edges: {graph.num_edges}")
            print(f"  Vertex weights: {[graph.get_vertex_weight(v) for v in range(min(5, graph.num_vertices))]}...")
        
        print(f"\n✅ Тестовые графы сохранены в {tmpdir}")


if __name__ == "__main__":
    test_all_formats()
    test_generate_test_graphs()