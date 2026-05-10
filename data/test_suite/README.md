# Test Suite for Graph Partitioning

## Format

```
<num_vertices> <num_edges> 1
<vertex_weight_1> <vertex_weight_2> ... <vertex_weight_n>
<u1> <v1> <edge_weight_1>
...
```

## Generated Graphs

### CLUSTER

| Size | Vertices | Edges | File |
|------|----------|-------|------|

### FAST_CLUSTER

| Size | Vertices | Edges | File |
|------|----------|-------|------|
| 50 | 50 | 400 | `fast_cluster/fast_cluster_50.txt` |
| 100 | 100 | 800 | `fast_cluster/fast_cluster_100.txt` |
| 200 | 200 | 1600 | `fast_cluster/fast_cluster_200.txt` |
| 500 | 500 | 4000 | `fast_cluster/fast_cluster_500.txt` |
| 1000 | 1000 | 8000 | `fast_cluster/fast_cluster_1000.txt` |
| 2000 | 2000 | 16000 | `fast_cluster/fast_cluster_2000.txt` |
| 5000 | 5000 | 40000 | `fast_cluster/fast_cluster_5000.txt` |

### BARABASI_ALBERT

| Size | Vertices | Edges | File |
|------|----------|-------|------|
| 50 | 50 | 100 | `barabasi_albert/barabasi_albert_50.txt` |
| 100 | 100 | 485 | `barabasi_albert/barabasi_albert_100.txt` |
| 200 | 200 | 985 | `barabasi_albert/barabasi_albert_200.txt` |
| 500 | 500 | 2485 | `barabasi_albert/barabasi_albert_500.txt` |
| 1000 | 1000 | 4985 | `barabasi_albert/barabasi_albert_1000.txt` |
| 2000 | 2000 | 9985 | `barabasi_albert/barabasi_albert_2000.txt` |
| 5000 | 5000 | 24985 | `barabasi_albert/barabasi_albert_5000.txt` |

### ERDOS_RENYI

| Size | Vertices | Edges | File |
|------|----------|-------|------|

### COMPLETE

| Size | Vertices | Edges | File |
|------|----------|-------|------|
| 50 | 50 | 1225 | `complete/complete_50.txt` |
| 100 | 100 | 4950 | `complete/complete_100.txt` |
| 200 | 200 | 19900 | `complete/complete_200.txt` |
| 500 | 500 | 124750 | `complete/complete_500.txt` |

### PATH

| Size | Vertices | Edges | File |
|------|----------|-------|------|
| 50 | 50 | 49 | `path/path_50.txt` |
| 100 | 100 | 99 | `path/path_100.txt` |
| 200 | 200 | 199 | `path/path_200.txt` |
| 500 | 500 | 499 | `path/path_500.txt` |
| 1000 | 1000 | 999 | `path/path_1000.txt` |
| 2000 | 2000 | 1999 | `path/path_2000.txt` |
| 5000 | 5000 | 4999 | `path/path_5000.txt` |

### GRID

| Size | Vertices | Edges | File |
|------|----------|-------|------|
| 49 | 49 | 84 | `grid/grid_49.txt` |
| 100 | 100 | 180 | `grid/grid_100.txt` |
| 196 | 196 | 364 | `grid/grid_196.txt` |
| 484 | 484 | 924 | `grid/grid_484.txt` |
| 961 | 961 | 1860 | `grid/grid_961.txt` |
| 1936 | 1936 | 3784 | `grid/grid_1936.txt` |
| 4900 | 4900 | 9660 | `grid/grid_4900.txt` |


## Usage

```python
from core.graph import Graph

graph = Graph.load_from_file('data/test_suite/cluster/cluster_100.txt')
print(f'Vertices: {graph.num_vertices}')
print(f'Edges: {graph.num_edges}')
print(f'Vertex weights: {[graph.get_vertex_weight(v) for v in range(5)]}')
```
