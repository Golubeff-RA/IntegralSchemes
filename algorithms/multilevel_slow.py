"""
Многоуровневое разбиение графа (Multilevel Partitioning)
Эффективная реализация с локальным KL только на границе
"""

import random
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.graph import Graph
from core.partition import Partition
from .base_partitioner import PartitionerWithStats


class CoarseningLevel:
    def __init__(self, graph: Graph, coarse_graph: Graph, node_map: List[int], reverse_map: Dict[int, List[int]], compression_ratio: float):
        self.graph = graph
        self.coarse_graph = coarse_graph
        self.node_map = node_map
        self.reverse_map = reverse_map
        self.compression_ratio = compression_ratio

class MultilevelPartitioner(PartitionerWithStats):
    """
    Многоуровневое разбиение: стягивание -> разбиение -> проекция + локальный KL.
    Локальный KL работает только с вершинами, которые могут улучшить разрез.
    """

    def __init__(self, coarsen_to: int = 50, max_passes: int = 10, seed: int = 42):
        super().__init__(name="Multilevel")
        self.coarsen_to = coarsen_to
        self.max_passes = max_passes
        self.seed = seed
        random.seed(seed)
        self.levels = []

    def _partition_impl(self, graph: Graph, balance_ratio: float = 0.5) -> Partition:
        print(f"\n  Multilevel partitioning (coarsen to {self.coarsen_to})")

        # 1. Coarsening
        current = graph
        self.levels = []
        while current.num_vertices > self.coarsen_to:
            matching = self._heavy_edge_matching(current)
            if not matching:
                break
            coarse, node_map, reverse_map = self._coarsen(current, matching)
            compression = coarse.num_vertices / current.num_vertices
            self.levels.append(CoarseningLevel(current, coarse, node_map, reverse_map, compression))
            current = coarse
            print(f"    Coarsened: {current.num_vertices} vertices")

        # 2. Initial partitioning on coarsest graph (try several seeds)
        best_part = None
        best_cut = float('inf')
        for trial in range(10):
            part = self._random_partition(current, balance_ratio, trial)
            # Full KL on coarse graph (small, so acceptable)
            part = self._full_kl(part, current, balance_ratio, passes=10)
            cut = part.cut_weight(current)
            if cut < best_cut:
                best_cut = cut
                best_part = part

        partition = best_part

        # 3. Uncoarsening with local boundary KL
        for level in reversed(self.levels):
            # Project partition to finer graph
            partition = self._project(partition, level.reverse_map, level.node_map)
            partition.update_weights(level.graph)

            # Local KL: only boundary vertices and their neighbours
            partition = self._local_kl(partition, level.graph, balance_ratio, passes=self.max_passes)

        return partition

    def _heavy_edge_matching(self, graph: Graph) -> List[Tuple[int, int]]:
        """Heavy‑edge matching (METIS style) – O(E log E) sorting"""
        used = [False] * graph.num_vertices
        matching = []
        edges = list(graph.edges())
        edges.sort(key=lambda x: x[2], reverse=True)
        for u, v, _ in edges:
            if not used[u] and not used[v]:
                matching.append((u, v))
                used[u] = used[v] = True
        return matching

    def _coarsen(self, graph: Graph, matching: List[Tuple[int, int]]) -> Tuple[Graph, List[int], Dict[int, List[int]]]:
        """Contract graph using matching, return coarse graph, node_map (old->new), reverse_map (new->list[old])"""
        n = graph.num_vertices
        node_map = [-1] * n
        next_id = 0

        for u, v in matching:
            node_map[u] = next_id
            node_map[v] = next_id
            next_id += 1

        for i in range(n):
            if node_map[i] == -1:
                node_map[i] = next_id
                next_id += 1

        reverse_map = {new: [] for new in range(next_id)}
        for old, new in enumerate(node_map):
            reverse_map[new].append(old)

        coarse = Graph(next_id)

        # Vertex weights: sum of original weights
        for new, old_list in reverse_map.items():
            total_w = sum(graph.get_vertex_weight(v) for v in old_list)
            coarse.set_vertex_weight(new, total_w)

        # Edge weights: sum of parallel edges between supernodes
        edge_weights = defaultdict(int)
        for u, v, w in graph.edges():
            nu, nv = node_map[u], node_map[v]
            if nu != nv:
                key = (min(nu, nv), max(nu, nv))
                edge_weights[key] += w

        for (u, v), w in edge_weights.items():
            coarse.add_edge(u, v, w)

        return coarse, node_map, reverse_map

    def _random_partition(self, graph: Graph, balance: float, seed: int) -> Partition:
        """Random balanced partition (by vertex count, not weight)"""
        random.seed(seed)
        n = graph.num_vertices
        part = Partition(n)
        target = int(n * balance)
        # Shuffle vertices to avoid bias
        vertices = list(range(n))
        random.shuffle(vertices)
        for i, v in enumerate(vertices):
            part.assign(v, 0 if i < target else 1)
        part.update_weights(graph)
        return part

    def _full_kl(self, partition: Partition, graph: Graph, balance: float, passes: int) -> Partition:
        """Full KL (used only on the coarsest graph)"""
        # We'll implement a simple gain-based local search that works on the whole graph
        # but only for small coarse graphs. This is fine because coarse graph is tiny.
        n = graph.num_vertices
        # Copy partition
        best_part = partition.copy()
        best_cut = partition.cut_weight(graph)
        
        for _ in range(passes):
            # Compute gains for all vertices
            gains = []
            for v in range(n):
                pv = partition.get_part(v)
                if pv == -1:
                    gains.append(0)
                    continue
                internal = external = 0
                for nb, w in graph.get_neighbors(v):
                    if partition.get_part(nb) == pv:
                        internal += w
                    else:
                        external += w
                gains.append(external - internal)
            
            # Find best swap pair
            best_delta = 0
            best_pair = None
            for i in range(n):
                if partition.get_part(i) != 0:
                    continue
                for j in range(n):
                    if partition.get_part(j) != 1:
                        continue
                    if i == j:
                        continue
                    delta = gains[i] + gains[j] - 2 * graph.get_edge_weight(i, j)
                    if delta > best_delta:
                        best_delta = delta
                        best_pair = (i, j)
            
            if best_delta <= 0:
                break
            
            i, j = best_pair
            partition.swap_vertices(i, j, graph)
        
        return partition

    def _project(self, part: Partition, reverse_map: Dict[int, List[int]], node_map: List[int]) -> Partition:
        """Project partition from coarse to fine using reverse map."""
        fine_n = sum(len(lst) for lst in reverse_map.values())
        fine = Partition(fine_n)
        for coarse_v, fine_list in reverse_map.items():
            coarse_part = part.get_part(coarse_v)
            if coarse_part != -1:
                for v in fine_list:
                    fine.assign(v, coarse_part)
        return fine

    def _local_kl(self, partition: Partition, graph: Graph, balance: float, passes: int) -> Partition:
        """
        Local Kernighan–Lin: only vertices that are on the boundary can move.
        For efficiency, we maintain a queue of boundary vertices and update gains only for neighbours.
        """
        n = graph.num_vertices
        
        # Helper to compute gain of a single vertex
        def gain(v: int) -> int:
            part_v = partition.get_part(v)
            if part_v == -1:
                return 0
            internal = external = 0
            for nb, w in graph.get_neighbors(v):
                if partition.get_part(nb) == part_v:
                    internal += w
                else:
                    external += w
            return external - internal

        # Initialise gains and boundary flags
        gains = [gain(v) for v in range(n)]
        boundary = [False] * n
        # Find boundary vertices
        for v in range(n):
            pv = partition.get_part(v)
            if pv == -1:
                continue
            for nb, _ in graph.get_neighbors(v):
                if partition.get_part(nb) != pv:
                    boundary[v] = True
                    break

        # KL iterations
        for _ in range(passes):
            # Collect boundary vertices with positive gain
            candidates = [v for v in range(n) if boundary[v] and gains[v] > 0]
            if not candidates:
                break

            # Sort by gain descending (largest improvement first)
            candidates.sort(key=lambda x: gains[x], reverse=True)

            moved = set()
            for v in candidates:
                if v in moved:
                    continue
                p_old = partition.get_part(v)
                if p_old == -1:
                    continue
                p_new = 1 - p_old
                # Check balance after moving
                if p_new == 0:
                    new0 = partition.size0 + 1
                    new1 = partition.size1 - 1
                else:
                    new0 = partition.size0 - 1
                    new1 = partition.size1 + 1
                target = n * balance
                max_sz = target * (1 + (1 - balance))
                min_sz = target * balance
                if new0 < min_sz or new1 < min_sz or new0 > max_sz or new1 > max_sz:
                    continue

                # Move vertex
                partition.move_vertex_to(v, p_new, graph)
                moved.add(v)

                # Update gains for neighbours and possibly v itself
                for nb, w in graph.get_neighbors(v):
                    if nb in moved:
                        continue
                    old_gain = gains[nb]
                    # Simple update: recompute gain from scratch (neighbors limited)
                    gains[nb] = gain(nb)
                    # Update boundary flag for neighbours
                    p_nb = partition.get_part(nb)
                    if p_nb == -1:
                        continue
                    is_boundary = False
                    for nnb, _ in graph.get_neighbors(nb):
                        if partition.get_part(nnb) != p_nb:
                            is_boundary = True
                            break
                    boundary[nb] = is_boundary
                # Recompute gain for moved vertex (it's now in new part)
                gains[v] = gain(v)
                boundary[v] = True  # after move it could still be boundary

        return partition

    def get_coarsening_history(self):
        return self.levels