from typing import List, Tuple
from functools import lru_cache
from dataclasses import dataclass
from joblib import Parallel, delayed
from heapq import heappush, heappushpop, nlargest
from threading import Lock

from k_tsp_solver import (
    Instance,
    Solution,
    Model,
    ModelName,
    KFactor,
    timeit,
    logger
)


@dataclass(unsafe_hash=True)
class NearestNeighborsV2(Model):
    name: str = str(ModelName.NEAREST_NEIGHBORS_V2.value)

    def __post_init__(self):
        self._min_edge_weight = None
        self._lock = Lock()
        self._top_k_heap = []

    @lru_cache
    def _sort_undirected_edges(self, instance: Instance) -> List[Tuple[int,int,dict]]:
        edges: List[Tuple[int,int,dict]] = []
        for u, v, data in instance.graph.edges(data=True):
            if u < v:
                edges.append((u, v, data))
        edges.sort(key=lambda x: x[2]["weight"])
        return edges

    @lru_cache
    def _neighbors_of(self, instance: Instance, vertex: int) -> List[Tuple[int,int,dict]]:
        return [
            (vertex, nbr, attr)
            for nbr, attr in instance.graph.adj[vertex].items()
            if nbr != vertex
        ]

    def _filter_unvisited(
        self,
        neighbors: List[Tuple[int,int,dict]],
        visited: set
    ) -> List[Tuple[int,int,dict]]:
        return [edge for edge in neighbors if edge[1] not in visited]

    def _pick_next(
        self,
        neighbors: List[Tuple[int,int,dict]]
    ) -> Tuple[int,int,dict]:
        return min(neighbors, key=lambda e: e[2]["weight"])

    def _edge_to_close_cycle(
        self,
        instance: Instance,
        path_edges: List[Tuple[int,int,dict]]
    ) -> Tuple[int,int,dict]:
        first_vertex = path_edges[0][0]
        last_vertex = path_edges[-1][1]
        return (last_vertex, first_vertex, instance.graph[last_vertex][first_vertex])

    def _build_from_seed(
        self,
        instance: Instance,
        k_factor: KFactor,
        has_closed_cycle: bool,
        seed_edge: Tuple[int,int,dict]
    ) -> Solution:
        u, v, _ = seed_edge
        path_edges: List[Tuple[int,int,dict]] = []
        visited: set = set()

        # Step 0: create a Solution shell (path_edges empty for now)
        solution = Solution(
            instance=instance,
            model=self,
            k_factor=k_factor,
            has_closed_cycle=has_closed_cycle,
            path_edges=path_edges
        )
        visited.add(u)
        current = u

        while len(visited) < solution.k_size:
            neighbors = self._neighbors_of(instance, current)
            unvisited = self._filter_unvisited(neighbors, visited)

            if not unvisited:
                break

            next_edge = self._pick_next(unvisited)
            path_edges.append(next_edge)
            visited.add(next_edge[1])
            current = next_edge[1]

        if has_closed_cycle:
            closing_edge = self._edge_to_close_cycle(instance, path_edges)
            path_edges.append(closing_edge)

        solution.evaluate_edge_path_length()
        solution.get_path_vertices()
        return solution

    def generate_solution(
        self,
        instance: Instance,
        k_factor: KFactor,
        has_closed_cycle: bool,
        n_solution: int = 0
    ) -> Solution:
        all_seeds = self._sort_undirected_edges(instance=instance)
        idx = n_solution % len(all_seeds)
        seed_edge = all_seeds[idx]

        return self._build_from_seed(
            instance=instance,
            k_factor=k_factor,
            has_closed_cycle=has_closed_cycle,
            seed_edge=seed_edge
        )

    @timeit
    def generate_multiple_solutions(
        self,
        instance: Instance,
        k_factor: KFactor,
        has_closed_cycle: bool,
        n_solutions: int
    ) -> List[Solution]:
        """
        Uses a global-edge lower bound to prune. For each seed (u,v):
        LB(u,v) = w_uv
                + sum of the (k_size - 2) globally smallest edge-weights
                + (if closing cycle: the single smallest global edge for return).

        This ensures we never prune a seed whose best possible cost < worst_in_heap,
        so the top-n_solutions distinct tours match the paper exactly.
        """

        # 1) Sort all undirected edges (u<v) by ascending weight. These are our seeds.
        all_seeds = self._sort_undirected_edges(instance=instance)
        total_seeds = len(all_seeds)

        # 2) Build a flat list of all edge-weights in the entire graph (undirected):
        #    We’ll use this to form global_sorted_weights once.
        #    Since instance.graph is undirected, each (u,v) appears once in all_seeds.
        global_weights = [data["weight"] for _, _, data in all_seeds]
        global_weights.sort()  # length = total_seeds

        # 3) Precompute a partial prefix-sums for the first (k_size - 2) global edges:
        n_vertices = instance.number_of_vertices
        k_size = int(k_factor.value * n_vertices)

        # We need exactly (k_size - 1) edges to form a path (or k_size to form a cycle).
        # When building a tour from a seed-edge (u->v), that seed already counts as 1 edge.
        # So we need (k_size - 2) additional edges if not closing, or (k_size - 2) + 1 if closing.
        need_middle = max(0, k_size - 2)
        # But if k_size < 2, need_middle=0. (Edge-case: trivial small k.)

        # Make a prefix sum array to get sum of first “need_middle” weights in O(1)
        prefix_sums = [0] * (len(global_weights) + 1)
        for i, w in enumerate(global_weights, start=1):
            prefix_sums[i] = prefix_sums[i - 1] + w

        # 4) Also keep track of the single smallest global edge (for a closing edge if needed)
        min_global_edge = global_weights[0]  # guaranteed exists if n_vertices >= 2

        # 5) Prepare a max-heap of size n_solutions. Tuple format: (−cost, tie_breaker, Solution)
        heap: List[Tuple[float,int,Solution]] = []
        seen_paths = set()  # for deduplication by vertex sequence
        lock = Lock()
        counter = 0  # simple integer for tie-breaking

        # 6) This inner function builds (or prunes) one seed:
        def build(edge, idx):
            nonlocal counter
            if idx % (total_seeds // 100 or 1) == 0:
                logger.info(f"[NNv2] Processing {instance.name} {k_factor} closed={has_closed_cycle} - seed {idx+1}/{total_seeds}...")

            u, v, data_uv = edge
            w_uv = data_uv["weight"]

            # 7) Compute global lower bound:
            #    - One edge = w_uv  (seed)
            #    - Next (k_size - 2) edges cost at least the first (k_size - 2) global_weights
            #    - If we need a closed cycle, add the very smallest global edge for the return.
            if need_middle > 0:
                if need_middle <= len(global_weights):
                    sum_middle = prefix_sums[need_middle]
                else:
                    # If need_middle > total edges, impossible to form k-tour → prune
                    sum_middle = float("inf")
            else:
                sum_middle = 0

            closing_edge_cost = min_global_edge if has_closed_cycle else 0
            lower_bound = w_uv + sum_middle + closing_edge_cost

            # 8) Early prune check (must hold heap_full first):
            with lock:
                if len(heap) >= n_solutions and lower_bound >= -heap[0][0]:
                    return  # prune this seed early

            # 9) Otherwise, build the full k-tour from (u->v):
            sol = self._build_from_seed(
                instance=instance,
                k_factor=k_factor,
                has_closed_cycle=has_closed_cycle,
                seed_edge=edge
            )
            sol.get_path_vertices()
            key = tuple(sol.path_vertices)

            # 10) Insert into heap (deduplicate + maintain top-n_solutions)
            with lock:
                if key in seen_paths:
                    return  # exact duplicate, skip
                seen_paths.add(key)

                entry = (-sol.path_length, counter, sol)
                counter += 1

                if len(heap) < n_solutions:
                    heappush(heap, entry)
                else:
                    # If this solution is strictly better than current worst-in-heap
                    if sol.path_length < -heap[0][0]:
                        heappushpop(heap, entry)

        # 11) Launch all seeds in parallel (threads share in‐memory data structures)
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(build)(edge, idx) for idx, edge in enumerate(all_seeds)
        )

        logger.info(f"[NNv2] Pruned {total_seeds - len(seen_paths) - len(heap)} seeds; "
            f"Built {len(seen_paths)} distinct solutions so far.")

        # 12) Extract the top-n_solutions from the heap and sort by actual path_length
        top_k = [entry[2] for entry in nlargest(n_solutions, heap)]
        top_k.sort(key=lambda s: s.path_length)
        return top_k

    # @timeit
    # def generate_multiple_solutions(
    #     self,
    #     instance: Instance,
    #     k_factor: KFactor,
    #     has_closed_cycle: bool,
    #     n_solutions: int
    # ) -> List[Solution]:
    #     all_seeds = self._sort_undirected_edges(instance=instance)
    #     total = len(all_seeds)
    #     log_every = max(total // 100, 1)

    #     def build(edge, idx):
    #         if idx % log_every == 0:
    #             logger.info(f"[NNv2] Building solution {idx+1}/{total}...")
    #         return self._build_from_seed(
    #             instance=instance,
    #             k_factor=k_factor,
    #             has_closed_cycle=has_closed_cycle,
    #             seed_edge=edge
    #         )

    #     candidates = Parallel(n_jobs=-1, prefer="threads")(
    #         delayed(build)(edge, i) for i, edge in enumerate(all_seeds)
    #     )

    #     # deduplicate by path
    #     seen = set()
    #     unique_solutions = []
    #     for sol in candidates:
    #         key = tuple(sol.path_vertices)
    #         if key in seen:
    #             continue
    #         seen.add(key)
    #         unique_solutions.append(sol)

    #     candidates.sort(key=lambda sol: sol.path_length)
    #     return candidates[:n_solutions]


# from typing import List, Tuple
# from functools import lru_cache
# from dataclasses import dataclass, field
# from joblib import Parallel, delayed
# import heapq

# from k_tsp_solver import (
#     Instance,
#     Solution,
#     Model,
#     ModelName,
#     KFactor,
#     timeit,
#     logger
# )


# @dataclass(unsafe_hash=True)
# class NearestNeighborsV2(Model):
#     """
#     A “paper‐style” nearest‐neighbor initializer that:
#       1) Sorts all undirected edges (u<v) by weight.
#       2) For each edge, builds exactly one k‐tour by marking only one endpoint first,
#          then repeatedly adding the nearest unvisited neighbor until k vertices are reached.
#       3) Optionally closes the cycle at the end.
#       4) To generate the initial GA population of size P, it creates all (n(n-1)/2) candidates,
#          then returns the top P—but uses a max‐heap with a tie-breaker and lower‐bound pruning
#          so we never build tours that cannot enter the top‐P.
#     """

#     name: str = str(ModelName.NEAREST_NEIGHBORS_V2.value)

#     @lru_cache
#     def _sort_undirected_edges(self, instance: Instance) -> List[Tuple[int,int,dict]]:
#         edges: List[Tuple[int,int,dict]] = []
#         for u, v, data in instance.graph.edges(data=True):
#             if u < v:
#                 edges.append((u, v, data))
#         edges.sort(key=lambda x: x[2]["weight"])
#         return edges

#     @lru_cache
#     def _neighbors_of(self, instance: Instance, vertex: int) -> List[Tuple[int,int,dict]]:
#         return [
#             (vertex, nbr, attr)
#             for nbr, attr in instance.graph.adj[vertex].items()
#             if nbr != vertex
#         ]

#     def _filter_unvisited(
#         self,
#         neighbors: List[Tuple[int,int,dict]],
#         visited: set
#     ) -> List[Tuple[int,int,dict]]:
#         return [edge for edge in neighbors if edge[1] not in visited]

#     def _pick_next(
#         self,
#         neighbors: List[Tuple[int,int,dict]]
#     ) -> Tuple[int,int,dict]:
#         return min(neighbors, key=lambda e: e[2]["weight"])

#     def _edge_to_close_cycle(
#         self,
#         instance: Instance,
#         path_edges: List[Tuple[int,int,dict]]
#     ) -> Tuple[int,int,dict]:
#         first_vertex = path_edges[0][0]
#         last_vertex = path_edges[-1][1]
#         return (last_vertex, first_vertex, instance.graph[last_vertex][first_vertex])

#     def _build_from_seed(
#         self,
#         instance: Instance,
#         k_factor: KFactor,
#         has_closed_cycle: bool,
#         seed_edge: Tuple[int,int,dict]
#     ) -> Solution:
#         """
#         Actually constructs one k-city tour from the given seed_edge = (u,v,data)
#         following the paper’s procedure. Uses Solution.k_size to know how many vertices.
#         """
#         u, v, _ = seed_edge
#         path_edges: List[Tuple[int,int,dict]] = []
#         visited: set = set()

#         # Step 0: create a Solution shell (path_edges empty for now)
#         solution = Solution(
#             instance=instance,
#             model=self,
#             k_factor=k_factor,
#             has_closed_cycle=has_closed_cycle,
#             path_edges=path_edges
#         )
#         k_size = solution.k_size  # number of vertices we need

#         # 1) Mark only u visited; current = u
#         visited.add(u)
#         current = u

#         # 2) Expand until we have k_size distinct vertices
#         while len(visited) < k_size:
#             neighbors = self._neighbors_of(instance, current)
#             unvisited = self._filter_unvisited(neighbors, visited)

#             if not unvisited:
#                 # In a complete graph this shouldn’t happen, but just in case.
#                 break

#             next_edge = self._pick_next(unvisited)
#             path_edges.append(next_edge)

#             # Mark the new endpoint visited, move current
#             visited.add(next_edge[1])
#             current = next_edge[1]

#         # 3) If closed cycle is desired, add the closing edge
#         if has_closed_cycle:
#             closing_edge = self._edge_to_close_cycle(instance, path_edges)
#             path_edges.append(closing_edge)

#         # 4) Compute cost & vertices
#         solution.evaluate_edge_path_length()  # sets solution.path_length
#         solution.get_path_vertices()
#         return solution

#     def generate_solution(
#         self,
#         instance: Instance,
#         k_factor: KFactor,
#         has_closed_cycle: bool,
#         n_solution: int = 0
#     ) -> Solution:
#         """
#         Return exactly one k-tour by using the n_solution-th undirected edge (in sorted order)
#         as the seed. This keeps the original signature so existing GA code won't break.
#         """
#         all_seeds = self._sort_undirected_edges(instance=instance)
#         idx = n_solution % len(all_seeds)
#         seed_edge = all_seeds[idx]
#         return self._build_from_seed(
#             instance=instance,
#             k_factor=k_factor,
#             has_closed_cycle=has_closed_cycle,
#             seed_edge=seed_edge
#         )

#     @timeit
#     def generate_multiple_solutions(
#         self,
#         instance: Instance,
#         k_factor: KFactor,
#         has_closed_cycle: bool,
#         n_solutions: int,
#         n_jobs: int = -1
#     ) -> List[Solution]:
#         """
#         Parallelized and optimized version of NNv2:
#         • Uses a max-heap to keep top-n_solutions.
#         • Skips seeds that can't possibly beat the current worst.
#         • Parallelizes solution construction across seeds using joblib.
#         """

#         all_seeds = self._sort_undirected_edges(instance=instance)
#         total_seeds = len(all_seeds)

#         dummy = Solution(instance=instance, model=self, k_factor=k_factor, has_closed_cycle=has_closed_cycle, path_edges=[])
#         k_size = dummy.k_size
#         global_min = all_seeds[0][2]["weight"]

#         # Prepare a list of seed edges that pass the lower-bound check
#         def should_keep(seed_edge):
#             w_seed = seed_edge[2]["weight"]
#             lb = w_seed + (k_size - 1) * global_min
#             return lb

#         # Pre-filter seeds for parallel processing
#         filtered_seeds: list = []
#         worst_known_cost = float("inf")  # dynamically updated later
#         for seed_edge in all_seeds:
#             lb = should_keep(seed_edge)
#             if len(filtered_seeds) < n_solutions or lb < worst_known_cost:
#                 filtered_seeds.append(seed_edge)
#                 if len(filtered_seeds) == n_solutions:
#                     # Only update worst_known_cost once we fill enough
#                     worst_known_cost = lb

#         # logger.info(f"[NNv2] Parallelizing over {len(filtered_seeds)} filtered seeds.")

#         # Parallel build of solutions
#         def build(seed_edge):
#             sol = self._build_from_seed(instance, k_factor, has_closed_cycle, seed_edge)
#             key = tuple(sol.path_vertices)
#             return (sol.path_length, key, sol)

#         results = Parallel(n_jobs=n_jobs, prefer="threads")(
#             delayed(build)(seed_edge) for seed_edge in filtered_seeds
#         )

#         # Deduplicate and push into a max-heap 
#         seen_routes = set()
#         heap: List[Tuple[float, int, Solution]] = []
#         counter = 0

#         for cost, key, sol in results:
#             if key in seen_routes:
#                 continue
#             seen_routes.add(key)

#             entry = (-cost, counter, sol)
#             counter += 1

#             if len(heap) < n_solutions:
#                 heapq.heappush(heap, entry)
#             else:
#                 if cost < -heap[0][0]:
#                     heapq.heapreplace(heap, entry)

#         top_solutions = [entry[2] for entry in heap]
#         top_solutions.sort(key=lambda s: s.path_length)
#         return top_solutions