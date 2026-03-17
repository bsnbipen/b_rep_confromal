import numpy as np
import heapq
import trimesh
import math


def build_patch_graph(patch_mesh):
    """
    Build an undirected weighted graph from a triangle patch mesh.

    Graph interpretation
    --------------------
    - mesh vertex  -> graph node
    - unique mesh edge -> graph edge
    - Euclidean edge length -> graph weight

    Parameters
    ----------
    patch_mesh : trimesh.Trimesh
        Connected patch mesh.

    Returns
    -------
    graph : dict
        {
            "vertices": (N,3) float array,
            "faces": (M,3) int array,
            "edges": (E,2) int array,
            "edge_lengths": (E,) float array,
            "adjacency": list of length N,
                where adjacency[i] = [(j, w_ij), ...]
        }
    """
    if patch_mesh is None or len(patch_mesh.vertices) == 0 or len(patch_mesh.faces) == 0:
        raise ValueError("patch_mesh is empty")

    vertices = np.asarray(patch_mesh.vertices, dtype=float)
    faces = np.asarray(patch_mesh.faces, dtype=np.int32)

    # Unique undirected mesh edges from trimesh
    edges = np.asarray(patch_mesh.edges_unique, dtype=np.int32)

    if len(edges) == 0:
        raise ValueError("patch_mesh has no edges")

    # Edge lengths = graph weights
    edge_vecs = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    edge_lengths = np.linalg.norm(edge_vecs, axis=1)

    # Build adjacency list
    n_vertices = len(vertices)
    adjacency = [[] for _ in range(n_vertices)]

    for (i, j), w in zip(edges, edge_lengths):
        adjacency[i].append((int(j), float(w)))
        adjacency[j].append((int(i), float(w)))

    graph = {
        "vertices": vertices,
        "faces": faces,
        "edges": edges,
        "edge_lengths": edge_lengths,
        "adjacency": adjacency,
    }

    print("\n[build_patch_graph] =====================")
    print("vertices     :", len(vertices))
    print("faces        :", len(faces))
    print("unique edges :", len(edges))

    deg = np.array([len(nbrs) for nbrs in adjacency], dtype=int)
    print("min degree   :", int(deg.min()))
    print("max degree   :", int(deg.max()))
    print("mean degree  :", float(deg.mean()))

    return graph


def map_seed_loop_to_source_vertices(seed_loop, patch_mesh):
    """
    Map a seed perimeter loop to a set of nearest patch-mesh vertex IDs.

    This is the first modular approximation for building a Dijkstra source set.

    Parameters
    ----------
    seed_loop : dict
        One seed perimeter dict with key "points", shape (N,3).
    patch_mesh : trimesh.Trimesh
        Patch mesh on which the seed loop lies.

    Returns
    -------
    source_vertex_ids : (K,) int array
        Unique patch vertex IDs nearest to the seed loop points.
    """
    if patch_mesh is None or len(patch_mesh.vertices) == 0:
        raise ValueError("patch_mesh is empty")

    if seed_loop is None or "points" not in seed_loop:
        raise ValueError("seed_loop must contain key 'points'")

    seed_pts = np.asarray(seed_loop["points"], dtype=float)
    patch_vertices = np.asarray(patch_mesh.vertices, dtype=float)

    if len(seed_pts) == 0:
        raise ValueError("seed_loop has no points")

    # nearest patch vertex for each seed point
    _, nearest_vids = patch_mesh.kdtree.query(seed_pts)
    nearest_vids = np.asarray(nearest_vids, dtype=np.int32)

    # unique source vertices
    source_vertex_ids = np.unique(nearest_vids)

    print("\n[map_seed_loop_to_source_vertices] =====")
    print("seed points            :", len(seed_pts))
    print("unique source vertices :", len(source_vertex_ids))

    return source_vertex_ids


def remesh_patch_to_edge_length(
    patch_mesh,
    target_edge_length,
    max_iter=8,
    max_retry=3,
    relax_factor=1.5,
):
    """
    Refine a patch mesh so no edge is longer than target_edge_length.

    Uses adaptive retries if trimesh.subdivide_to_size hits max_iter.

    Parameters
    ----------
    patch_mesh : trimesh.Trimesh
    target_edge_length : float
        Desired maximum mesh edge length.
    max_iter : int
        Initial subdivision iteration cap.
    max_retry : int
        Number of adaptive retries.
    relax_factor : float
        If repeated failures happen, relax target edge length by this factor.

    Returns
    -------
    remeshed_patch : trimesh.Trimesh
    """
    if patch_mesh is None or len(patch_mesh.vertices) == 0 or len(patch_mesh.faces) == 0:
        raise ValueError("patch_mesh is empty")

    if target_edge_length is None or target_edge_length <= 0:
        raise ValueError("target_edge_length must be > 0")

    v = np.asarray(patch_mesh.vertices, dtype=float)
    f = np.asarray(patch_mesh.faces, dtype=np.int32)

    # measure current max edge
    edges = np.asarray(patch_mesh.edges_unique, dtype=np.int32)
    edge_vecs = v[edges[:, 1]] - v[edges[:, 0]]
    edge_lens = np.linalg.norm(edge_vecs, axis=1)

    current_max_edge = float(np.max(edge_lens))
    current_mean_edge = float(np.mean(edge_lens))

    print("\n[remesh_patch_to_edge_length] =========")
    print("input vertices   :", len(v))
    print("input faces      :", len(f))
    print("current max edge :", current_max_edge)
    print("current mean edge:", current_mean_edge)
    print("target edge len  :", float(target_edge_length))

    # estimated required number of subdivision rounds
    # each subdivision roughly halves long edges
    if current_max_edge > target_edge_length:
        est_iter = int(math.ceil(math.log(current_max_edge / target_edge_length, 2))) + 2
    else:
        est_iter = 1

    trial_max_iter = max(max_iter, est_iter)
    trial_target = float(target_edge_length)

    last_err = None

    for attempt in range(max_retry):
        print(f"[remesh] attempt={attempt + 1}, target={trial_target:.6f}, max_iter={trial_max_iter}")

        try:
            v_new, f_new = trimesh.remesh.subdivide_to_size(
                vertices=v,
                faces=f,
                max_edge=trial_target,
                max_iter=trial_max_iter,
            )

            remeshed_patch = trimesh.Trimesh(
                vertices=v_new,
                faces=f_new,
                process=True
            )

            edges_new = np.asarray(remeshed_patch.edges_unique, dtype=np.int32)
            edge_vecs_new = remeshed_patch.vertices[edges_new[:, 1]] - remeshed_patch.vertices[edges_new[:, 0]]
            edge_lens_new = np.linalg.norm(edge_vecs_new, axis=1)

            print("output vertices  :", len(remeshed_patch.vertices))
            print("output faces     :", len(remeshed_patch.faces))
            print("output max edge  :", float(np.max(edge_lens_new)))
            print("output mean edge :", float(np.mean(edge_lens_new)))

            return remeshed_patch

        except ValueError as e:
            last_err = e

            if "max_iter exceeded" not in str(e):
                raise

            print("[remesh] max_iter exceeded, retrying adaptively...")

            # first increase iteration budget
            trial_max_iter = int(trial_max_iter * 1.75) + 2

            # after first failure, also relax target slightly
            if attempt >= 1:
                trial_target *= relax_factor

    raise ValueError(
        f"Adaptive remesh failed after {max_retry} attempts. "
        f"Last target={trial_target}, last max_iter={trial_max_iter}. "
        f"Original error: {last_err}"
    )


def compute_distance_field_dijkstra(
    graph,
    source_vertex_ids=None,
    initial_distance=None,
):
    """
    Compute shortest-path distance on the patch graph.

    Two usage modes
    ---------------
    1) Classic multi-source Dijkstra:
       - pass source_vertex_ids
       - those vertices start with distance 0

    2) General initialized Dijkstra:
       - pass initial_distance as an (N,) array
       - any finite entries are used as initial labels
       - useful later for better perimeter-source approximations

    Parameters
    ----------
    graph : dict
        Output from build_patch_graph(...), must contain "adjacency".
    source_vertex_ids : (K,) int array-like or None
        Patch vertex IDs used as zero-distance Dijkstra sources.
    initial_distance : (N,) float array or None
        Optional initial distance at each graph vertex.
        Finite values are treated as seeded labels, inf means uninitialized.

    Returns
    -------
    vertex_distance : (N,) float array
        Shortest graph distance field on patch vertices.
    """
    if graph is None or "adjacency" not in graph:
        raise ValueError("graph is invalid or missing adjacency")

    adjacency = graph["adjacency"]
    n_vertices = len(adjacency)

    if initial_distance is not None:
        vertex_distance = np.asarray(initial_distance, dtype=float).reshape(-1)
        if len(vertex_distance) != n_vertices:
            raise ValueError("initial_distance length does not match graph size")
    else:
        vertex_distance = np.full(n_vertices, np.inf, dtype=float)

    # If source vertices are provided, stamp them to zero
    if source_vertex_ids is not None:
        source_vertex_ids = np.asarray(source_vertex_ids, dtype=np.int32).reshape(-1)

        valid_sources = source_vertex_ids[
            (source_vertex_ids >= 0) & (source_vertex_ids < n_vertices)
        ]

        if len(valid_sources) > 0:
            valid_sources = np.unique(valid_sources)
            vertex_distance[valid_sources] = 0.0
        else:
            valid_sources = np.array([], dtype=np.int32)
    else:
        valid_sources = np.array([], dtype=np.int32)

    # Priority queue starts from every finite initialized vertex
    finite_mask = np.isfinite(vertex_distance)
    if not np.any(finite_mask):
        raise ValueError("No valid Dijkstra sources or initialized distances found")

    pq = []
    for vid in np.where(finite_mask)[0]:
        heapq.heappush(pq, (float(vertex_distance[vid]), int(vid)))

    visited = np.zeros(n_vertices, dtype=bool)

    while pq:
        d_curr, u = heapq.heappop(pq)

        if visited[u]:
            continue
        visited[u] = True

        if d_curr > vertex_distance[u]:
            continue

        for v, w in adjacency[u]:
            new_d = d_curr + w
            if new_d < vertex_distance[v]:
                vertex_distance[v] = new_d
                heapq.heappush(pq, (new_d, int(v)))

    finite_out = np.isfinite(vertex_distance)

    print("\n[compute_distance_field_dijkstra] =====")
    print("num vertices        :", n_vertices)
    print("num zero-sources    :", len(valid_sources))
    print("num finite seeds    :", int(np.sum(finite_mask)))
    print("reachable verts     :", int(np.sum(finite_out)))
    print("min dist            :", float(np.min(vertex_distance[finite_out])))
    print("max dist            :", float(np.max(vertex_distance[finite_out])))

    return vertex_distance

