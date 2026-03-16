import numpy as np
import heapq


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

