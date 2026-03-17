import numpy as np
from perimeter_gen import *
from outline import *




def _point_to_segment_distances(points, a, b, eps=1e-12):
    """
    Distance from many 3D points to one 3D line segment [a,b].
    Robust against degenerate or non-finite segments.
    """
    points = np.asarray(points, dtype=float)
    a = np.asarray(a, dtype=float).reshape(3)
    b = np.asarray(b, dtype=float).reshape(3)

    # Reject invalid segment endpoints
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return np.full(len(points), np.inf, dtype=float)

    ab = b - a
    ab2 = float(np.dot(ab, ab))

    # Degenerate segment -> treat as point source
    if (not np.isfinite(ab2)) or ab2 < eps:
        diff = points - a[None, :]
        bad = ~np.all(np.isfinite(diff), axis=1)
        d = np.linalg.norm(np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
        d[bad] = np.inf
        return d

    ap = points - a[None, :]

    # Dot product per point
    numer = np.einsum("ij,j->i", ap, ab)

    # Guard against non-finite numerators
    numer = np.where(np.isfinite(numer), numer, np.inf)

    t = numer / ab2
    t = np.clip(t, 0.0, 1.0)

    proj = a[None, :] + t[:, None] * ab[None, :]
    diff = points - proj

    bad = ~np.all(np.isfinite(diff), axis=1)
    d = np.linalg.norm(np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0), axis=1)
    d[bad] = np.inf
    return d

def build_initial_distance_to_seed_segments(seed_loop, patch_mesh, dedup_tol=1e-10):
    """
    Build an initial scalar field at patch vertices based on Euclidean
    distance to the seed perimeter segments.

    Cleans the seed loop first:
    - removes non-finite points
    - removes consecutive duplicates
    """
    if patch_mesh is None or len(patch_mesh.vertices) == 0:
        raise ValueError("patch_mesh is empty")

    if seed_loop is None or "points" not in seed_loop:
        raise ValueError("seed_loop must contain key 'points'")

    verts = np.asarray(patch_mesh.vertices, dtype=float)
    pts = np.asarray(seed_loop["points"], dtype=float)

    if len(pts) < 2:
        raise ValueError("seed_loop has too few points")

    # Keep only finite points
    finite_mask = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite_mask]

    if len(pts) < 2:
        raise ValueError("seed_loop has too few finite points")

    is_closed = bool(seed_loop.get("is_closed", True))

    # Remove duplicate closure point temporarily
    if is_closed and np.linalg.norm(pts[0] - pts[-1]) < dedup_tol:
        pts = pts[:-1]

    if len(pts) < 2:
        raise ValueError("seed_loop has too few unique points")

    # Remove consecutive duplicates / near-duplicates
    cleaned = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - cleaned[-1]) > dedup_tol:
            cleaned.append(p)

    pts_work = np.asarray(cleaned, dtype=float)

    # If closed, also avoid first-last duplicate after cleaning
    if is_closed and len(pts_work) > 1 and np.linalg.norm(pts_work[0] - pts_work[-1]) < dedup_tol:
        pts_work = pts_work[:-1]

    if len(pts_work) < 2:
        raise ValueError("seed_loop collapsed after cleaning")

    # Build segment list
    if is_closed:
        seg_starts = pts_work
        seg_ends = np.roll(pts_work, -1, axis=0)
    else:
        seg_starts = pts_work[:-1]
        seg_ends = pts_work[1:]

    initial_distance = np.full(len(verts), np.inf, dtype=float)

    kept_segments = 0
    skipped_segments = 0

    for a, b in zip(seg_starts, seg_ends):
        if (not np.all(np.isfinite(a))) or (not np.all(np.isfinite(b))):
            skipped_segments += 1
            continue

        if np.linalg.norm(b - a) <= dedup_tol:
            skipped_segments += 1
            continue

        d = _point_to_segment_distances(verts, a, b)
        initial_distance = np.minimum(initial_distance, d)
        kept_segments += 1

    print("\n[build_initial_distance_to_seed_segments] ===")
    print("patch vertices      :", len(verts))
    print("seed points raw     :", len(seed_loop['points']))
    print("seed points cleaned :", len(pts_work))
    print("segments kept       :", kept_segments)
    print("segments skipped    :", skipped_segments)
    print("min init dist       :", float(np.min(initial_distance[np.isfinite(initial_distance)])))
    print("max init dist       :", float(np.max(initial_distance[np.isfinite(initial_distance)])))

    return initial_distance

def _interp_iso_point_on_edge(p0, p1, d0, d1, iso_value, eps=1e-12):
    """
    Interpolate iso-crossing point on one edge.
    """
    if not np.isfinite(d0) or not np.isfinite(d1):
        return None

    if abs(d1 - d0) < eps:
        return None

    t = (iso_value - d0) / (d1 - d0)

    if t < -eps or t > 1.0 + eps:
        return None

    t = np.clip(t, 0.0, 1.0)
    return p0 + t * (p1 - p0)


def _deduplicate_points(points, tol=1e-9):
    """
    Remove nearly duplicate points from a small point list.
    """
    unique = []
    for p in points:
        if not any(np.linalg.norm(p - q) <= tol for q in unique):
            unique.append(p)
    return unique


def _stitch_iso_segments(segments, merge_tol=1e-6):
    """
    Stitch unordered contour segments into ordered polylines/loops.

    Parameters
    ----------
    segments : list of tuple((3,), (3,))
        Unordered contour segments.
    merge_tol : float
        Tolerance used to merge segment endpoints.

    Returns
    -------
    loops : list of (N,3) arrays
        Ordered contour polylines/loops.
    """
    if len(segments) == 0:
        return []

    def key_of_point(p):
        return tuple(np.round(np.asarray(p) / merge_tol).astype(np.int64))

    nodes = {}
    adjacency = {}

    def add_node(p):
        k = key_of_point(p)
        if k not in nodes:
            nodes[k] = np.asarray(p, dtype=float)
            adjacency[k] = set()
        return k

    for p0, p1 in segments:
        k0 = add_node(p0)
        k1 = add_node(p1)

        if k0 == k1:
            continue

        adjacency[k0].add(k1)
        adjacency[k1].add(k0)

    unvisited_edges = set()
    for k0, nbrs in adjacency.items():
        for k1 in nbrs:
            edge = tuple(sorted((k0, k1)))
            unvisited_edges.add(edge)

    loops = []

    def edge_key(a, b):
        return tuple(sorted((a, b)))

    def walk_chain(start):
        """
        Walk one chain/loop from start using unvisited edges.
        """
        path = [start]
        prev = None
        curr = start

        while True:
            candidates = [
                nb for nb in adjacency[curr]
                if edge_key(curr, nb) in unvisited_edges and nb != prev
            ]

            if len(candidates) == 0:
                break

            nxt = candidates[0]
            unvisited_edges.remove(edge_key(curr, nxt))
            path.append(nxt)

            prev, curr = curr, nxt

            if curr == start:
                break

        return path

    # First walk open chains (degree 1)
    open_starts = [k for k, nbrs in adjacency.items() if len(nbrs) == 1]

    for s in open_starts:
        while True:
            has_unvisited = any(edge_key(s, nb) in unvisited_edges for nb in adjacency[s])
            if not has_unvisited:
                break

            path_keys = walk_chain(s)
            if len(path_keys) >= 2:
                pts = np.asarray([nodes[k] for k in path_keys], dtype=float)
                loops.append(pts)

    # Then walk closed loops
    while len(unvisited_edges) > 0:
        some_edge = next(iter(unvisited_edges))
        start = some_edge[0]

        path_keys = walk_chain(start)
        if len(path_keys) >= 2:
            pts = np.asarray([nodes[k] for k in path_keys], dtype=float)

            # close if loop-like
            if np.linalg.norm(pts[0] - pts[-1]) > merge_tol:
                pts = np.vstack([pts, pts[0]])

            loops.append(pts)

    return loops


def extract_isocontours_from_scalar_field(
    patch_mesh,
    vertex_distance,
    iso_value,
    merge_tol=1e-6,
    edge_eps=1e-12,
):
    """
    Extract iso-contours from a scalar field defined at patch vertices.

    Parameters
    ----------
    patch_mesh : trimesh.Trimesh
        Patch mesh.
    vertex_distance : (N,) array
        Scalar distance value at each patch vertex.
    iso_value : float
        Target contour value.
    merge_tol : float
        Tolerance for stitching segment endpoints.
    edge_eps : float
        Small tolerance for edge interpolation checks.

    Returns
    -------
    loops : list of (K,3) arrays
        Ordered iso-contour loops/polylines in 3D.
    """
    if patch_mesh is None or len(patch_mesh.vertices) == 0 or len(patch_mesh.faces) == 0:
        raise ValueError("patch_mesh is empty")

    vertices = np.asarray(patch_mesh.vertices, dtype=float)
    faces = np.asarray(patch_mesh.faces, dtype=np.int32)
    vertex_distance = np.asarray(vertex_distance, dtype=float).reshape(-1)

    if len(vertex_distance) != len(vertices):
        raise ValueError("vertex_distance length does not match patch vertices")

    segments = []

    tri_edge_ids = [(0, 1), (1, 2), (2, 0)]

    for tri in faces:
        tri_pts = vertices[tri]            # (3,3)
        tri_vals = vertex_distance[tri]    # (3,)

        if not np.all(np.isfinite(tri_vals)):
            continue

        crossings = []

        for a, b in tri_edge_ids:
            p0 = tri_pts[a]
            p1 = tri_pts[b]
            d0 = tri_vals[a]
            d1 = tri_vals[b]

            # Entire edge exactly on iso -> ambiguous; skip for now
            if abs(d0 - iso_value) < edge_eps and abs(d1 - iso_value) < edge_eps:
                continue

            # Standard crossing test
            if (d0 - iso_value) * (d1 - iso_value) <= 0.0:
                p_iso = _interp_iso_point_on_edge(p0, p1, d0, d1, iso_value, eps=edge_eps)
                if p_iso is not None:
                    crossings.append(p_iso)

        crossings = _deduplicate_points(crossings, tol=merge_tol)

        # In a well-behaved triangle contour cut, we expect 2 crossings
        if len(crossings) == 2:
            segments.append((crossings[0], crossings[1]))

    loops = _stitch_iso_segments(segments, merge_tol=merge_tol)

    print("\n[extract_isocontours_from_scalar_field] =====")
    print("iso_value         :", float(iso_value))
    print("raw segments      :", len(segments))
    print("stitched contours :", len(loops))
    for i, lp in enumerate(loops):
        print(f"  contour {i}: npts={len(lp)}")

    return loops

def package_contour_as_toolpath(
    contour_points,
    patch_mesh,
    ring_id,
    family_id=0,
    closed_tol=1e-3,
    point_spacing=None,
):
    """
    Convert one extracted contour loop/polyline into a slicer toolpath dict.

    Parameters
    ----------
    contour_points : (N,3) array
        Ordered contour points from iso-contour extraction.
    patch_mesh : trimesh.Trimesh
        Patch mesh on which the contour lies.
    ring_id : int
        Perimeter index (0 = seed perimeter, 1 = first inward, etc.).
    family_id : int
        Loop family/group index.
    closed_tol : float
        Distance tolerance for deciding whether the contour is closed.
    point_spacing : float or None
        Optional resampling spacing. If None, keeps input sampling.

    Returns
    -------
    path_data : dict or None
        Standard perimeter toolpath dict.
    """
    if contour_points is None:
        return None

    pts = np.asarray(contour_points, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        print("[package_contour_as_toolpath] FAIL: invalid contour_points")
        return None

    if patch_mesh is None or len(patch_mesh.faces) == 0:
        print("[package_contour_as_toolpath] FAIL: empty patch mesh")
        return None

    is_closed = np.linalg.norm(pts[0] - pts[-1]) <= closed_tol

    # Optional resampling
    if point_spacing is not None and point_spacing > 0:
        pts = resample_polyline_uniform(
            pts,
            spacing=point_spacing,
            closed=is_closed
        )

    # Sample normals from patch
    normals, valid = sample_barycentric_normals_on_mesh(
        patch_mesh,
        pts
    )

    if np.sum(valid) < 2:
        print("[package_contour_as_toolpath] FAIL: too few valid normals")
        return None

    # Fallback nearest-vertex normals where needed
    if not np.all(valid):
        try:
            _, vertex_indices = patch_mesh.kdtree.query(pts[~valid])
            fallback_normals = patch_mesh.vertex_normals[vertex_indices]

            L = np.linalg.norm(fallback_normals, axis=1, keepdims=True)
            good = L.squeeze() > 1e-12
            fallback_normals[good] = fallback_normals[good] / L[good]

            normals[~valid] = fallback_normals
        except Exception:
            pass

    tangents, inward_dirs, side_dirs_raw = compute_loop_frames(
        pts,
        normals,
        patch_mesh=patch_mesh,
        is_closed=is_closed
    )

    path_data = {
        "points": pts,
        "normals": normals,
        "tangents": tangents,
        "inward_dirs": inward_dirs,
        "is_closed": is_closed,
        "ring_id": int(ring_id),
        "family_id": int(family_id),
    }

    print("\n[package_contour_as_toolpath] =========")
    print("ring_id     :", int(ring_id))
    print("family_id   :", int(family_id))
    print("npts        :", len(pts))
    print("is_closed   :", is_closed)

    return path_data


def generate_inner_perimeters_dijkstra(
    patch_mesh,
    seed_loop,
    perimeter_spacing,
    n_inner_perimeters,
    point_spacing=0.5,
    family_id=0,
    merge_tol=1e-6,
):
    """
    Generate inner perimeter toolpaths from one seed loop using:
    patch graph -> Dijkstra distance field -> iso-contours.

    Parameters
    ----------
    patch_mesh : trimesh.Trimesh
        Patch mesh.
    seed_loop : dict
        Seed perimeter dict (ring 0).
    perimeter_spacing : float
        Desired spacing between adjacent perimeters.
    n_inner_perimeters : int
        Number of inner perimeters to generate.
        Example:
            0 -> return []
            1 -> generate ring 1 only
            2 -> generate ring 1 and ring 2
    point_spacing : float
        Resampling spacing for packaged contour toolpaths.
    family_id : int
        Family/group index.
    merge_tol : float
        Segment stitching tolerance for iso-contours.

    Returns
    -------
    inner_toolpaths : list of dict
        Packaged inner perimeter toolpaths with ring_id = 1..n_inner_perimeters
    vertex_distance : (N,) float array
        Dijkstra scalar field on patch vertices.
    """
    if patch_mesh is None or len(patch_mesh.faces) == 0:
        print("[generate_inner_perimeters_dijkstra] FAIL: empty patch mesh")
        return [], None

    if seed_loop is None or "points" not in seed_loop:
        print("[generate_inner_perimeters_dijkstra] FAIL: invalid seed_loop")
        return [], None

    if perimeter_spacing <= 0:
        print("[generate_inner_perimeters_dijkstra] FAIL: perimeter_spacing must be > 0")
        return [], None

    if n_inner_perimeters <= 0:
        print("[generate_inner_perimeters_dijkstra] no inner perimeters requested")
        return [], None

    print("\n[generate_inner_perimeters_dijkstra] =====")
    print("perimeter_spacing :", perimeter_spacing)
    print("n_inner_perimeters:", n_inner_perimeters)

    # ------------------------------------------
    # 1. Patch graph
    # ------------------------------------------
    graph = build_patch_graph(patch_mesh)

    initial_distance = build_initial_distance_to_seed_segments(
        seed_loop=seed_loop,
        patch_mesh=patch_mesh
    )

    vertex_distance = compute_distance_field_dijkstra(
        graph,
        initial_distance=initial_distance
    )

    # ------------------------------------------
    # 4. Extract iso-contours for each ring
    # ------------------------------------------
    inner_toolpaths = []

    for ring_id in range(1, int(n_inner_perimeters) + 1):
        iso_value = ring_id * perimeter_spacing

        print(f"\n[generate_inner_perimeters_dijkstra] ring_id={ring_id}, iso={iso_value}")

        loops = extract_isocontours_from_scalar_field(
            patch_mesh,
            vertex_distance,
            iso_value=iso_value,
            merge_tol=merge_tol,
        )

        if len(loops) == 0:
            print(f"[generate_inner_perimeters_dijkstra] ring_id={ring_id}: no contours found")
            continue

        for loop_idx, loop_pts in enumerate(loops):
            pd = package_contour_as_toolpath(
                contour_points=loop_pts,
                patch_mesh=patch_mesh,
                ring_id=ring_id,
                family_id=family_id,
                point_spacing=point_spacing,
            )

            if pd is not None:
                inner_toolpaths.append(pd)

    print("\n[generate_inner_perimeters_dijkstra] DONE")
    print("returned inner toolpaths:", len(inner_toolpaths))

    return inner_toolpaths, vertex_distance

def resample_polyline_uniform(
    points,
    spacing,
    closed=False,
    preserve_corners=True,
    angle_deg_threshold=15.0,
):
    """
    Resample a 3D polyline/loop to approximately uniform arc-length spacing,
    while optionally preserving sharp-corner vertices exactly.

    Parameters
    ----------
    points : (N,3) array
        Ordered polyline points.
    spacing : float
        Desired point-to-point spacing.
    closed : bool
        Whether the path is a closed loop.
    preserve_corners : bool
        If True, sharp vertices are forced into the resampled output.
    angle_deg_threshold : float
        Turning-angle threshold in degrees for corner preservation.
        Larger value = only sharper corners are preserved.

    Returns
    -------
    out : (M,3) array
        Resampled points.
    """
    pts = np.asarray(points, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        return pts.copy()

    if spacing is None or spacing <= 0:
        return pts.copy()

    # Remove duplicate closure point temporarily
    duplicate_last = closed and np.linalg.norm(pts[0] - pts[-1]) < 1e-12
    if duplicate_last:
        pts_work = pts[:-1]
    else:
        pts_work = pts.copy()

    if len(pts_work) < 2:
        return pts.copy()

    # -------------------------------------------------
    # A. Build segment list
    # -------------------------------------------------
    if closed:
        seg_starts = pts_work
        seg_ends = np.roll(pts_work, -1, axis=0)
    else:
        seg_starts = pts_work[:-1]
        seg_ends = pts_work[1:]

    seg_vecs = seg_ends - seg_starts
    seg_lens = np.linalg.norm(seg_vecs, axis=1)

    if np.all(seg_lens < 1e-12):
        return pts.copy()

    cumlen = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cumlen[-1]

    # Arc-length positions of original vertices
    if closed:
        vertex_s = cumlen[:-1]   # one value per unique vertex
    else:
        vertex_s = cumlen        # includes last endpoint

    # -------------------------------------------------
    # B. Uniform sample positions
    # -------------------------------------------------
    sample_s = np.arange(0.0, total_len, spacing)
    if len(sample_s) == 0:
        sample_s = np.array([0.0])

    if total_len - sample_s[-1] > 1e-9:
        sample_s = np.append(sample_s, total_len)

    # -------------------------------------------------
    # C. Corner detection
    # -------------------------------------------------
    if preserve_corners and len(pts_work) >= 3:
        m = len(pts_work)
        corner_mask = np.zeros(m, dtype=bool)

        for i in range(m):
            if not closed and (i == 0 or i == m - 1):
                corner_mask[i] = True
                continue

            p_prev = pts_work[(i - 1) % m]
            p_curr = pts_work[i]
            p_next = pts_work[(i + 1) % m]

            v_in = p_curr - p_prev
            v_out = p_next - p_curr

            L1 = np.linalg.norm(v_in)
            L2 = np.linalg.norm(v_out)

            if L1 < 1e-12 or L2 < 1e-12:
                continue

            v_in = v_in / L1
            v_out = v_out / L2

            # 0 deg -> straight, larger angle -> sharper turn
            dot = np.clip(np.dot(v_in, v_out), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(dot))

            if angle_deg >= angle_deg_threshold:
                corner_mask[i] = True

        corner_s = vertex_s[corner_mask]

        # Merge uniform samples + corner samples with tolerance
        all_s = np.sort(np.concatenate([sample_s, corner_s]))
        merged = [all_s[0]]
        for s in all_s[1:]:
            if abs(s - merged[-1]) > 1e-9:
                merged.append(s)
        sample_s = np.asarray(merged, dtype=float)

    # -------------------------------------------------
    # D. Interpolate output points
    # -------------------------------------------------
    out = []
    seg_idx = 0

    for s in sample_s:
        while seg_idx < len(seg_lens) - 1 and s > cumlen[seg_idx + 1]:
            seg_idx += 1

        L = seg_lens[seg_idx]
        if L < 1e-12:
            out.append(seg_starts[seg_idx].copy())
            continue

        t = (s - cumlen[seg_idx]) / L
        p = seg_starts[seg_idx] + t * seg_vecs[seg_idx]
        out.append(p)

    out = np.asarray(out, dtype=float)

    # Re-close loop if needed
    if closed:
        if np.linalg.norm(out[0] - out[-1]) > 1e-9:
            out = np.vstack([out, out[0]])

    return out