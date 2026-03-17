import trimesh
import numpy as np
from intersection_patch import *
from UV_parametrization import *
from iso_surface import generate_inner_perimeters_dijkstra
from perimeter_gen import remesh_patch_to_edge_length


def extract_first_layer_intersection_perimeter(
    layer_zero_surface,
    print_model,
    layer_height,
    gap_tol=0.10,
    euclid_tol=None,
    min_component_faces=20,
    keep_largest_only=True,
    remesh_target_edge=None,
    seed_point_spacing=1.0,
):
    patch_mesh, debug = extract_model_patch_from_layer_zero(
        layer_zero_surface=layer_zero_surface,
        print_model=print_model,
        layer_height=layer_height,
        gap_tol=gap_tol,
        euclid_tol=euclid_tol,
        min_component_faces=min_component_faces,
        keep_largest_only=keep_largest_only,
    )

    if patch_mesh is None or len(patch_mesh.faces) == 0:
        return patch_mesh, [], debug

    # NEW: refine patch before extracting seed perimeter
    if remesh_target_edge is not None and remesh_target_edge > 0:
        patch_mesh = remesh_patch_to_edge_length(
            patch_mesh,
            target_edge_length=remesh_target_edge
        )

    seed_loops = extract_ordered_perimeter_5axis(
        interface_mesh=patch_mesh,
        normal_source_mesh=patch_mesh,
        point_spacing=seed_point_spacing
    )

    return patch_mesh, seed_loops, debug



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

def sample_barycentric_normals_on_mesh(mesh, query_points):
    """
    Sample smooth normals using nearest-face barycentric interpolation
    of the mesh vertex normals.
    """
    query_points = np.asarray(query_points, dtype=float)

    if mesh is None or len(mesh.faces) == 0 or len(query_points) == 0:
        return np.zeros_like(query_points), np.zeros(len(query_points), dtype=bool)

    closest_pts, distances, face_ids = trimesh.proximity.closest_point(mesh, query_points)

    valid = (face_ids >= 0) & (face_ids < len(mesh.faces))
    normals = np.zeros_like(query_points, dtype=float)

    if not np.any(valid):
        return normals, valid

    tri_faces = mesh.faces[face_ids[valid]]          # (M,3)
    tri_vertices = mesh.vertices[tri_faces]          # (M,3,3)
    tri_vnormals = mesh.vertex_normals[tri_faces]    # (M,3,3)

    bary = trimesh.triangles.points_to_barycentric(
        tri_vertices,
        closest_pts[valid]
    )                                                # (M,3)

    interp = (
        bary[:, [0]] * tri_vnormals[:, 0, :] +
        bary[:, [1]] * tri_vnormals[:, 1, :] +
        bary[:, [2]] * tri_vnormals[:, 2, :]
    )

    L = np.linalg.norm(interp, axis=1, keepdims=True)
    good = L.squeeze() > 1e-12
    interp[good] = interp[good] / L[good]

    normals[valid] = interp
    return normals, valid

def extract_ordered_perimeter_5axis(
    interface_mesh,
    normal_source_mesh=None,
    closed_tol=1e-3,
    point_spacing=0.5,
):
    """
    Extract ordered perimeter loops and corresponding normals/tangents/inward directions.
    """
    if interface_mesh is None or len(interface_mesh.faces) == 0:
        return []

    path_obj = interface_mesh.outline()
    if path_obj is None:
        return []

    discrete_paths = path_obj.discrete
    if discrete_paths is None or len(discrete_paths) == 0:
        return []

    if normal_source_mesh is None:
        normal_source_mesh = interface_mesh

    results = []

    for pts in discrete_paths:
        pts = np.asarray(pts, dtype=float)

        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            continue

        is_closed = np.linalg.norm(pts[0] - pts[-1]) <= closed_tol

        # 1. Uniform perimeter resampling
        pts_resampled = resample_polyline_uniform(
            pts,
            spacing=point_spacing,
            closed=is_closed
        )

        # 2. Smooth normals from patch mesh
        normals, valid = sample_barycentric_normals_on_mesh(
            normal_source_mesh,
            pts_resampled
        )

        # Fallback to nearest-vertex normals where needed
        if not np.all(valid):
            try:
                _, vertex_indices = normal_source_mesh.kdtree.query(pts_resampled[~valid])
                fallback_normals = normal_source_mesh.vertex_normals[vertex_indices]

                L = np.linalg.norm(fallback_normals, axis=1, keepdims=True)
                good = L.squeeze() > 1e-12
                fallback_normals[good] = fallback_normals[good] / L[good]

                normals[~valid] = fallback_normals
            except Exception:
                pass

        # 3. Tangent + inward surface direction
        tangents, inward_dirs, side_dirs_raw = compute_loop_frames(
            pts_resampled,
            normals,
            patch_mesh=interface_mesh,
            is_closed=is_closed
        )

        results.append({
            'points': pts_resampled,
            'normals': normals,
            'tangents': tangents,
            'inward_dirs': inward_dirs,
            'is_closed': is_closed
        })

    return results



def compute_loop_frames(points, normals, patch_mesh, is_closed=True, test_step=None):
    """
    Compute tangents and inward in-surface directions for a perimeter loop.

    Inward/outward sign is determined locally by testing which side stays
    closer to the patch mesh.

    Parameters
    ----------
    points : (N,3) array
        Ordered perimeter points.
    normals : (N,3) array
        Surface normals at those points.
    patch_mesh : trimesh.Trimesh
        Patch mesh used to decide inward/outward.
    is_closed : bool
        Whether the loop is closed.
    test_step : float or None
        Small offset step for inward/outward testing.
        If None, estimated from local point spacing.

    Returns
    -------
    tangents : (N,3)
    inward_dirs : (N,3)
    side_dirs_raw : (N,3)
    """
    pts = np.asarray(points, dtype=float)
    nrms = np.asarray(normals, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {pts.shape}")
    if nrms.ndim != 2 or nrms.shape[1] != 3:
        raise ValueError(f"normals must have shape (N,3), got {nrms.shape}")
    if len(pts) != len(nrms):
        raise ValueError("points and normals must have same length")
    if len(pts) < 2:
        raise ValueError("Need at least 2 points")
    if patch_mesh is None or len(patch_mesh.faces) == 0:
        raise ValueError("patch_mesh is empty")

    N = len(pts)

    # ------------------------------------------
    # 1. Normalize normals
    # ------------------------------------------
    normals_u = nrms.copy()
    Ln = np.linalg.norm(normals_u, axis=1, keepdims=True)
    good_n = Ln.squeeze() > 1e-12
    normals_u[good_n] /= Ln[good_n]

    # ------------------------------------------
    # 2. Compute tangents
    # ------------------------------------------
    tangents = np.zeros_like(pts)

    if is_closed:
        duplicate_last = np.linalg.norm(pts[0] - pts[-1]) < 1e-9
        if duplicate_last and N > 2:
            pts_work = pts[:-1]
            tangents_work = np.zeros_like(pts_work)
            M = len(pts_work)

            for i in range(M):
                p_prev = pts_work[(i - 1) % M]
                p_next = pts_work[(i + 1) % M]
                t = p_next - p_prev
                Lt = np.linalg.norm(t)
                if Lt > 1e-12:
                    tangents_work[i] = t / Lt

            tangents[:-1] = tangents_work
            tangents[-1] = tangents_work[0]
            normals_u[-1] = normals_u[0]
        else:
            for i in range(N):
                p_prev = pts[(i - 1) % N]
                p_next = pts[(i + 1) % N]
                t = p_next - p_prev
                Lt = np.linalg.norm(t)
                if Lt > 1e-12:
                    tangents[i] = t / Lt
    else:
        if N == 2:
            t = pts[1] - pts[0]
            Lt = np.linalg.norm(t)
            if Lt > 1e-12:
                tangents[:] = t / Lt
        else:
            t0 = pts[1] - pts[0]
            L0 = np.linalg.norm(t0)
            if L0 > 1e-12:
                tangents[0] = t0 / L0

            for i in range(1, N - 1):
                t = pts[i + 1] - pts[i - 1]
                Lt = np.linalg.norm(t)
                if Lt > 1e-12:
                    tangents[i] = t / Lt

            t1 = pts[-1] - pts[-2]
            L1 = np.linalg.norm(t1)
            if L1 > 1e-12:
                tangents[-1] = t1 / L1

    # ------------------------------------------
    # 3. Raw sideways direction on surface
    #    side = n x t
    # ------------------------------------------
    side_dirs_raw = np.cross(normals_u, tangents)
    Ls = np.linalg.norm(side_dirs_raw, axis=1, keepdims=True)
    good_s = Ls.squeeze() > 1e-12
    side_dirs_raw[good_s] /= Ls[good_s]

    # ------------------------------------------
    # 4. Choose test step
    # ------------------------------------------
    if test_step is None:
        segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if len(segs) > 0 and np.any(segs > 1e-12):
            test_step = 0.35 * np.median(segs[segs > 1e-12])
        else:
            test_step = 0.25

    # ------------------------------------------
    # 5. Local inward/outward test using patch distance
    # ------------------------------------------
    plus_pts = pts + test_step * side_dirs_raw
    minus_pts = pts - test_step * side_dirs_raw

    _, d_plus, _ = trimesh.proximity.closest_point(patch_mesh, plus_pts)
    _, d_minus, _ = trimesh.proximity.closest_point(patch_mesh, minus_pts)

    inward_dirs = side_dirs_raw.copy()

    # If +side is farther from the patch than -side, flip it
    flip_mask = d_plus > d_minus
    inward_dirs[flip_mask] *= -1.0

    # ------------------------------------------
    # 6. Fallback for near-ties using local centroid heuristic
    # ------------------------------------------
    tie_mask = np.abs(d_plus - d_minus) < 1e-8
    if np.any(tie_mask):
        if is_closed and np.linalg.norm(pts[0] - pts[-1]) < 1e-9 and N > 2:
            centroid = np.mean(pts[:-1], axis=0)
        else:
            centroid = np.mean(pts, axis=0)

        for i in np.where(tie_mask)[0]:
            to_center = centroid - pts[i]
            n = normals_u[i]
            to_center_tan = to_center - np.dot(to_center, n) * n
            Ltc = np.linalg.norm(to_center_tan)
            if Ltc > 1e-12:
                to_center_tan /= Ltc
                if np.dot(inward_dirs[i], to_center_tan) < 0:
                    inward_dirs[i] *= -1.0

    return tangents, inward_dirs, side_dirs_raw



def generate_perimeter_toolpaths_from_patch(
    patch_mesh,
    seed_loops,
    n_perimeters,
    perimeter_spacing,
    point_spacing=0.3,
    smooth_iterations=3,
    smooth_alpha=0.35,
    max_proj_dist=None,
    min_clearance_to_prev=None,
    min_keep_ratio=0.75,
    min_points=12,
):
    """
    Generate perimeter toolpaths from patch boundary.

    Current method:
    - ring 0 = seed perimeter
    - ring 1..N = Dijkstra distance-field iso-contours
    """
    if patch_mesh is None or len(patch_mesh.faces) == 0:
        print("[toolpaths] FAIL: patch mesh is empty")
        return []

    if seed_loops is None or len(seed_loops) == 0:
        print("[toolpaths] FAIL: no seed loops")
        return []

    if n_perimeters <= 0:
        print("[toolpaths] FAIL: n_perimeters <= 0")
        return []

    print("\n[toolpaths] =====================================")
    print(f"[toolpaths] requested total perimeters: {n_perimeters}")
    print(f"[toolpaths] perimeter spacing         : {perimeter_spacing}")
    print(f"[toolpaths] point spacing             : {point_spacing}")
    print(f"[toolpaths] number of seed loops      : {len(seed_loops)}")

    toolpaths = []

    for family_id, seed in enumerate(seed_loops):
        print(f"\n[toolpaths] ---- family_id={family_id} ----")
        print(f"[toolpaths] seed points: {len(seed['points'])}")

        # ------------------------------------------------------------
        # Ring 0 = exact seed perimeter
        # ------------------------------------------------------------
        seed_path = {
            "points": np.asarray(seed["points"], dtype=float).copy(),
            "normals": np.asarray(seed["normals"], dtype=float).copy(),
            "is_closed": bool(seed.get("is_closed", True)),
            "family_id": family_id,
            "ring_id": 0,
        }

        tangents, inward_dirs, _ = compute_loop_frames(
            seed_path["points"],
            seed_path["normals"],
            patch_mesh=patch_mesh,
            is_closed=seed_path["is_closed"]
        )
        seed_path["tangents"] = tangents
        seed_path["inward_dirs"] = inward_dirs

        toolpaths.append(seed_path)
        print(f"[toolpaths] ring_id=0 ACCEPTED, npts={len(seed_path['points'])}")

        # ------------------------------------------------------------
        # Inner perimeters by Dijkstra iso-contours
        # n_perimeters includes the seed ring
        # so inner count = total requested - 1
        # ------------------------------------------------------------
        n_inner = int(n_perimeters) - 1

        if n_inner > 0:
            inner_toolpaths, vertex_distance = generate_inner_perimeters_dijkstra(
                patch_mesh=patch_mesh,
                seed_loop=seed_path,
                perimeter_spacing=perimeter_spacing,
                n_inner_perimeters=n_inner,
                point_spacing=point_spacing,
                family_id=family_id,
            )

            for pd in inner_toolpaths:
                toolpaths.append(pd)
                print(f"[toolpaths] ring_id={pd['ring_id']} ACCEPTED, npts={len(pd['points'])}")

    print(f"\n[toolpaths] total returned loops: {len(toolpaths)}")
    return toolpaths


def build_patch_perimeter_toolpaths(
    layer_zero_surface,
    print_model,
    layer_height,
    n_perimeters=3,
    perimeter_spacing=0.3,
    gap_tol=0.10,
    euclid_tol=None,
    min_component_faces=20,
    keep_largest_only=True,
    point_spacing=None,
    max_proj_dist=None,
    min_clearance_to_prev=None,
    min_keep_ratio=0.75,
    min_points=12,
):
    """
    Full pipeline:
    1. extract patch from layer-zero surface
    2. extract seed perimeter from patch
    3. package seed perimeter as toolpath

    Current cleaned version:
    - UV parameterization/debug is disabled
    - only seed perimeter(s) are returned
    """
    print("\n[build_patch] ==================================")
    print(f"[build_patch] requested n_perimeters: {n_perimeters}")
    print(f"[build_patch] perimeter_spacing     : {perimeter_spacing}")
    print(f"[build_patch] layer_height          : {layer_height}")

    patch_mesh, seed_loops, debug = extract_first_layer_intersection_perimeter(
    layer_zero_surface=layer_zero_surface,
    print_model=print_model,
    layer_height=layer_height,
    gap_tol=gap_tol,
    euclid_tol=euclid_tol,
    min_component_faces=min_component_faces,
    keep_largest_only=keep_largest_only,
    remesh_target_edge=10,
    seed_point_spacing=point_spacing if point_spacing is not None else 0.5,
)

    if patch_mesh is None or len(patch_mesh.faces) == 0:
        print("[build_patch] FAIL: patch mesh empty")
        return patch_mesh, [], debug


    print(f"[build_patch] patch faces: {len(patch_mesh.faces)}")
    print(f"[build_patch] seed loops : {len(seed_loops)}")
    for i, seed in enumerate(seed_loops):
        print(f"[build_patch] seed loop {i}: npts={len(seed['points'])}, closed={seed.get('is_closed', None)}")

    if point_spacing is None:
        point_spacing = max(0.25, perimeter_spacing / 2.0)

    if max_proj_dist is None:
        max_proj_dist = max(0.5 * perimeter_spacing, 0.25)

    if min_clearance_to_prev is None:
        min_clearance_to_prev = 0.35 * perimeter_spacing

    print(f"[build_patch] point_spacing         : {point_spacing}")
    print(f"[build_patch] max_proj_dist         : {max_proj_dist}")
    print(f"[build_patch] min_clearance_to_prev : {min_clearance_to_prev}")
    print(f"[build_patch] min_keep_ratio        : {min_keep_ratio}")
    print(f"[build_patch] min_points            : {min_points}")

    toolpaths = generate_perimeter_toolpaths_from_patch(
        patch_mesh=patch_mesh,
        seed_loops=seed_loops,
        n_perimeters=n_perimeters,
        perimeter_spacing=perimeter_spacing,
        point_spacing=point_spacing,
        max_proj_dist=max_proj_dist,
        min_clearance_to_prev=min_clearance_to_prev,
        min_keep_ratio=min_keep_ratio,
        min_points=min_points,
    )
    print(f"[build_patch] FINAL returned loops: {len(toolpaths)}")
    return patch_mesh, toolpaths, debug



