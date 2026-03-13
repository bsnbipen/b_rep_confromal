import trimesh
import numpy as np
from intersection_patch import *
from UV_parametrization import *


def extract_first_layer_intersection_perimeter(
    layer_zero_surface,
    print_model,
    layer_height,
    gap_tol=0.10,
    euclid_tol=None,
    min_component_faces=20,
    keep_largest_only=True,
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

    seed_loops = extract_ordered_perimeter_5axis(
        interface_mesh=patch_mesh,
        normal_source_mesh=patch_mesh,
        point_spacing=1.0
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

def fit_local_plane_axes(points):
    """
    Fit a local plane to 3D points using PCA and return centroid + 2D axes.
    """
    pts = np.asarray(points, dtype=float)
    centroid = np.mean(pts, axis=0)

    X = pts - centroid
    _, _, vh = np.linalg.svd(X, full_matrices=False)

    axis_u = vh[0]
    axis_v = vh[1]
    normal = vh[2]

    return centroid, axis_u, axis_v, normal


def project_points_to_local_2d(points, centroid, axis_u, axis_v):
    pts = np.asarray(points, dtype=float)
    X = pts - centroid
    u = X @ axis_u
    v = X @ axis_v
    return np.column_stack([u, v])


def orient2d(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect_2d(p1, p2, q1, q2, eps=1e-12):
    o1 = orient2d(p1, p2, q1)
    o2 = orient2d(p1, p2, q2)
    o3 = orient2d(q1, q2, p1)
    o4 = orient2d(q1, q2, p2)

    return ((o1 * o2) < -eps) and ((o3 * o4) < -eps)


def loop_has_self_intersection_2d(points2d, is_closed=True):
    """
    Detect if a polyline/loop has self-intersections in 2D.
    """
    pts = np.asarray(points2d, dtype=float)

    if len(pts) < 4:
        return False

    if is_closed and np.linalg.norm(pts[0] - pts[-1]) > 1e-12:
        pts = np.vstack([pts, pts[0]])

    nseg = len(pts) - 1

    for i in range(nseg):
        a1, a2 = pts[i], pts[i + 1]

        for j in range(i + 1, nseg):
            b1, b2 = pts[j], pts[j + 1]

            # skip adjacent segments
            if abs(i - j) <= 1:
                continue

            # skip first-last adjacency in closed loop
            if is_closed and i == 0 and j == nseg - 1:
                continue

            if segments_intersect_2d(a1, a2, b1, b2):
                return True

    return False

def displace_perimeter_inward(points, inward_dirs, offset_distance):
    pts = np.asarray(points, dtype=float)
    inward = np.asarray(inward_dirs, dtype=float)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {pts.shape}")
    if inward.ndim != 2 or inward.shape[1] != 3:
        raise ValueError(f"inward_dirs must have shape (N,3), got {inward.shape}")
    if len(pts) != len(inward):
        raise ValueError("points and inward_dirs must have same length")

    return pts + float(offset_distance) * inward

def project_points_to_patch(points_guess, patch_mesh):
    """
    Step 2: project guessed inward points back onto the patch surface
    and sample new patch normals there.

    Parameters
    ----------
    points_guess : (N,3) array
        Inward-displaced guess points.
    patch_mesh : trimesh.Trimesh
        Patch mesh used for projection.

    Returns
    -------
    pts_proj : (N,3) array
        Closest points on the patch.
    normals_proj : (N,3) array
        Patch normals sampled at projected points.
    distances_proj : (N,) array
        Distance from guess points to projected points.
    valid_proj : (N,) bool array
        Valid projection mask.
    face_ids_proj : (N,) int array
        Matched patch face ids.
    """
    pts_guess = np.asarray(points_guess, dtype=float)

    if patch_mesh is None or len(patch_mesh.faces) == 0:
        raise ValueError("patch_mesh is empty")

    if pts_guess.ndim != 2 or pts_guess.shape[1] != 3:
        raise ValueError(f"points_guess must have shape (N,3), got {pts_guess.shape}")

    pts_proj, distances_proj, face_ids_proj = trimesh.proximity.closest_point(
        patch_mesh,
        pts_guess
    )

    valid_proj = (face_ids_proj >= 0) & (face_ids_proj < len(patch_mesh.faces))

    normals_proj, valid_normals = sample_barycentric_normals_on_mesh(
        patch_mesh,
        pts_proj
    )

    valid_proj = valid_proj & valid_normals

    return pts_proj, normals_proj, distances_proj, valid_proj, face_ids_proj

def smooth_polyline(points, is_closed=True, iterations=3, alpha=0.35):
    """
    Laplacian-like smoothing for an ordered 3D polyline.

    Parameters
    ----------
    points : (N,3) array
    is_closed : bool
    iterations : int
    alpha : float
        0 -> no smoothing
        1 -> fully move to neighbor average

    Returns
    -------
    pts_sm : (N,3) array
    """
    pts = np.asarray(points, dtype=float).copy()

    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
        return pts

    duplicate_last = is_closed and np.linalg.norm(pts[0] - pts[-1]) < 1e-9

    if duplicate_last:
        work = pts[:-1].copy()
    else:
        work = pts.copy()

    M = len(work)
    if M < 3:
        return pts

    for _ in range(iterations):
        new_work = work.copy()

        if is_closed:
            for i in range(M):
                prev_p = work[(i - 1) % M]
                next_p = work[(i + 1) % M]
                avg_nb = 0.5 * (prev_p + next_p)
                new_work[i] = (1.0 - alpha) * work[i] + alpha * avg_nb
        else:
            # keep endpoints fixed
            for i in range(1, M - 1):
                prev_p = work[i - 1]
                next_p = work[i + 1]
                avg_nb = 0.5 * (prev_p + next_p)
                new_work[i] = (1.0 - alpha) * work[i] + alpha * avg_nb

        work = new_work

    if duplicate_last:
        return np.vstack([work, work[0]])
    return work

def resample_and_smooth_projected_loop(
    points_proj,
    patch_mesh,
    is_closed=True,
    point_spacing=0.5,
    smooth_iterations=0,
    smooth_alpha=0.35,
):
    """
    Step 3:
    - resample projected loop to uniform spacing
    - smooth it
    - project it back to the patch
    - resample again
    - project again so final points really lie on patch
    """
    pts_proj = np.asarray(points_proj, dtype=float)

    if pts_proj.ndim != 2 or pts_proj.shape[1] != 3 or len(pts_proj) < 2:
        return pts_proj, np.zeros_like(pts_proj), np.array([], dtype=float), np.array([], dtype=bool)

    # 1. Uniform resampling
    pts_rs = resample_polyline_uniform(
        pts_proj,
        spacing=point_spacing,
        closed=is_closed
    )

    # 2. Smooth
    pts_sm = smooth_polyline(
        pts_rs,
        is_closed=is_closed,
        iterations=smooth_iterations,
        alpha=smooth_alpha
    )

    # 3. Project back
    pts_sm_proj, norms_sm, d_sm, valid_sm, face_ids_sm = project_points_to_patch(
        pts_sm,
        patch_mesh
    )

    # 4. Resample again
    pts_final_guess = resample_polyline_uniform(
        pts_sm_proj,
        spacing=point_spacing,
        closed=is_closed
    )

    # 5. Final projection so final points sit on patch
    pts_final, norms_final, d_final, valid_final, face_ids_final = project_points_to_patch(
        pts_final_guess,
        patch_mesh
    )

    return pts_final, norms_final, d_final, valid_final

def check_offset_loop_validity(
    original_pts,
    offset_pts,
    valid_mask,
    min_inward_move=0.15,
    max_proj_dist=None,
    proj_distances=None,
):
    """
    Step 4: basic validity check for a new inward perimeter candidate.

    Parameters
    ----------
    original_pts : (N,3)
        Original perimeter points.
    offset_pts : (N,3)
        Candidate inward perimeter points after Step 3.
    valid_mask : (N,) bool
        Projection/normal validity mask.
    min_inward_move : float
        Minimum displacement magnitude required.
    max_proj_dist : float or None
        Optional max allowed projection distance.
    proj_distances : (N,) or None
        Distances from smoothed points to projected patch points.

    Returns
    -------
    keep_mask : (N,) bool
        Which points are acceptable.
    """
    original_pts = np.asarray(original_pts, dtype=float)
    offset_pts = np.asarray(offset_pts, dtype=float)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    keep_mask = valid_mask.copy()

    # Must have actually moved inward enough
    move_dist = np.linalg.norm(offset_pts - original_pts, axis=1)
    keep_mask &= (move_dist >= min_inward_move)

    # Optional projection-distance sanity check
    if max_proj_dist is not None and proj_distances is not None:
        proj_distances = np.asarray(proj_distances, dtype=float)
        keep_mask &= (proj_distances <= max_proj_dist)

    return keep_mask

def remove_or_reject_self_intersections(
    points,
    normals,
    is_closed=True,
    planarity_tol_ratio=0.15,
):
    """
    Step 5:
    Detect self-intersections in a local 2D projection only if the loop
    is planar enough. If the loop is too curved in 3D, skip rejection.

    Parameters
    ----------
    points : (N,3)
    normals : (N,3)
    is_closed : bool
    planarity_tol_ratio : float
        If loop thickness / loop size is larger than this, treat as non-planar
        and skip 2D self-intersection rejection.

    Returns
    -------
    pts_out : (N,3)
    nrms_out : (N,3)
    ok : bool
        True means accept loop
        False means reject loop
    """
    pts = np.asarray(points, dtype=float)
    nrms = np.asarray(normals, dtype=float)

    if len(pts) < 4:
        return pts, nrms, True

    centroid, axis_u, axis_v, plane_n = fit_local_plane_axes(pts)

    # ------------------------------------------
    # 1. Measure loop non-planarity
    # ------------------------------------------
    rel = pts - centroid
    signed_plane_dist = rel @ plane_n
    max_plane_dev = np.max(np.abs(signed_plane_dist))

    bbox_diag = np.linalg.norm(np.ptp(pts, axis=0))
    if bbox_diag < 1e-12:
        return pts, nrms, True

    nonplanarity_ratio = max_plane_dev / bbox_diag

    print(f"[step5] max_plane_dev      : {max_plane_dev:.6f}")
    print(f"[step5] bbox_diag          : {bbox_diag:.6f}")
    print(f"[step5] nonplanarity_ratio: {nonplanarity_ratio:.6f}")

    # ------------------------------------------
    # 2. If loop is too non-planar, skip 2D rejection
    # ------------------------------------------
    if nonplanarity_ratio > planarity_tol_ratio:
        print("[step5] loop is too non-planar for reliable 2D intersection test; accepting loop")
        return pts, nrms, True

    # ------------------------------------------
    # 3. Otherwise do 2D self-intersection test
    # ------------------------------------------
    pts2d = project_points_to_local_2d(pts, centroid, axis_u, axis_v)
    has_x = loop_has_self_intersection_2d(pts2d, is_closed=is_closed)

    print(f"[step5] planar enough, 2D self-intersection = {has_x}")

    if has_x:
        return np.empty((0, 3)), np.empty((0, 3)), False

    return pts, nrms, True


def resample_and_smooth_projected_loop(
    points_proj,
    patch_mesh,
    is_closed=True,
    point_spacing=0.5,
    smooth_iterations=3,
    smooth_alpha=0.35,
):
    """
    Step 3:
    - resample projected loop to uniform spacing
    - smooth it
    - project it back to the patch
    - resample again
    - project again so final points really lie on patch
    """
    pts_proj = np.asarray(points_proj, dtype=float)

    if pts_proj.ndim != 2 or pts_proj.shape[1] != 3 or len(pts_proj) < 2:
        return pts_proj, np.zeros_like(pts_proj), np.array([], dtype=float), np.array([], dtype=bool)

    # 1. Uniform resampling
    pts_rs = resample_polyline_uniform(
        pts_proj,
        spacing=point_spacing,
        closed=is_closed
    )

    # 2. Smooth
    pts_sm = smooth_polyline(
        pts_rs,
        is_closed=is_closed,
        iterations=smooth_iterations,
        alpha=smooth_alpha
    )

    # 3. Project back
    pts_sm_proj, norms_sm, d_sm, valid_sm, face_ids_sm = project_points_to_patch(
        pts_sm,
        patch_mesh
    )

    # 4. Resample again
    pts_final_guess = resample_polyline_uniform(
        pts_sm_proj,
        spacing=point_spacing,
        closed=is_closed
    )

    # 5. Final projection so final points sit on patch
    pts_final, norms_final, d_final, valid_final, face_ids_final = project_points_to_patch(
        pts_final_guess,
        patch_mesh
    )

    return pts_final, norms_final, d_final, valid_final

def nearest_pointset_distances(query_points, ref_points):
    """
    Brute-force nearest distance from each query point to a reference point set.
    Fine for perimeter-sized point clouds.
    """
    q = np.asarray(query_points, dtype=float)
    r = np.asarray(ref_points, dtype=float)

    if len(q) == 0 or len(r) == 0:
        return np.array([], dtype=float)

    diff = q[:, None, :] - r[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    return d.min(axis=1)


def build_next_perimeter_candidate(
    current_path_data,
    patch_mesh,
    offset_distance,
    point_spacing=0.5,
    smooth_iterations=3,
    smooth_alpha=0.35,
    max_proj_dist=None,
    min_clearance_to_prev=None,
    min_keep_ratio=0.75,
    min_points=12,
):
    if patch_mesh is None or len(patch_mesh.faces) == 0:
        print("[build_next] FAIL: empty patch mesh")
        return None

    pts = np.asarray(current_path_data["points"], dtype=float)
    norms = np.asarray(current_path_data["normals"], dtype=float)
    is_closed = bool(current_path_data.get("is_closed", True))

    print("\n[build_next] ------------------------------")
    print(f"[build_next] input pts: {len(pts)}")
    print(f"[build_next] offset_distance: {offset_distance}")
    print(f"[build_next] point_spacing : {point_spacing}")

    if len(pts) < 2:
        print("[build_next] FAIL: too few input points")
        return None

    tangents = current_path_data.get("tangents", None)
    inward_dirs = current_path_data.get("inward_dirs", None)

    if tangents is None or inward_dirs is None:
        tangents, inward_dirs, _ = compute_loop_frames(
            pts,
            norms,
            patch_mesh=patch_mesh,
            is_closed=is_closed
        )

    if max_proj_dist is None:
        max_proj_dist = max(0.5 * point_spacing, 0.6 * offset_distance)

    if min_clearance_to_prev is None:
        min_clearance_to_prev = 0.35 * offset_distance

    print(f"[build_next] max_proj_dist      : {max_proj_dist}")
    print(f"[build_next] min_clearance_prev: {min_clearance_to_prev}")
    print(f"[build_next] min_keep_ratio    : {min_keep_ratio}")
    print(f"[build_next] min_points        : {min_points}")

    # ------------------------------
    # Step 1: inward displacement
    # ------------------------------
    pts_guess = displace_perimeter_inward(
        pts,
        inward_dirs,
        offset_distance
    )
    print(f"[build_next] step1 guess pts: {len(pts_guess)}")

    # ------------------------------
    # Step 2: project to patch
    # ------------------------------
    pts_proj, norms_proj, d_proj, valid_proj, face_ids_proj = project_points_to_patch(
        pts_guess,
        patch_mesh
    )

    print(f"[build_next] step2 valid proj: {np.sum(valid_proj)} / {len(valid_proj)}")
    if len(d_proj) > 0:
        print(f"[build_next] step2 proj dist min/max: {np.min(d_proj):.6f} / {np.max(d_proj):.6f}")

    if np.sum(valid_proj) < min_points:
        print("[build_next] FAIL at step2: too few valid projected points")
        return None

    # ------------------------------
    # Step 3: resample / smooth / reproject
    # ------------------------------
    pts_step3, norms_step3, d_step3, valid_step3 = resample_and_smooth_projected_loop(
        pts_proj,
        patch_mesh=patch_mesh,
        is_closed=is_closed,
        point_spacing=point_spacing,
        smooth_iterations=smooth_iterations,
        smooth_alpha=smooth_alpha
    )

    print(f"[build_next] step3 pts: {len(pts_step3)}")
    print(f"[build_next] step3 valid: {np.sum(valid_step3)} / {len(valid_step3)}")
    if len(d_step3) > 0:
        print(f"[build_next] step3 proj dist min/max: {np.min(d_step3):.6f} / {np.max(d_step3):.6f}")

    if len(pts_step3) < min_points:
        print("[build_next] FAIL at step3: too few points after smoothing")
        return None

    # ------------------------------
    # Step 4: validity / inside-ness
    # ------------------------------
    keep_mask = np.asarray(valid_step3, dtype=bool).copy()

    if len(d_step3) == len(pts_step3):
        keep_mask &= (d_step3 <= max_proj_dist)

    prev_d = nearest_pointset_distances(pts_step3, pts)
    if len(prev_d) == len(pts_step3):
        keep_mask &= (prev_d >= min_clearance_to_prev)

    keep_ratio = float(np.mean(keep_mask)) if len(keep_mask) > 0 else 0.0
    print(f"[build_next] step4 kept pts: {np.sum(keep_mask)} / {len(keep_mask)}")
    print(f"[build_next] step4 keep_ratio: {keep_ratio:.6f}")
    if len(prev_d) > 0:
        print(f"[build_next] step4 prev_dist min/max: {np.min(prev_d):.6f} / {np.max(prev_d):.6f}")

    if keep_ratio < min_keep_ratio or np.sum(keep_mask) < min_points:
        print("[build_next] FAIL at step4: keep ratio or point count too low")
        return None

    pts_step4 = pts_step3[keep_mask]
    norms_step4 = norms_step3[keep_mask]

    pts_step4 = resample_polyline_uniform(
        pts_step4,
        spacing=point_spacing,
        closed=is_closed
    )
    pts_step4, norms_step4, d_step4b, valid_step4b, face_ids_step4b = project_points_to_patch(
        pts_step4,
        patch_mesh
    )

    print(f"[build_next] step4b valid reproj: {np.sum(valid_step4b)} / {len(valid_step4b)}")

    if np.sum(valid_step4b) < min_points:
        print("[build_next] FAIL at step4b: too few valid reprojection points")
        return None

    # ------------------------------
    # Step 5: self-intersection
    # ------------------------------
    pts_step5, norms_step5, ok_step5 = remove_or_reject_self_intersections(
        pts_step4,
        norms_step4,
        is_closed=is_closed
    )

    print(f"[build_next] step5 self-intersection ok: {ok_step5}")
    print(f"[build_next] step5 pts: {len(pts_step5)}")

    if not ok_step5 or len(pts_step5) < min_points:
        print("[build_next] FAIL at step5: self-intersection or too few points")
        return None

    tangents_next, inward_next, _ = compute_loop_frames(
        pts_step5,
        norms_step5,
        patch_mesh=patch_mesh,
        is_closed=is_closed
    )

    next_path_data = {
        "points": pts_step5,
        "normals": norms_step5,
        "tangents": tangents_next,
        "inward_dirs": inward_next,
        "is_closed": is_closed,
        "offset_guess_points": pts_guess,
        "offset_projected_points": pts_proj,
        "offset_smoothed_points": pts_step3,
        "offset_keep_mask": keep_mask,
        "offset_keep_ratio": keep_ratio,
    }

    print("[build_next] SUCCESS: next perimeter built")
    return next_path_data

def generate_perimeter_toolpaths_from_patch(
    patch_mesh,
    seed_loops,
    n_perimeters,
    perimeter_spacing,
    point_spacing=0.5,
    smooth_iterations=3,
    smooth_alpha=0.35,
    max_proj_dist=None,
    min_clearance_to_prev=None,
    min_keep_ratio=0.75,
    min_points=12,
):
    """
    Generate perimeter toolpaths from patch boundary.
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
    print(f"[toolpaths] min_keep_ratio            : {min_keep_ratio}")
    print(f"[toolpaths] min_points                : {min_points}")
    print(f"[toolpaths] min_clearance_to_prev     : {min_clearance_to_prev}")
    print(f"[toolpaths] max_proj_dist             : {max_proj_dist}")

    toolpaths = []

    for family_id, seed in enumerate(seed_loops):
        print(f"\n[toolpaths] ---- family_id={family_id} ----")
        print(f"[toolpaths] seed points: {len(seed['points'])}")

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

        current = seed_path

        # build inward perimeters
        for ring_id in range(1, int(n_perimeters)):
            print(f"\n[toolpaths] attempting ring_id={ring_id}")

            nxt = build_next_perimeter_candidate(
                current_path_data=current,
                patch_mesh=patch_mesh,
                offset_distance=perimeter_spacing,
                point_spacing=point_spacing,
                smooth_iterations=smooth_iterations,
                smooth_alpha=smooth_alpha,
                max_proj_dist=max_proj_dist,
                min_clearance_to_prev=min_clearance_to_prev,
                min_keep_ratio=min_keep_ratio,
                min_points=min_points,
            )

            if nxt is None:
                print(f"[toolpaths] ring_id={ring_id} FAILED")
                break

            nxt["family_id"] = family_id
            nxt["ring_id"] = ring_id
            toolpaths.append(nxt)
            current = nxt

            print(f"[toolpaths] ring_id={ring_id} ACCEPTED, npts={len(nxt['points'])}")

    print(f"\n[toolpaths] total returned loops: {len(toolpaths)}")
    return toolpaths


def build_patch_perimeter_toolpaths(
    layer_zero_surface,
    print_model,
    layer_height,
    n_perimeters=2,
    perimeter_spacing=0.3,
    gap_tol=0.10,
    euclid_tol=None,
    min_component_faces=20,
    keep_largest_only=True,
    point_spacing=None,
    # expose these so you can tune from GUI/debugging later
    max_proj_dist=None,
    min_clearance_to_prev=None,
    min_keep_ratio=0.75,
    min_points=12,
):
    """
    Full pipeline:
    1. extract patch from layer-zero surface
    2. extract seed perimeter from patch
    3. generate requested number of perimeter toolpaths
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
    )

    if patch_mesh is None or len(patch_mesh.faces) == 0:
        print("[build_patch] FAIL: patch mesh empty")
        return patch_mesh, [], debug

    V_patch = np.asarray(patch_mesh.vertices, dtype=float)
    F_patch = np.asarray(patch_mesh.faces, dtype=np.int32)

    UV_patch, bnd_patch, bnd_uv = compute_patch_uv_lscm(V_patch, F_patch)
    plot_patch_uv(UV_patch, F_patch, bnd_patch=bnd_patch)


    seed_pts_3d = seed_loops[0]["points"]

    seed_uv, valid_uv, seed_face_ids, seed_closest = map_points_3d_to_uv_on_patch(
        seed_pts_3d,
        V_patch,
        F_patch,
        UV_patch
    )

    print("UV_patch shape:", UV_patch.shape)
    print("Boundary vertices:", len(bnd_patch))
    print("Boundary UV shape:", bnd_uv.shape)

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
        smooth_iterations=3,
        smooth_alpha=0.35,
        max_proj_dist=max_proj_dist,
        min_clearance_to_prev=min_clearance_to_prev,
        min_keep_ratio=min_keep_ratio,
        min_points=min_points,
    )

    print(f"[build_patch] FINAL returned loops: {len(toolpaths)}")
    return patch_mesh, toolpaths, debug