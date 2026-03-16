import trimesh
import numpy as np
from intersection_patch import *
from UV_parametrization import *


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



# -------------------------------------------------------------------
# LEGACY 3D POINTWISE OFFSET PIPELINE
# Not needed in cleaned outline.py for now.
# Keep commented for reference or move later to legacy_offset.py
# -------------------------------------------------------------------
"""
def build_next_perimeter_candidate(
    current_path_data,
    patch_mesh,
    offset_distance,
    point_spacing=0.5,
    smooth_iterations=0,
    smooth_alpha=0.0,
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
"""
