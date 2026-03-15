import numpy as np
import igl
import matplotlib.pyplot as plt
from outline import *
from shapely.geometry import Polygon,MultiPolygon


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

def compute_patch_uv_lscm(V_patch, F_patch):
    """
    Compute a local UV parameterization of a connected triangle patch using LSCM.
    """
    V_patch = np.asarray(V_patch, dtype=float)
    F_patch = np.asarray(F_patch, dtype=np.int32)

    if len(V_patch) == 0 or len(F_patch) == 0:
        raise ValueError("Empty patch mesh")

    bnd = igl.boundary_loop(F_patch)

    if bnd is None or len(bnd) < 2:
        raise ValueError("Patch boundary loop not found or too short for LSCM")

    bnd_uv = igl.map_vertices_to_circle(V_patch, bnd)

    # Robust against binding return-order / extra-return differences
    res = igl.lscm(V_patch, F_patch, bnd.astype(np.int32), bnd_uv)

    """
    print("type(res):", type(res))
    if isinstance(res, tuple):
        print("tuple length:", len(res))
        for i, item in enumerate(res):
            print(f"  res[{i}] type:", type(item))
    """
    if isinstance(res, tuple):
        UV = res[0]
    else:
        UV = res

    UV = np.asarray(UV, dtype=float)

    if UV.ndim != 2 or UV.shape[1] != 2 or UV.shape[0] != V_patch.shape[0]:
        raise RuntimeError(f"Unexpected LSCM UV shape: {UV.shape}")

    return UV, np.asarray(bnd, dtype=np.int32), np.asarray(bnd_uv, dtype=float)

def barycentric_coords_2d_point_in_tri(p, a, b, c):
    """
    Barycentric coordinates of 2D point p in 2D triangle (a,b,c).
    """
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-15:
        return None

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=float)


def point_in_triangle_barycentric(bary, tol=1e-9):
    if bary is None:
        return False
    return np.all(bary >= -tol) and np.all(bary <= 1.0 + tol)


def map_points_3d_to_uv_on_patch(points3d, V_patch, F_patch, UV_patch):
    """
    Map arbitrary 3D points on/near the patch to UV by:
    1. nearest point on patch
    2. face lookup
    3. barycentric interpolation in that face
    """
    pts3d = np.asarray(points3d, dtype=float)
    V_patch = np.asarray(V_patch, dtype=float)
    F_patch = np.asarray(F_patch, dtype=np.int32)
    UV_patch = np.asarray(UV_patch, dtype=float)

    # libigl return order:
    #   sqrD, I, C
    sqrD, face_ids, closest_pts = igl.point_mesh_squared_distance(pts3d, V_patch, F_patch)

    sqrD = np.asarray(sqrD).reshape(-1)
    face_ids = np.asarray(face_ids).reshape(-1).astype(int)
    closest_pts = np.asarray(closest_pts, dtype=float)

    uv_pts = np.zeros((len(pts3d), 2), dtype=float)
    valid = np.zeros(len(pts3d), dtype=bool)

    for i, fid in enumerate(face_ids):
        if fid < 0 or fid >= len(F_patch):
            continue

        tri = F_patch[fid]
        tri_xyz = V_patch[tri]
        tri_uv = UV_patch[tri]

        bary = igl.barycentric_coordinates(
            np.array([closest_pts[i]], dtype=float),
            np.array([tri_xyz[0]], dtype=float),
            np.array([tri_xyz[1]], dtype=float),
            np.array([tri_xyz[2]], dtype=float)
        )[0]

        uv_pts[i] = bary[0] * tri_uv[0] + bary[1] * tri_uv[1] + bary[2] * tri_uv[2]
        valid[i] = True

    return uv_pts, valid, face_ids, closest_pts

def map_points_uv_to_3d_on_patch(points_uv, V_patch, F_patch, UV_patch, tol=1e-9):
    """
    Lift UV points back to 3D patch using face search + barycentric interpolation.

    Parameters
    ----------
    points_uv : (N,2) array
    V_patch   : (#V,3) array
    F_patch   : (#F,3) array
    UV_patch  : (#V,2) array

    Returns
    -------
    pts3d : (N,3) array
    valid : (N,) bool array
    face_ids : (N,) int array
    """
    points_uv = np.asarray(points_uv, dtype=float)
    V_patch = np.asarray(V_patch, dtype=float)
    F_patch = np.asarray(F_patch, dtype=np.int32)
    UV_patch = np.asarray(UV_patch, dtype=float)

    pts3d = np.zeros((len(points_uv), 3), dtype=float)
    valid = np.zeros(len(points_uv), dtype=bool)
    face_ids = -np.ones(len(points_uv), dtype=int)

    tri_uv_all = UV_patch[F_patch]   # (#F,3,2)
    tri_xyz_all = V_patch[F_patch]   # (#F,3,3)

    for i, puv in enumerate(points_uv):
        found = False
        for fid in range(len(F_patch)):
            uv_tri = tri_uv_all[fid]
            bary = barycentric_coords_2d_point_in_tri(puv, uv_tri[0], uv_tri[1], uv_tri[2])

            if point_in_triangle_barycentric(bary, tol=tol):
                xyz_tri = tri_xyz_all[fid]
                pts3d[i] = bary[0] * xyz_tri[0] + bary[1] * xyz_tri[1] + bary[2] * xyz_tri[2]
                valid[i] = True
                face_ids[i] = fid
                found = True
                break

        if not found:
            # leave invalid; caller may repair or reject
            pass

    return pts3d, valid, face_ids


def plot_patch_uv(UV_patch, F_patch, bnd_patch=None, seed_uv=None, title="Patch UV (LSCM)"):
    """
    Visualize triangle patch UV parameterization.

    Parameters
    ----------
    UV_patch : (#V,2) array
        UV coordinates of patch vertices.
    F_patch : (#F,3) int array
        Patch faces.
    bnd_patch : (#B,) int array or None
        Boundary loop vertex indices.
    seed_uv : (N,2) array or None
        Optional perimeter/toolpath in UV to overlay.
    """
    UV_patch = np.asarray(UV_patch, dtype=float)
    F_patch = np.asarray(F_patch, dtype=np.int32)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw mesh edges in UV
    for tri in F_patch:
        tri_uv = UV_patch[tri]
        tri_closed = np.vstack([tri_uv, tri_uv[0]])
        ax.plot(tri_closed[:, 0], tri_closed[:, 1], color='lightgray', linewidth=0.5)

    # Draw patch boundary if available
    if bnd_patch is not None and len(bnd_patch) > 0:
        bnd_uv = UV_patch[bnd_patch]
        if np.linalg.norm(bnd_uv[0] - bnd_uv[-1]) > 1e-12:
            bnd_uv = np.vstack([bnd_uv, bnd_uv[0]])
        ax.plot(bnd_uv[:, 0], bnd_uv[:, 1], color='red', linewidth=2, label='Patch Boundary')

    # Draw seed perimeter if available
    if seed_uv is not None and len(seed_uv) > 0:
        seed_uv = np.asarray(seed_uv, dtype=float)
        ax.plot(seed_uv[:, 0], seed_uv[:, 1], color='blue', linewidth=2, label='Seed Perimeter (UV)')
        ax.scatter(seed_uv[:, 0], seed_uv[:, 1], s=8, color='blue')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.show()

def debug_seed_uv_mapping(patch_mesh, seed_loop, show_plot=True):
    """
    Debug check:
    1. parameterize patch with LSCM
    2. map seed perimeter 3D -> UV
    3. map UV -> 3D
    4. compute back-mapping error
    """
    V_patch = np.asarray(patch_mesh.vertices, dtype=float)
    F_patch = np.asarray(patch_mesh.faces, dtype=np.int32)

    # Patch UV
    UV_patch, bnd_patch, bnd_uv = compute_patch_uv_lscm(V_patch, F_patch)

    # Seed perimeter in 3D
    seed_pts_3d = np.asarray(seed_loop["points"], dtype=float)

    # 3D -> UV
    seed_uv, valid_uv, seed_face_ids, seed_closest = map_points_3d_to_uv_on_patch(
        seed_pts_3d,
        V_patch,
        F_patch,
        UV_patch
    )

    # UV -> 3D
    seed_back_3d, valid_back, back_face_ids = map_points_uv_to_3d_on_patch(
        seed_uv,
        V_patch,
        F_patch,
        UV_patch
    )

    # Back-mapping error
    err = np.linalg.norm(seed_back_3d - seed_pts_3d, axis=1)

    print("\n[UV DEBUG] ===============================")
    print("seed points          :", len(seed_pts_3d))
    print("valid 3D->UV         :", np.sum(valid_uv), "/", len(valid_uv))
    print("valid UV->3D         :", np.sum(valid_back), "/", len(valid_back))
    print("back-map error min   :", np.min(err))
    print("back-map error max   :", np.max(err))
    print("back-map error mean  :", np.mean(err))

    if show_plot:
        plot_patch_uv(
            UV_patch,
            F_patch,
            bnd_patch=bnd_patch,
            seed_uv=seed_uv,
            title="Patch UV + Seed Perimeter"
        )

    return {
        "UV_patch": UV_patch,
        "bnd_patch": bnd_patch,
        "bnd_uv": bnd_uv,
        "seed_uv": seed_uv,
        "seed_back_3d": seed_back_3d,
        "valid_uv": valid_uv,
        "valid_back": valid_back,
        "error": err,
    }


def polyline_length(points, closed=False):
    pts = np.asarray(points, dtype=float)

    if pts.ndim != 2 or len(pts) < 2:
        return 0.0

    if closed and np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
        pts = pts[:-1]

    if len(pts) < 2:
        return 0.0

    L = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()

    if closed:
        L += np.linalg.norm(pts[-1] - pts[0])

    return float(L)


def detect_corner_vertices_nd(points, angle_deg_threshold=25.0, is_closed=True):
    pts = np.asarray(points, dtype=float)
    n = len(pts)

    if n < 3:
        return np.zeros(n, dtype=bool)

    duplicate_last = is_closed and np.linalg.norm(pts[0] - pts[-1]) < 1e-9
    if duplicate_last:
        pts_work = pts[:-1]
    else:
        pts_work = pts

    m = len(pts_work)
    corner_mask = np.zeros(m, dtype=bool)

    for i in range(m):
        if not is_closed and (i == 0 or i == m - 1):
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

        v_in /= L1
        v_out /= L2

        dot = np.clip(np.dot(v_in, v_out), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot))

        if angle_deg >= angle_deg_threshold:
            corner_mask[i] = True

    if duplicate_last:
        corner_mask = np.append(corner_mask, corner_mask[0])

    return corner_mask


def resample_polyline_uniform_nd(
    points,
    spacing,
    closed=False,
    preserve_corners=True,
    angle_deg_threshold=25.0,
):
    """
    Dimension-independent version for 2D or 3D polylines.
    """
    pts = np.asarray(points, dtype=float)

    if pts.ndim != 2 or len(pts) < 2:
        return pts.copy()

    if spacing is None or spacing <= 0:
        return pts.copy()

    duplicate_last = closed and np.linalg.norm(pts[0] - pts[-1]) < 1e-12
    if duplicate_last:
        pts_work = pts[:-1]
    else:
        pts_work = pts.copy()

    if len(pts_work) < 2:
        return pts.copy()

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

    if closed:
        vertex_s = cumlen[:-1]
    else:
        vertex_s = cumlen

    sample_s = np.arange(0.0, total_len, spacing)
    if len(sample_s) == 0:
        sample_s = np.array([0.0])

    if total_len - sample_s[-1] > 1e-9:
        sample_s = np.append(sample_s, total_len)

    if preserve_corners and len(pts_work) >= 3:
        corner_mask = detect_corner_vertices_nd(
            pts_work,
            angle_deg_threshold=angle_deg_threshold,
            is_closed=closed
        )
        corner_s = vertex_s[corner_mask]

        all_s = np.sort(np.concatenate([sample_s, corner_s]))
        merged = [all_s[0]]
        for s in all_s[1:]:
            if abs(s - merged[-1]) > 1e-9:
                merged.append(s)
        sample_s = np.asarray(merged, dtype=float)

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

    if closed:
        if np.linalg.norm(out[0] - out[-1]) > 1e-9:
            out = np.vstack([out, out[0]])

    return out

def build_next_perimeter_candidate_uv(
    current_path_data,
    patch_mesh,
    V_patch,
    F_patch,
    UV_patch,
    offset_distance_3d,
    point_spacing_3d=0.5,
    min_points=12,
    angle_deg_threshold=25.0,
):
    """
    Build next inward perimeter using a true UV-domain offset.

    Notes
    -----
    offset_distance_3d is converted to UV using the current loop's
    UV-length / 3D-length ratio. This is a first approximation.
    """
    if patch_mesh is None or len(patch_mesh.faces) == 0:
        return None

    pts_3d = np.asarray(current_path_data["points"], dtype=float)
    pts_uv = current_path_data.get("points_uv", None)
    is_closed = bool(current_path_data.get("is_closed", True))

    if pts_uv is None:
        pts_uv, valid_uv, face_ids_uv, closest_uv = map_points_3d_to_uv_on_patch(
            pts_3d, V_patch, F_patch, UV_patch
        )
        if np.sum(valid_uv) < min_points:
            return None
    else:
        pts_uv = np.asarray(pts_uv, dtype=float)

    if len(pts_uv) < min_points:
        return None

    # Remove duplicate last point for polygon construction
    if is_closed and np.linalg.norm(pts_uv[0] - pts_uv[-1]) < 1e-9:
        pts_uv_work = pts_uv[:-1]
    else:
        pts_uv_work = pts_uv.copy()

    if len(pts_uv_work) < 4:
        return None

    # -----------------------------------------
    # A. Convert desired 3D spacing to UV spacing
    # -----------------------------------------
    L3 = polyline_length(pts_3d, closed=is_closed)
    Luv = polyline_length(pts_uv, closed=is_closed)

    if L3 < 1e-12 or Luv < 1e-12:
        return None

    uv_per_3d = Luv / L3
    offset_distance_uv = offset_distance_3d * uv_per_3d
    point_spacing_uv = point_spacing_3d * uv_per_3d

    print("\n[UV OFFSET] ----------------------------")
    print("3D loop length      :", L3)
    print("UV loop length      :", Luv)
    print("uv_per_3d scale     :", uv_per_3d)
    print("offset_distance_uv  :", offset_distance_uv)
    print("point_spacing_uv    :", point_spacing_uv)

    # -----------------------------------------
    # B. Build polygon and inward offset in UV
    # -----------------------------------------
    poly = Polygon(pts_uv_work)

    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.is_empty or poly.area <= 1e-12:
        print("[UV OFFSET] FAIL: seed UV polygon invalid/empty")
        return None

    # join_style=2 -> miter corners
    inner = poly.buffer(-offset_distance_uv, join_style=2)

    if inner.is_empty:
        print("[UV OFFSET] FAIL: inward UV offset produced empty polygon")
        return None

    if isinstance(inner, MultiPolygon):
        inner = max(inner.geoms, key=lambda g: g.area)

    if inner.is_empty or inner.area <= 1e-12:
        print("[UV OFFSET] FAIL: inward UV offset invalid after MultiPolygon handling")
        return None

    next_uv = np.asarray(inner.exterior.coords, dtype=float)

    if len(next_uv) < min_points:
        print("[UV OFFSET] FAIL: too few points in raw offset UV loop")
        return None

    # -----------------------------------------
    # C. Resample in UV while preserving corners
    # -----------------------------------------
    next_uv = resample_polyline_uniform_nd(
        next_uv,
        spacing=point_spacing_uv,
        closed=True,
        preserve_corners=True,
        angle_deg_threshold=angle_deg_threshold,
    )

    if len(next_uv) < min_points:
        print("[UV OFFSET] FAIL: too few points after UV resampling")
        return None

    # -----------------------------------------
    # D. Map UV back to 3D patch
    # -----------------------------------------
    next_pts_3d, valid_back, face_ids_back = map_points_uv_to_3d_on_patch(
        next_uv,
        V_patch,
        F_patch,
        UV_patch
    )

    if np.sum(valid_back) < min_points:
        print("[UV OFFSET] FAIL: too few valid UV->3D mapped points")
        return None

    next_uv = next_uv[valid_back]
    next_pts_3d = next_pts_3d[valid_back]

    if len(next_pts_3d) < min_points:
        print("[UV OFFSET] FAIL: too few valid next 3D points")
        return None

    # -----------------------------------------
    # E. Sample patch normals on new 3D loop
    # -----------------------------------------
    next_normals, valid_normals = sample_barycentric_normals_on_mesh(
        patch_mesh,
        next_pts_3d
    )

    if np.sum(valid_normals) < min_points:
        print("[UV OFFSET] FAIL: too few valid normals on next loop")
        return None

    next_uv = next_uv[valid_normals]
    next_pts_3d = next_pts_3d[valid_normals]
    next_normals = next_normals[valid_normals]

    tangents_next, inward_next, _ = compute_loop_frames(
        next_pts_3d,
        next_normals,
        patch_mesh=patch_mesh,
        is_closed=True
    )

    print("[UV OFFSET] SUCCESS: next UV perimeter built")
    print("[UV OFFSET] next loop points:", len(next_pts_3d))

    return {
        "points": next_pts_3d,
        "points_uv": next_uv,
        "normals": next_normals,
        "tangents": tangents_next,
        "inward_dirs": inward_next,
        "is_closed": True,
    }

