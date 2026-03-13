import numpy as np
import igl
import matplotlib.pyplot as plt

def compute_patch_uv_lscm(V_patch, F_patch):
    """
    Compute a local UV parameterization of a connected triangle patch using LSCM.

    Parameters
    ----------
    V_patch : (#V,3) float array
        Patch vertices.
    F_patch : (#F,3) int array
        Patch triangle faces.

    Returns
    -------
    UV : (#V,2) float array
        UV coordinates for each patch vertex.
    bnd : (#B,) int array
        Boundary loop vertex indices.
    bnd_uv : (#B,2) float array
        Boundary UV coordinates used as constraints.
    """
    V_patch = np.asarray(V_patch, dtype=float)
    F_patch = np.asarray(F_patch, dtype=np.int32)

    if len(V_patch) == 0 or len(F_patch) == 0:
        raise ValueError("Empty patch mesh")

    # Boundary loop of the triangle patch
    bnd = igl.boundary_loop(F_patch)

    if bnd is None or len(bnd) < 2:
        raise ValueError("Patch boundary loop not found or too short for LSCM")

    # Put boundary vertices on a circle, preserving boundary edge proportions
    bnd_uv = igl.map_vertices_to_circle(V_patch, bnd)

    # LSCM solve
    UV,ok = igl.lscm(V_patch, F_patch, bnd.astype(np.int32), bnd_uv)

    if not ok:
        raise RuntimeError("igl.lscm failed on this patch")

    return np.asarray(UV, dtype=float), np.asarray(bnd, dtype=np.int32), np.asarray(bnd_uv, dtype=float)

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

    closest_pts, distances, face_ids = igl.point_mesh_squared_distance(pts3d, V_patch, F_patch)
    # libigl returns squared distances in some bindings, but closest point and face id are what we use

    uv_pts = np.zeros((len(pts3d), 2), dtype=float)
    valid = np.zeros(len(pts3d), dtype=bool)

    for i, fid in enumerate(face_ids):
        if fid < 0 or fid >= len(F_patch):
            continue

        tri = F_patch[fid]
        tri_xyz = V_patch[tri]
        tri_uv = UV_patch[tri]

        bary = igl.barycentric_coordinates_tri(
            np.array([closest_pts[i]]),
            np.array([tri_xyz[0]]),
            np.array([tri_xyz[1]]),
            np.array([tri_xyz[2]])
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