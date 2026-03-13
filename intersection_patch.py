import trimesh
import numpy as np

def extract_model_patch_from_layer_zero(
    layer_zero_surface,
    print_model,
    layer_height,
    gap_tol=0.10,
    euclid_tol=None,
    normal_opp_threshold=-0.35,
    normal_offset_ratio_threshold = 0.75,
    vertex_band_ratio=1,
    min_component_faces=20,
    keep_largest_only=True,
):
    """
    Build a model-side first-layer patch from the substrate-side layer_zero_surface.

    Improved robust version:
    - center-based signed-gap test
    - local normal-opposition test
    - NEW: vertex support test to reject last side-wall triangles

    Parameters
    ----------
    layer_zero_surface : trimesh.Trimesh
        Substrate-side working patch.
    print_model : trimesh.Trimesh
        Full print model mesh.
    layer_height : float
        Layer height offset.
    gap_tol : float
        Tolerance for signed gap.
    euclid_tol : float or None
        Euclidean guard band. If None, auto-chosen.
    normal_opp_threshold : float
        Dot threshold between model-face normal and matched substrate-patch normal.
        Typical:
            -0.2  loose
            -0.35 moderate
            -0.5  strict
    vertex_band_ratio : float
        Fraction of triangle vertices that must pass the band test.
        2/3 is a very good default.
    """
    if layer_zero_surface is None or len(layer_zero_surface.faces) == 0:
        empty = trimesh.Trimesh(
            vertices=np.empty((0, 3)),
            faces=np.empty((0, 3), dtype=int)
        )
        return empty, {}

    if euclid_tol is None:
        euclid_tol = layer_height + 2.0 * gap_tol

    lz = layer_zero_surface.copy()

    try:
        lz.fix_normals()
    except Exception:
        pass

    # -------------------------------------------------
    # A. FACE-CENTER TESTS
    # -------------------------------------------------
    pm_face_centers = print_model.triangles_center
    pm_face_normals = print_model.face_normals



    closest_pts_c, distances_c, lz_face_ids_c = trimesh.proximity.closest_point(lz, pm_face_centers)
    # closest point, distance and id of triangle in lz or layer_0

    valid_c = (lz_face_ids_c >= 0) & (lz_face_ids_c < len(lz.faces))
    # for elements in lz_face_ids check their validity: its like a sanity check


    # checking if any of the get any validity or not
    if not np.any(valid_c):
        empty = trimesh.Trimesh(
            vertices=np.empty((0, 3)),
            faces=np.empty((0, 3), dtype=int)
        )
        return empty, {
            "face_mask": np.zeros(len(print_model.faces), dtype=bool),
            "signed_gap_center": np.full(len(print_model.faces), np.nan),
            "distances_center": distances_c,
            "normal_dot": np.full(len(print_model.faces), np.nan),
            "normal_offset_ratio": np.full(len(print_model.faces), np.nan),
        }

    # create lz_normals shape (N,3) where N is number of triangles in PM model
    lz_normals_c = np.zeros_like(pm_face_centers)

    #so each N will get normals from a triangle in LZ which is closes to corresponding PM
    lz_normals_c[valid_c] = lz.face_normals[lz_face_ids_c[valid_c]]

    # getting a direction vector between two center points (from print model to layer 0)
    gap_vec_c = pm_face_centers - closest_pts_c
    #gap_vec_c: (N,3)


    #give the distance between two center along the normal of cace in lz surface: a scalar unit
    signed_gap_center = np.einsum("ij,ij->i", gap_vec_c, lz_normals_c)


    #near_band: filter for finding PM model "valid" faces near layer_z and within layer_offset distance (0 < dist < layer_h)
    near_band = valid_c & (distances_c <= (layer_height + 2.0 * gap_tol))

    #check for the common sign in signed_dist (negative or positive) and set that sign
    if np.any(near_band):
        if np.mean(signed_gap_center[near_band]) < 0:
            signed_gap_center = -signed_gap_center

    #calculating the angle between face_normals of PM and nearest lz_face normals
    normal_dot = np.einsum("ij,ij->i", pm_face_normals, lz_normals_c)

    # NEW: require center-to-surface offset to be mostly normal, not tangential
    #its like a normal triangle: hypotenuse is like Euclidean, normal is like signed dist and tangential is like base
    normal_offset_ratio = np.abs(signed_gap_center) / (distances_c + 1e-8)
    #normal_offset_ratio=1 means gap is mostly normal,
    # normal_offset_ratio=0 means gap is mostly tangential,

    center_mask = (
            valid_c
            & (signed_gap_center >= -gap_tol)
            & (signed_gap_center <= layer_height + gap_tol)
            & (distances_c <= euclid_tol)
            & (normal_dot <= normal_opp_threshold)
            & (normal_offset_ratio >= normal_offset_ratio_threshold)
    )

    # -------------------------------------------------
    # B. VERTEX SUPPORT TEST
    # -------------------------------------------------
    # Query all model vertices against layer_zero_surface
    pm_vertices = print_model.vertices
    closest_pts_v, distances_v, lz_face_ids_v = trimesh.proximity.closest_point(lz, pm_vertices)

    #sanity check for all vertices in PM model
    valid_v = (lz_face_ids_v >= 0) & (lz_face_ids_v < len(lz.faces))

    #for the comparison of normals with PM normals, keep N equal
    lz_normals_v = np.zeros_like(pm_vertices)
    lz_normals_v[valid_v] = lz.face_normals[lz_face_ids_v[valid_v]]

    gap_vec_v = pm_vertices - closest_pts_v
    signed_gap_v = np.einsum("ij,ij->i", gap_vec_v, lz_normals_v)

    if np.any(valid_v):
        if np.nanmean(signed_gap_v[valid_v]) < 0:
            signed_gap_v = -signed_gap_v

    vertex_good = (
        valid_v
        & (signed_gap_v >= -gap_tol)
        & (signed_gap_v <= layer_height + gap_tol)
        & (distances_v <= euclid_tol)
    )

    # For each face, count how many of its 3 vertices pass
    face_vertex_ids = print_model.faces
    good_counts = vertex_good[face_vertex_ids].sum(axis=1)

    #exmaple good counts: [[True, True, True], [False, True, False]] based on faces vertex we get from face_vertex_ids

    #band ratio: how many vertices should a triangle have to be passed as good
    required_vertex_count = int(np.ceil(3 * vertex_band_ratio))
    vertex_support_mask = good_counts >= required_vertex_count

    # -------------------------------------------------
    # C. FINAL FACE MASK
    # -------------------------------------------------
    face_mask = center_mask & vertex_support_mask

    face_ids = np.where(face_mask)[0]

    if len(face_ids) == 0:
        empty = trimesh.Trimesh(
            vertices=np.empty((0, 3)),
            faces=np.empty((0, 3), dtype=int)
        )
        return empty, {
            "face_mask": face_mask,
            "signed_gap_center": signed_gap_center,
            "distances_center": distances_c,
            "normal_dot": normal_dot,
            "vertex_good_counts": good_counts,
        }

    patch_mesh = print_model.submesh([face_ids], append=True)

    # -------------------------------------------------
    # D. COMPONENT CLEANUP
    # -------------------------------------------------
    try:
        #if we have patches in the mesh that we got just now: it will turn them into individual components
        components = patch_mesh.split(only_watertight=False)

        #delete any components with less than min_component_faces
        components = [c for c in components if len(c.faces) >= min_component_faces]

        if len(components) == 0:
        #when there is not patches, all were small
            patch_mesh = trimesh.Trimesh(
                vertices=np.empty((0, 3)),
                faces=np.empty((0, 3), dtype=int)
            )
        elif keep_largest_only:
            #as the name suggest
            patch_mesh = max(components, key=lambda m: len(m.faces))
        else:
            #else join all of them
            patch_mesh = trimesh.util.concatenate(components)
    except Exception:
        pass

    debug = {
        "face_mask": face_mask,
        "signed_gap_center": signed_gap_center,
        "distances_center": distances_c,
        "normal_dot": normal_dot,
        "vertex_good_counts": good_counts,
    }

    return patch_mesh, debug