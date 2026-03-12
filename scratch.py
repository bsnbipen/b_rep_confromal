

import trimesh
import pyvista as pv
import numpy as np
import os
import trimesh.proximity
import trimesh
import numpy as np
from GUI_Design import *
from collections import deque


def get_master_slicing_surface(print_model, substrate, padding=5):
    """
    Choice A: Slices based on the MAXIMUM EXTENT of the model.
    Ensures that even at 90-degree rotations, the footprint is 'Full'.
    """
    # 1. Calculate the largest dimension of the model once
    # We use the peak-to-peak (ptp) value of the original size
    extents = print_model.bounds[1] - print_model.bounds[0]
    max_dim = extents.max()  # This will be ~55mm for your part


    # 2. Get the current center of the rotated model
    center = print_model.centroid

    # --- DEBUG SECTION ---
    print(f"\n--- [DEBUG: CHOICE A] ---")
    print(f"Model World Center: {center}")
    print(f"Model AABB (Current): Min {print_model.bounds[0]} | Max {print_model.bounds[1]}")
    print(f"Calculated Max Dimension: {max_dim:.2f} mm")

    # 3. Create a fixed-size footprint centered on the model
    # This prevents the 'sliver' effect by forcing a square ROI
    half_size = (max_dim / 2.0) + padding

    min_x, max_x = center[0] - half_size, center[0] + half_size
    min_y, max_y = center[1] - half_size, center[1] + half_size

    # 4. Clipping Logic
    planes = [
        ([min_x, 0, 0], [1, 0, 0]), ([max_x, 0, 0], [-1, 0, 0]),
        ([0, min_y, 0], [0, 1, 0]), ([0, max_y, 0], [0, -1, 0])
    ]

    master_surface = substrate.copy()

    for origin, normal in planes:
        master_surface = master_surface.slice_plane(plane_origin=origin, plane_normal=normal)

    if master_surface.is_empty:
        print("[FAIL] Orientation moved model outside substrate bounds.")
        return None, print_model, substrate

    # 5. Extract the upward-facing surface
    # We keep the threshold low to maintain the curved B-Rep quality
    up_idx = np.where(master_surface.face_normals[:, 2] > 0.01)[0]

    if len(up_idx) == 0:
        print("[FAIL] No upward faces found. Check rotation.")
        return None, print_model, substrate

    master_skin = master_surface.submesh([up_idx], append=True)

    print(f"[SUCCESS] 'Choice A' Skin Extracted: {len(master_skin.faces)} faces.")
    return master_skin, print_model, substrate


from collections import deque
import numpy as np
import trimesh

from collections import deque
import numpy as np
import trimesh

def get_layer_zero(
    print_model,
    substrate,
    dist_tol=1.0,
    opposing_threshold=-0.7,
    padding=5,
    envelope_dist_tol=10,
    smooth_threshold=0.80,
    max_growth_rings=15,
):
    """
    Contact-centered substrate patch extraction.

    Goal:
    - Build a strict contact core from proximity + opposing normals
    - Expand that core into a larger substrate envelope for later
      offset/displacement + intersection operations

    For a normal comparison (dot product): -1 opposite, 0: perpendicular, 1: parallel
    Parameters
    ----------
    dist_tol : float
        Strict distance threshold for core seeds.
    opposing_threshold : float
        Strict normal opposition threshold for core seeds. (Ideal Case opposite in direction)
    padding : float
        XY padding around print_model bounds for ROI cropping.
    envelope_dist_tol : float
        Looser distance limit used only for the expanded envelope.
        This should usually be larger than dist_tol.
    smooth_threshold : float
        Dot product threshold between adjacent substrate face normals
        during envelope growth (Ideal Case same direction).
    max_growth_rings : int
        Maximum number of adjacency "rings" grown outward from the seed core.
        This is a very useful way to control surface padding.
    """

    # --- 1. DYNAMIC ROTATED BOUNDS ---
    bounds = print_model.bounds
    min_pts, max_pts = bounds[0], bounds[1]

    center_x = (max_pts[0] + min_pts[0]) / 2.0
    center_y = (max_pts[1] + min_pts[1]) / 2.0

    min_x, max_x = min_pts[0] - padding, max_pts[0] + padding
    min_y, max_y = min_pts[1] - padding, max_pts[1] + padding

    # --- 2. DEBUG ROI BOX ---
    height_z = max_pts[2] - min_pts[2]
    debug_roi_box = trimesh.creation.box(
        extents=[max_x - min_x, max_y - min_y, height_z + 10],
        transform=trimesh.transformations.translation_matrix(
            [center_x, center_y, (max_pts[2] + min_pts[2]) / 2.0]
        )
    )

    # --- 3. CROP THE SUBSTRATE ---
    cropped_sub = substrate.copy()
    planes = [
        ([min_x, 0, 0], [1, 0, 0]),
        ([max_x, 0, 0], [-1, 0, 0]),
        ([0, min_y, 0], [0, 1, 0]),
        ([0, max_y, 0], [0, -1, 0]),
    ]

    for origin, normal in planes:
        cropped_sub = cropped_sub.slice_plane(origin, normal)

    cropped_sub.merge_vertices()

    if len(cropped_sub.faces) == 0:
        empty_mesh = trimesh.Trimesh(
            vertices=np.empty((0, 3)),
            faces=np.empty((0, 3), dtype=int)
        )
        return empty_mesh, debug_roi_box, None

    # --- 4. FACE DATA ---
    face_centers = cropped_sub.triangles_center
    sub_normals = cropped_sub.face_normals
    n_faces = len(cropped_sub.faces)

    # --- 5. LOCAL FACE-PAIR MATCHING ---
    _, distances, pm_face_ids = trimesh.proximity.closest_point(print_model, face_centers)

    valid_pm = (pm_face_ids >= 0) & (pm_face_ids < len(print_model.faces))

    nearest_model_normals = np.zeros_like(sub_normals)
    nearest_model_normals[valid_pm] = print_model.face_normals[pm_face_ids[valid_pm]]

    normal_dot = np.einsum('ij,ij->i', sub_normals, nearest_model_normals)

    # --- 6. STRICT CONTACT CORE ---
    core_mask = valid_pm & (distances <= dist_tol) & (normal_dot <= opposing_threshold)
    core_indices = np.where(core_mask)[0]

    # --- 7. BUILD FACE ADJACENCY ---
    neighbors = [[] for _ in range(n_faces)]
    for f0, f1 in cropped_sub.face_adjacency:
        neighbors[f0].append(f1)
        neighbors[f1].append(f0)

    # --- 8. ENVELOPE GROWTH ---
    # Important:
    # Growth is NOT strict contact anymore.
    # It is just a controlled substrate expansion around the contact core.
    if len(core_indices) > 0:
        accepted = set(core_indices.tolist())
        queue = deque((idx, 0) for idx in core_indices.tolist())  # (face_id, ring_depth)

        while queue:
            current, depth = queue.popleft()

            if depth >= max_growth_rings:
                continue

            for nb in neighbors[current]:
                if nb in accepted:
                    continue

                if not valid_pm[nb]:
                    continue

                # Condition 1: stay within a broader model-neighborhood band
                within_envelope_band = distances[nb] <= envelope_dist_tol

                # Condition 2: remain a smooth continuation on the substrate
                smooth_enough = np.dot(sub_normals[current], sub_normals[nb]) >= smooth_threshold

                if within_envelope_band and smooth_enough:
                    accepted.add(nb)
                    queue.append((nb, depth + 1))

        final_indices = np.array(sorted(accepted), dtype=int)
    else:
        final_indices = np.array([], dtype=int)

    print(f"DEBUG: Cropped faces = {len(face_centers)}")
    print(f"DEBUG: Core count = {len(core_indices)}")
    print(f"DEBUG: Final envelope count = {len(final_indices)}")

    if len(core_indices) > 0:
        print(f"DEBUG: Core dist min/max = {distances[core_indices].min():.4f} / {distances[core_indices].max():.4f}")
        print(f"DEBUG: Core dot min/max  = {normal_dot[core_indices].min():.4f} / {normal_dot[core_indices].max():.4f}")

    # --- 9. BUILD OUTPUT ---
    if len(final_indices) > 0:
        layer_zero_surface = cropped_sub.submesh([final_indices], append=True)
    else:
        layer_zero_surface = trimesh.Trimesh(
            vertices=np.empty((0, 3)),
            faces=np.empty((0, 3), dtype=int)
        )

    # Cleanup shrapnel
    try:
        components = layer_zero_surface.split(only_watertight=False)
        if len(components) > 0:
            layer_zero_surface = max(components, key=lambda m: len(m.faces))
    except:
        pass

    core_mesh = cropped_sub.submesh([core_indices], append=True) if len(core_indices) > 0 else None

    return layer_zero_surface, debug_roi_box, core_mesh

def generate_offset_surface(base_skin, layer_index, layer_height):
    """
    Displaces the base contact skin outward, strictly enforcing an
    upward (+Z) build direction for additive manufacturing.
    """
    d = layer_index * layer_height
    V_base = base_skin.vertices

    # 1. Get the raw, chaotic normal vectors from the open mesh
    N_hat = base_skin.vertex_normals.copy()

    # 2. THE BULLETPROOF FIX: The Z-Hemisphere Lock
    # Extract just the Z-direction of every normal vector
    z_components = N_hat[:, 2]

    # Find all the normals that are pointing down (Z < 0)
    needs_flipping = z_components < 0

    # Mathematically flip ONLY the vectors that are pointing the wrong way
    N_hat[needs_flipping] *= -1.0

    # 3. The Vector Displacement Math
    V_new = V_base + (N_hat * d)

    # 4. Reconstruct the Surface
    offset_surface = trimesh.Trimesh(vertices=V_new, faces=base_skin.faces)

    return offset_surface

def offset_layer(skin_mesh, layer_height):
    """
    Mathematically displaces a B-Rep surface along its vertex normals.
    """
    print(f"Offsetting layer by {layer_height} units...")

    # 1. Grab the original vertices and their normal vectors
    V_old = skin_mesh.vertices
    N_hat = skin_mesh.vertex_normals

    # 2. The Vector Math: V_new = V_old + (h * N_hat)
    V_new = V_old + (N_hat * layer_height)

    # 3. Rebuild the Mesh
    # We create a brand new trimesh object using the displaced vertices,
    # but we keep the exact same triangle connections (faces) as Layer 0.
    offset_skin = trimesh.Trimesh(vertices=V_new, faces=skin_mesh.faces)

    return offset_skin

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

def extract_first_layer_intersection_perimeter(
    layer_zero_surface,
    print_model,
    layer_height,
    gap_tol=0.10,
    euclid_tol=None,
    min_component_faces=20,
    keep_largest_only=True,
):
    """
    Full helper:
    1. Build model-side first-layer patch
    2. Extract ordered perimeter loops from that patch
    """
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

    perimeter_loops = extract_ordered_perimeter_5axis(
        patch_mesh,
        normal_source_mesh=print_model
    )

    return patch_mesh, perimeter_loops, debug

def extract_ordered_perimeter_5axis(interface_mesh, normal_source_mesh=None, closed_tol=1e-3):
    """
    Extract ordered perimeter loops and corresponding normals.

    Uses path_obj.discrete so each returned path is already an ordered
    Nx3 array of points instead of a segment array like (N,2,3).

    Parameters
    ----------
    interface_mesh : trimesh.Trimesh
        Patch whose perimeter we want.
    normal_source_mesh : trimesh.Trimesh or None
        Mesh from which normals will be sampled. If None, uses interface_mesh.
    closed_tol : float
        Distance tolerance to determine whether the path is closed.

    Returns
    -------
    results : list of dict
        Each dict contains:
        - 'points': (N,3) ordered path points
        - 'normals': (N,3) sampled normals
        - 'is_closed': bool
    """
    if interface_mesh is None or len(interface_mesh.faces) == 0:
        return []

    path_obj = interface_mesh.outline()

    if path_obj is None:
        return []

    discrete_paths = path_obj.discrete
    if discrete_paths is None or len(discrete_paths) == 0:
        return []


    #normal inheritance if normal_source_mesh is None, then we use interface mesh normal
    if normal_source_mesh is None:
        normal_source_mesh = interface_mesh

    results = []

    for pts in discrete_paths:
        pts = np.asarray(pts)

        # Make sure shape is (N,3)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            continue

        # Sample normals from nearest vertices on the source mesh
        _, vertex_indices = normal_source_mesh.kdtree.query(pts)
        normals = normal_source_mesh.vertex_normals[vertex_indices]

        # Detect closed loop
        is_closed = np.linalg.norm(pts[0] - pts[-1]) <= closed_tol

        results.append({
            'points': pts,
            'normals': normals,
            'is_closed': is_closed
        })

    return results




def main():
    # ---------------------------------------------------------
    # A. Load your files (Initial state)
    # ---------------------------------------------------------
    base_dir = r"C:\Users\bb237\PycharmProjects\b_repp_offsetting\stl_files\ring"
    model_file = os.path.join(base_dir, r'cylinder_ring - ring_55mm_dia-1.stl')
    substrate_file = os.path.join(base_dir, r"cylinder_ring - 50_mm_dia-1.stl")

    print("[1/4] Loading meshes from disk...")
    mesh_in = trimesh.load_mesh(model_file)
    sub_in = trimesh.load_mesh(substrate_file)

    # ---------------------------------------------------------
    # B. Orientation & Manual Positioning
    # ---------------------------------------------------------
    print("[2/4] Launching Orientation GUI...")
    rotated_model, rotated_substrate, final_angles = get_user_orientation_gui(mesh_in, sub_in)

    print(f"\n[3/4] Orientation Locked at Euler Angles: {final_angles}")

    # ---------------------------------------------------------
    # C. Extract layer-zero / substrate envelope
    # ---------------------------------------------------------
    print("[4/4] Extracting layer-zero surface...")
    master_skin, roi_box, core_mesh = get_layer_zero(rotated_model, rotated_substrate)

    # ---------------------------------------------------------
    # D. Visualize result
    # ---------------------------------------------------------
    p = pv.Plotter()
    p.set_background("white")

    # rotated model
    if rotated_model is not None and len(rotated_model.faces) > 0:
        p.add_mesh(
            pv.wrap(rotated_model),
            color="#3070B3",
            opacity=0.5,
            show_edges=True,
            edge_color="black",
            label="Rotated Print Model"
        )

    # rotated substrate
    if rotated_substrate is not None and len(rotated_substrate.faces) > 0:
        p.add_mesh(
            pv.wrap(rotated_substrate),
            color="lightgray",
            opacity=0.2,
            show_edges=True,
            edge_color="gray",
            label="Rotated Substrate"
        )

    # green envelope surface
    if master_skin is not None and len(master_skin.faces) > 0:
        p.add_mesh(
            pv.wrap(master_skin),
            color="green",
            opacity=0.8,
            show_edges=True,
            edge_color="darkgreen",
            label="Master Slicing Surface (Layer 0)"
        )

    # red core/contact zone
    if core_mesh is not None and len(core_mesh.faces) > 0:
        p.add_mesh(
            pv.wrap(core_mesh),
            color="red",
            opacity=1.0,
            label="Contact Core"
        )

    # ROI box
    if roi_box is not None and len(roi_box.faces) > 0:
        p.add_mesh(
            pv.wrap(roi_box),
            color="black",
            style="wireframe",
            opacity=0.3,
            label="ROI Box"
        )

    p.add_legend()
    p.show_grid()
    p.show()

    # 3. Now that the parts are locked in, run your Broad/Narrow-phase extraction!
    #layer_zero, print_model, substrate = get_layer_zero(print_model_path, substrate_path)

"""
extracts the layer 0 or slicing layer  and then show the substrate, print_model and slicing layer

    # 2. Extract Layer 0
    layer_0_trimesh, print_model, substrate = get_master_slicing_surface(model_file, substrate_file)

    layer_height = 0.5  # Your DIW extrusion height in mm
    num_layers_to_preview = 3

    offset_layers_trimesh = []

    # Loop to generate Layer 1, Layer 2, and Layer 3
    for i in range(1, num_layers_to_preview + 1):
        offset_mesh = generate_offset_surface(layer_0_trimesh, layer_index=i, layer_height=layer_height)
        offset_layers_trimesh.append(offset_mesh)

    # ---------------------------------------------------------
    # 4. Visual Verification (Using PyVista)
    # ---------------------------------------------------------
    # Wrap Layer 0 and the substrate
    print_model_pv = pv.wrap(print_model)
    layer_0_pv = pv.wrap(layer_0_trimesh)
    #substrate_pv = pv.read(substrate_file)

    p = pv.Plotter()

    p.add_mesh(print_model_pv, color='lightblue', opacity=0.3, label="Print Model")  # <-- Added to plot
    #p.add_mesh(substrate_pv, color='lightgray', opacity=0.5, label="Substrate")
    p.add_mesh(layer_0_pv, color='red', show_edges=True, line_width=2, label="Layer 0 (Contact)")

    # Plot our newly generated offset layers with different colors
    colors = ['blue', 'green', 'orange']
    p.add_legend()
    p.show()
"""

"""
showing the displaced layers

    for i, offset_mesh in enumerate(offset_layers_trimesh):
        layer_pv = pv.wrap(offset_mesh)
        layer_dist = layer_height * (i + 1)
        p.add_mesh(
            layer_pv,
            color=colors[i],
            show_edges=True,
            line_width=1,
            opacity=0.8,
            label=f"Layer {i + 1} (+{layer_dist:.1f}mm)"
        )
"""



if __name__ == "__main__":
    main()