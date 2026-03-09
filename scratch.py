

import trimesh
import pyvista as pv
import numpy as np
import os
import trimesh.proximity
import trimesh
import numpy as np
from GUI_Design import *


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


def get_layer_zero(print_model, substrate, dist_tol=1.0, opposing_threshold=-0.7, padding=15):
    """
    Total Coverage Version: Dynamically expands the yellow ROI box to
    ensure the entire rotated footprint is captured.
    """
    # --- 1. DYNAMIC ROTATED BOUNDS ---
    # We get the min/max of the model in its CURRENT orientation
    bounds = print_model.bounds
    min_pts, max_pts = bounds[0], bounds[1]

    # Calculate the current width and depth
    width_x = max_pts[0] - min_pts[0]
    width_y = max_pts[1] - min_pts[1]

    # The center of the 'shadow' on the XY plane
    center_x = (max_pts[0] + min_pts[0]) / 2.0
    center_y = (max_pts[1] + min_pts[1]) / 2.0

    # Expand by padding
    min_x, max_x = min_pts[0] - padding, max_pts[0] + padding
    min_y, max_y = min_pts[1] - padding, max_pts[1] + padding

    # --- 2. DEBUG ASSET: THE YELLOW BOX ---
    # We make the box tall enough to enclose the whole model height
    height_z = max_pts[2] - min_pts[2]
    debug_roi_box = trimesh.creation.box(
        extents=[max_x - min_x, max_y - min_y, height_z + 10],
        transform=trimesh.transformations.translation_matrix([center_x, center_y, (max_pts[2] + min_pts[2]) / 2.0])
    )

    # --- 3. THE GUILLOTINE ---
    cropped_sub = substrate.copy()
    planes = [
        ([min_x, 0, 0], [1, 0, 0]), ([max_x, 0, 0], [-1, 0, 0]),
        ([0, min_y, 0], [0, 1, 0]), ([0, max_y, 0], [0, -1, 0])                                                         #creating planes along with normals, we can tell which side of plane are we keeping
    ]
    for origin, normal in planes:
        cropped_sub = cropped_sub.slice_plane(origin, normal)

    cropped_sub.merge_vertices()

    # --- 4. SEED & GROWTH (Same as before but with expanded area) ---
    face_centers = cropped_sub.triangles_center
    _, distances, pm_face_ids = trimesh.proximity.closest_point(print_model, face_centers)
    sub_normals = cropped_sub.face_normals
    pm_normals = print_model.face_normals[pm_face_ids]
    dot_products = np.sum(sub_normals * pm_normals, axis=1)

    seed_indices = np.where((distances <= dist_tol) & (dot_products < opposing_threshold))[0]                           #plane underneath the print model in substrate

    # Directional alignment for the 'Green' extension
    v_nozzle = -print_model.face_normals.mean(axis=0)   #ver
    v_nozzle /= (np.linalg.norm(v_nozzle) + 1e-8)
    alignment = np.dot(sub_normals, v_nozzle)

    # Take EVERYTHING in the box that faces the nozzle
    valid_indices = np.where(alignment > 0.05)[0]

    # Combine Seed + ROI
    final_indices = list(set(seed_indices).union(set(valid_indices)))
    layer_zero_surface = cropped_sub.submesh([final_indices], append=True)

    # Clean up shrapnel
    try:
        components = layer_zero_surface.split(only_watertight=False)
        layer_zero_surface = max(components, key=lambda m: len(m.faces))
    except:
        pass

    seed_mesh = cropped_sub.submesh([seed_indices], append=True) if len(seed_indices) > 0 else None

    return layer_zero_surface, debug_roi_box, seed_mesh

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


def extract_ordered_perimeter(interface_mesh):
    # If the mesh has no boundary, outline() returns an empty Path3D
    path_obj = interface_mesh.outline()
    segments = path_obj.discrete

    if not segments or len(segments) == 0:
        print("DEBUG: No discrete segments found in outline.")
        return []  # Return empty list, not None

    results = []
    from trimesh.proximity import closest_point

    for seg in segments:
        _, _, face_indices = closest_point(interface_mesh, seg)
        path_normals = interface_mesh.face_normals[face_indices]

        results.append({
            'points': seg,
            'normals': path_normals
        })
    print(results)
    return results


def extract_ordered_perimeter(interface_mesh):
    if interface_mesh is None or len(interface_mesh.faces) == 0:
        return None, None

    path_obj = interface_mesh.outline()
    segments = path_obj.discrete
    if not segments: return None, None

    path_points = max(segments, key=len)

    # USE THE MODULE CALL HERE TOO
    from trimesh.proximity import closest_point
    _, _, face_indices = closest_point(interface_mesh, path_points)
    path_normals = interface_mesh.face_normals[face_indices]

    return path_points, path_normals


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
    # This returns the already-rotated Trimesh objects
    rotated_model, rotated_substrate, final_angles = get_user_orientation_gui(mesh_in, sub_in)

    print(f"\n[3/4] Orientation Locked at Euler Angles: {final_angles}")

    # ---------------------------------------------------------
    # C. Geometric Extraction (The Cookie Cutter)
    # ---------------------------------------------------------
    # We call the cropping function on the ROTATED data
    master_skin = get_master_slicing_surface(rotated_model, rotated_substrate, padding=5)

    if master_skin is not None:
        print("[4/4] Initializing Dual Contouring Slicing Engine...")

        # --- VISUAL VALIDATION BEFORE SLICING ---
        # This confirms everything is ready for your 5-axis layers
        p = pv.Plotter()
        p.add_mesh(rotated_model, color='#3070B3', label='Target Model', opacity=0.6)
        p.add_mesh(master_skin, color='green', label='Master Slicing Surface (Layer 0)')
        p.add_legend()
        p.show(title="Final Validation: Pre-Slicing Check")

        # ---------------------------------------------------------
        # TRIGGER YOUR DUAL CONTOURING HERE
        # ---------------------------------------------------------
        # layers = your_slicer.generate_conformal_layers(base_surface=master_skin, target=rotated_model)

        print("\nSlicing Complete. Exporting G-Code...")
    else:
        print("[CRITICAL ERROR] Master skin could not be generated. Aborting.")

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