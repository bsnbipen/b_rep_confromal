

import trimesh
import pyvista as pv
import numpy as np
import os

import trimesh
import numpy as np
from GUI_Design import *


def get_master_slicing_surface(print_model, substrate, padding=5):
    """
    Ensures the slice is 'On-Par' by calculating bounds of the
    ACTUALLY rotated trimesh object.
    """
    # 1. Force a re-calculation of the vertex-aligned Bounding Box
    # Using 'bounds' on a rotated trimesh gives the World-Axis-Aligned Box
    bounds = print_model.bounds
    min_b, max_b = bounds[0], bounds[1]

    # 2. Padded ROI based on the NEW orientation
    min_x, max_x = min_b[0] - padding, max_b[0] + padding
    min_y, max_y = min_b[1] - padding, max_b[1] + padding

    # 3. Apply vertical clipping
    # These planes act as the 'Projection' of the print head
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

    # 4. Critical: The "Upward" filter must align with the Print Vector (0,0,1)
    # We use a slightly higher threshold (0.05) to filter noise in complex B-Reps
    up_idx = np.where(master_surface.face_normals[:, 2] > 0.05)[0]

    if len(up_idx) == 0:
        return None, print_model, substrate

    master_skin = master_surface.submesh([up_idx], append=True)

    return master_skin, print_model, substrate


def get_layer_zero(print_model, substrate, dist_tol=0.5, opposing_threshold=-0.8,padding=0):
    """
    Extracts the contact skin using Relative Opposing Normals.
    Even if the substrate is completely vertical, it will find the contact skin.
    """
    #print("Loading explicit B-Rep meshes...")
    #print_model = trimesh.load_mesh(print_model_path)
    #substrate = trimesh.load_mesh(substrate_path)

    bounds = print_model.bounds
    min_x, max_x = bounds[0][0] - padding, bounds[1][0] + padding
    min_y, max_y = bounds[0][1] - padding, bounds[1][1] + padding

    vertical_planes = [
        ([min_x, 0, 0], [1, 0, 0]),
        ([max_x, 0, 0], [-1, 0, 0]),
        ([0, min_y, 0], [0, 1, 0]),
        ([0, max_y, 0], [0, -1, 0])
    ]

    # We create a smaller, temporary substrate for the heavy math
    cropped_substrate = substrate.copy()
    for origin, normal in vertical_planes:
        cropped_substrate = cropped_substrate.slice_plane(plane_origin=origin, plane_normal=normal)
        if cropped_substrate.is_empty:
            print("Error: Print model is entirely outside the substrate XY bounds.")
            return None, None, None


    print("Running Opposing-Normal Contact Extraction...")

    # 1. Get centers and find closest substrate points
    face_centers = cropped_substrate.triangles_center
    closest_points, distances, pm_face_ids = trimesh.proximity.closest_point(
        print_model, face_centers
    )

    # FILTER 1: DISTANCE
    # Expanded to 0.5mm to aggressively swallow CAD chordal deviation gapsgit remote add origin git@github.com:bsnbipen/b_rep_confromal.git
    is_close_enough = distances <= dist_tol

    # FILTER 2: OPPOSING NORMALS
    sub_normals = cropped_substrate.face_normals
    # Get the exact normal of the substrate face that sits directly under the print model face
    pm_normals = print_model.face_normals[pm_face_ids]

    # Calculate the dot product between the two faces.
    # If they face directly towards each other, dot product = -1.0
    # If they are perpendicular (like a side wall hitting a floor), dot product = 0.0
    # We use sum(a * b, axis=1) for fast row-wise dot products in numpy
    dot_products = np.sum(sub_normals * pm_normals, axis=1)

    # Accept faces that are pointing strongly towards the substrate
    is_opposing = dot_products < opposing_threshold

    # COMBINE FILTERS
    valid_face_indices = np.where(is_close_enough & is_opposing)[0]

    if len(valid_face_indices) == 0:
        print("Error: No valid contact faces found. Check CAD assembly.")
        return None, None, None

    # Extract the pristine B-Rep skin
    layer_zero_surface = cropped_substrate.submesh([valid_face_indices], append=True)

    return layer_zero_surface, print_model, substrate


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


def main():
    # ---------------------------------------------------------
    # A. Load your files (Initial state)
    # ---------------------------------------------------------
    base_dir = r"C:\Users\bb237\PycharmProjects\b_repp_offsetting\stl_files\ring"
    model_file = os.path.join(base_dir, r'cylinder_ring - ring_55mm_dia-1.STL')
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