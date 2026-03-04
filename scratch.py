

import trimesh
import pyvista as pv
import numpy as np
import os

import trimesh
import numpy as np


def get_layer_zero(print_model_path, substrate_path, dist_tol=0.5, opposing_threshold=-0.8):
    """
    Extracts the contact skin using Relative Opposing Normals.
    Even if the substrate is completely vertical, it will find the contact skin.
    """
    print("Loading explicit B-Rep meshes...")
    print_model = trimesh.load_mesh(print_model_path)
    substrate = trimesh.load_mesh(substrate_path)

    print("Running Opposing-Normal Contact Extraction...")

    # 1. Get centers and find closest substrate points
    face_centers = print_model.triangles_center
    closest_points, distances, substrate_face_ids = trimesh.proximity.closest_point(
        substrate, face_centers
    )

    # FILTER 1: DISTANCE
    # Expanded to 0.5mm to aggressively swallow CAD chordal deviation gapsgit remote add origin git@github.com:bsnbipen/b_rep_confromal.git
    is_close_enough = distances <= dist_tol

    # FILTER 2: OPPOSING NORMALS
    pm_normals = print_model.face_normals
    # Get the exact normal of the substrate face that sits directly under the print model face
    sub_normals = substrate.face_normals[substrate_face_ids]

    # Calculate the dot product between the two faces.
    # If they face directly towards each other, dot product = -1.0
    # If they are perpendicular (like a side wall hitting a floor), dot product = 0.0
    # We use sum(a * b, axis=1) for fast row-wise dot products in numpy
    dot_products = np.sum(pm_normals * sub_normals, axis=1)

    # Accept faces that are pointing strongly towards the substrate
    is_opposing = dot_products < opposing_threshold

    # COMBINE FILTERS
    valid_face_indices = np.where(is_close_enough & is_opposing)[0]

    if len(valid_face_indices) == 0:
        print("Error: No valid contact faces found. Check CAD assembly.")
        return None, None, None

    # Extract the pristine B-Rep skin
    layer_zero_surface = print_model.submesh([valid_face_indices], append=True)

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
    # 1. Define your file paths
    # ---------------------------------------------------------
    base_dir=r"C:\Users\bb237\OneDrive - The University of Akron\Projects\5 Axis Sys\stil files\Curve"
    model_filename=r"Assem1 - Part3-1.stl"
    substrate_filename=r"Assem1 - year_3_3_cm_radius v1.stl"

    model_file = os.path.join(base_dir, model_filename)
    substrate_file = os.path.join(base_dir, substrate_filename)


    # 2. Extract Layer 0
    layer_0_trimesh, print_model, substrate = get_layer_zero(model_file, substrate_file)

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
    layer_0_pv = pv.wrap(layer_0_trimesh)
    substrate_pv = pv.read(substrate_file)

    p = pv.Plotter()
    p.add_mesh(substrate_pv, color='lightgray', opacity=0.5, label="Substrate")
    p.add_mesh(layer_0_pv, color='red', show_edges=True, line_width=2, label="Layer 0 (Contact)")

    # Plot our newly generated offset layers with different colors
    colors = ['blue', 'green', 'orange']
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

    p.add_legend()
    p.show()


if __name__ == "__main__":
    main()