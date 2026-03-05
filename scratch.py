

import trimesh
import pyvista as pv
import numpy as np
import os

import trimesh
import numpy as np


def get_master_slicing_surface(print_model_path, substrate_path, padding=5):
    """
    Crops the Substrate using the XY Bounding Box of the Print Model.
    This creates a Master Slicing Surface wide enough to handle overhangs
    and inverted pyramids.
    """

    """
        Extracts the contact skin using Relative Opposing Normals.
        Even if the substrate is completely vertical, it will find the contact skin.
        """
    print("Loading explicit B-Rep meshes...")
    print_model = trimesh.load_mesh(print_model_path)
    substrate = trimesh.load_mesh(substrate_path)

    print("Running Opposing-Normal Contact Extraction...")

    print("Calculating Print Model AABB and cropping Substrate...")

    # 1. Get the bounding box of the Print Model
    # bounds[0] is [minX, minY, minZ], bounds[1] is [maxX, maxY, maxZ]
    bounds = print_model.bounds
    min_b = bounds[0]
    max_b = bounds[1]

    # 1. Apply the Padding to the X and Y bounds
    min_x = min_b[0] - padding
    max_x = max_b[0] + padding
    min_y = min_b[1] - padding
    max_y = max_b[1] + padding

    # 2. Define the 4 vertical slicing planes with the new padded coordinates
    vertical_planes = [
        ([min_x, 0, 0], [1, 0, 0]),  # Min X wall (pointing Right)
        ([max_x, 0, 0], [-1, 0, 0]),  # Max X wall (pointing Left)
        ([0, min_y, 0], [0, 1, 0]),  # Min Y wall (pointing Forward)
        ([0, max_y, 0], [0, -1, 0])  # Max Y wall (pointing Backward)
    ]

    master_surface = substrate.copy()

    # 4. Sequentially guillotine the substrate with the 4 planes
    for origin, normal in vertical_planes:
        master_surface = master_surface.slice_plane(plane_origin=origin, plane_normal=normal)

        if master_surface.is_empty:
            print("Error: Substrate cropping failed. Is the print model hovering outside the substrate XY bounds?")
            return None

    print("Filtering out the bottom base to isolate the top curved skin...")

    # We look at the Z-component of every triangle's normal vector.
    # > 0.01 guarantees we keep faces pointing UP, while throwing away
    # perfectly vertical walls (0.0) and the bottom flat base (-1.0).
    upward_facing_indices = np.where(master_surface.face_normals[:, 2] > 0.01)[0]

    if len(upward_facing_indices) == 0:
        print("Error: No upward-facing surface found after cropping.")
        return None

    # Extract only the pristine top skin
    master_skin = master_surface.submesh([upward_facing_indices], append=True)

    print(f"Success! Master Slicing Skin generated with {len(master_skin.faces)} faces.")


    return master_skin,print_model,substrate


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
    face_centers = substrate.triangles_center
    closest_points, distances, pm_face_ids = trimesh.proximity.closest_point(
        print_model, face_centers
    )

    # FILTER 1: DISTANCE
    # Expanded to 0.5mm to aggressively swallow CAD chordal deviation gapsgit remote add origin git@github.com:bsnbipen/b_rep_confromal.git
    is_close_enough = distances <= dist_tol

    # FILTER 2: OPPOSING NORMALS
    sub_normals = substrate.face_normals
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
    layer_zero_surface = substrate.submesh([valid_face_indices], append=True)

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
    base_dir=r'/Users/bipendrabasnet/PycharmProjects/b_rep_confromal/'
    model_filename=r'cylinder_ring - ring_55mm_dia-1.STL'
    substrate_filename=r"cylinder_ring - 50_mm_dia-1.stl"

    model_file = os.path.join(base_dir, model_filename)
    substrate_file = os.path.join(base_dir, substrate_filename)


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
    layer_0_pv = pv.wrap(layer_0_trimesh)
    #substrate_pv = pv.read(substrate_file)

    p = pv.Plotter()
    #p.add_mesh(substrate_pv, color='lightgray', opacity=0.5, label="Substrate")
    p.add_mesh(layer_0_pv, color='red', show_edges=True, line_width=2, label="Layer 0 (Contact)")

    # Plot our newly generated offset layers with different colors
    colors = ['blue', 'green', 'orange']
    """
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
    p.add_legend()
    p.show()


if __name__ == "__main__":
    main()