import os
import numpy as np
import trimesh
import pyvista as pv
# ---------------------------------------------------------
# Adjust these imports to your actual file names
# ---------------------------------------------------------
# If get_layer_zero is in this same file, remove this import.
from scratch import get_layer_zero


from CUBE import *
from DC import qef_solution



def cube_has_iso_crossing(cube, iso_level=0.0, eps=1e-12):
    """
    Check whether a cube has a sign change around the target iso-level.
    """
    if cube.sdf_values is None:
        return False

    if cube.corner_valid is not None:
        valid_mask = cube.corner_valid
        if not np.any(valid_mask):
            return False
        vals = cube.sdf_values[valid_mask] - iso_level
    else:
        vals = cube.sdf_values - iso_level

    return (np.any(vals > eps) and np.any(vals < -eps))

def pv_box_from_bounds(bounds_min, bounds_max):
    center = 0.5 * (bounds_min + bounds_max)
    extents = bounds_max - bounds_min

    return pv.Cube(
        center=center,
        x_length=extents[0],
        y_length=extents[1],
        z_length=extents[2],
    )

def show_dc_solution(result, point_size=10.0, cube_center_size=5.0):
    """
    Visualize the current DC solution state.

    Parameters
    ----------
    result : dict
        Output dictionary from main()
    """
    plotter = pv.Plotter()
    plotter.set_background("white")

    print_model = result["print_model"]
    substrate = result["substrate"]
    layer_zero_surface = result["layer_zero_surface"]
    debug_roi_box = result.get("debug_roi_box", None)

    roi_min = result["roi_min"]
    roi_max = result["roi_max"]

    active_cubes = result["active_cubes"]
    dual_vertices = result["dual_vertices"]

    # ---------------------------------------------------------
    # A. Show print model
    # ---------------------------------------------------------
    if print_model is not None and len(print_model.faces) > 0:
        plotter.add_mesh(
            pv.wrap(print_model),
            color="#3070B3",
            opacity=0.35,
            show_edges=True,
            edge_color="black",
            label="Print Model"
        )

    # ---------------------------------------------------------
    # B. Show substrate
    # ---------------------------------------------------------
    if substrate is not None and len(substrate.faces) > 0:
        plotter.add_mesh(
            pv.wrap(substrate),
            color="lightgray",
            opacity=0.18,
            show_edges=True,
            edge_color="gray",
            label="Substrate"
        )

    # ---------------------------------------------------------
    # C. Show layer zero surface
    # ---------------------------------------------------------
    if layer_zero_surface is not None and len(layer_zero_surface.faces) > 0:
        plotter.add_mesh(
            pv.wrap(layer_zero_surface),
            color="#2ECC71",
            opacity=0.85,
            show_edges=True,
            edge_color="darkgreen",
            label="Layer 0 Surface"
        )

    # ---------------------------------------------------------
    # D. Show original debug ROI box if available
    # ---------------------------------------------------------
    if debug_roi_box is not None:
        try:
            plotter.add_mesh(
                pv.wrap(debug_roi_box),
                color="black",
                style="wireframe",
                opacity=0.15,
                line_width=1,
                label="Layer 0 ROI"
            )
        except Exception:
            pass

    # ---------------------------------------------------------
    # E. Show tight DC ROI box
    # ---------------------------------------------------------
    roi_box = pv_box_from_bounds(np.asarray(roi_min), np.asarray(roi_max))
    plotter.add_mesh(
        roi_box,
        color="magenta",
        style="wireframe",
        opacity=0.35,
        line_width=2,
        label="DC ROI"
    )

    # ---------------------------------------------------------
    # F. Show active cube centers
    # ---------------------------------------------------------
    if active_cubes is not None and len(active_cubes) > 0:
        centers = np.array([cube.center for cube in active_cubes], dtype=float)
        center_cloud = pv.PolyData(centers)
        plotter.add_mesh(
            center_cloud,
            color="orange",
            point_size=cube_center_size,
            render_points_as_spheres=True,
            label="Active Cube Centers"
        )

    # ---------------------------------------------------------
    # G. Show dual contour solution points
    # ---------------------------------------------------------
    if dual_vertices is not None and len(dual_vertices) > 0:
        dual_pts = np.asarray(dual_vertices, dtype=float)
        dual_cloud = pv.PolyData(dual_pts)
        plotter.add_mesh(
            dual_cloud,
            color="red",
            point_size=point_size,
            render_points_as_spheres=True,
            label="DC Dual Vertices"
        )

    plotter.add_legend()
    plotter.show_grid()
    plotter.show()


def build_tight_roi_bounds_from_layer_zero(
    layer_zero_surface,
    layer_height,
    voxel_size,
    xy_margin=None,
    z_margin=None,
):
    """
    Build a tighter local voxel ROI around layer_zero_surface for the first layer.

    Since we only need layer 1, we do not need a big margin.
    """

    lz_min, lz_max = layer_zero_surface.bounds

    # Tight XY margin: just enough to catch neighboring active cells
    if xy_margin is None:
        xy_margin = max(voxel_size, 0.5 * layer_height)

    # Tight Z margin: enough to include the target layer surface and a little slack
    if z_margin is None:
        z_margin = max(voxel_size, 1.5 * layer_height)

    roi_min = np.array([
        lz_min[0] - xy_margin,
        lz_min[1] - xy_margin,
        lz_min[2] - z_margin,
    ], dtype=float)

    roi_max = np.array([
        lz_max[0] + xy_margin,
        lz_max[1] + xy_margin,
        lz_max[2] + z_margin,
    ], dtype=float)

    return roi_min, roi_max

def build_roi_bounds_from_layer_zero(layer_zero_surface, layer_height, voxel_size, extra_margin=0.0):
    """
    Build a local voxel ROI around the layer_zero_surface.

    We enlarge it enough so the target layer surface can exist inside the ROI.
    """
    lz_min, lz_max = layer_zero_surface.bounds

    # conservative local margin
    margin = max(2.0 * layer_height, voxel_size) + extra_margin
    pad = np.array([margin, margin, margin], dtype=float)

    roi_min = lz_min - pad
    roi_max = lz_max + pad

    return roi_min, roi_max


def generate_cube_centers(roi_min, roi_max, voxel_size):
    """
    Generate cube centers for an axis-aligned voxel grid.
    """
    xs = np.arange(roi_min[0] + voxel_size / 2.0, roi_max[0], voxel_size)
    ys = np.arange(roi_min[1] + voxel_size / 2.0, roi_max[1], voxel_size)
    zs = np.arange(roi_min[2] + voxel_size / 2.0, roi_max[2], voxel_size)

    centers = []
    for x in xs:
        for y in ys:
            for z in zs:
                centers.append(np.array([x, y, z], dtype=float))

    return centers


def collect_active_cubes(centers, voxel_size, roi_min, roi_max, field_evaluator):
    """
    Create cubes, sample the field at corners, and keep only active cubes.
    """
    active_cubes = []
    all_cubes = []

    for i,center in enumerate(centers):
        #print(i)
        cube = Cube(
            center=center,
            voxel_size=voxel_size,
            min_bounds=roi_min,
            max_bounds=roi_max
        )

        cube.sample_corners(field_evaluator)

        if cube_has_iso_crossing(cube, iso_level=0.0):
           #print("finding solutions")
            active_cubes.append(cube)

        all_cubes.append(cube)

    return all_cubes, active_cubes


def export_dual_points_as_ply(dual_points, out_path):
    """
    Save dual vertices as a point cloud for quick inspection.
    """
    if dual_points is None or len(dual_points) == 0:
        print("[WARN] No dual points to export.")
        return

    pts = np.asarray(dual_points, dtype=float)
    cloud = trimesh.points.PointCloud(pts)
    cloud.export(out_path)
    print(f"[OK] Exported dual points to: {out_path}")


def main(
    model_file,
    substrate_file,
    layer_height=0.4,
    voxel_size=0.2,
    roi_extra_margin=0.0,
    export_dual_points=False,
    dual_points_out="layer1_dual_points.ply",
):
    # ---------------------------------------------------------
    # A. Load meshes
    # ---------------------------------------------------------
    print("[1/6] Loading meshes from disk...")
    print_model = trimesh.load_mesh(model_file)
    substrate = trimesh.load_mesh(substrate_file)

    print(f"       Print model faces: {len(print_model.faces)}")
    print(f"       Substrate faces  : {len(substrate.faces)}")

    # ---------------------------------------------------------
    # B. Compute layer-zero substrate patch
    # ---------------------------------------------------------
    print("[2/6] Computing layer_zero_surface...")
    layer_zero_surface, debug_roi_box, core_mesh = get_layer_zero(print_model, substrate)

    if layer_zero_surface is None or len(layer_zero_surface.faces) == 0:
        raise RuntimeError("get_layer_zero returned an empty layer_zero_surface.")

    print(f"       Layer-zero faces : {len(layer_zero_surface.faces)}")
    if core_mesh is not None:
        print(f"       Core faces       : {len(core_mesh.faces)}")

    # ---------------------------------------------------------
    # C. Build field evaluator for Layer 1
    # ---------------------------------------------------------
    print("[3/6] Building conformal field evaluator for Layer 1...")
    field_evaluator = ConformalLayerFieldEvaluator(
        layer_zero_surface=layer_zero_surface,
        print_model_vertices=print_model.vertices,
        print_model_faces=print_model.faces,
        target_height=layer_height,
        patch_margin=2.0 * layer_height,
        model_band=1.5 * layer_height,
    )

    # ---------------------------------------------------------
    # D. Build voxel ROI and cube centers
    # ---------------------------------------------------------
    print("[4/6] Building ROI voxel grid...")
    roi_min, roi_max = build_tight_roi_bounds_from_layer_zero(
        layer_zero_surface=layer_zero_surface,
        layer_height=layer_height,
        voxel_size=voxel_size,
        xy_margin=max(voxel_size, 0.5 * layer_height),
        z_margin=max(voxel_size, 1.5 * layer_height),
    )

    print(f"       ROI min: {roi_min}")
    print(f"       ROI max: {roi_max}")

    centers = generate_cube_centers(roi_min, roi_max, voxel_size)
    print(f"       Total cubes in ROI: {len(centers)}")

    # ---------------------------------------------------------
    # E. Sample cubes and keep active cubes for Layer 1
    # ---------------------------------------------------------
    print("[5/6] Sampling cubes and finding active layer-1 cells...")
    all_cubes, active_cubes = collect_active_cubes(
        centers=centers,
        voxel_size=voxel_size,
        roi_min=roi_min,
        roi_max=roi_max,
        field_evaluator=field_evaluator,
    )

    print(f"       Active cubes: {len(active_cubes)}")

    if len(active_cubes) == 0:
        print("[WARN] No active cubes found for layer 1.")
        return {
            "print_model": print_model,
            "substrate": substrate,
            "layer_zero_surface": layer_zero_surface,
            "debug_roi_box": debug_roi_box,
            "core_mesh": core_mesh,
            "field_evaluator": field_evaluator,
            "roi_min": roi_min,
            "roi_max": roi_max,
            "all_cubes": all_cubes,
            "active_cubes": active_cubes,
            "dual_vertices": [],
            "cube_map": {},
            "normal_map": {},
        }

    # ---------------------------------------------------------
    # F. Solve QEF for active cubes -> dual vertices for Layer 1
    # ---------------------------------------------------------
    print("[6/6] Solving QEF for active cubes...")
    dual_vertices, cube_map, normal_map = qef_solution(active_cubes, field_evaluator)

    print(f"       Dual vertices: {len(dual_vertices)}")

    if export_dual_points and len(dual_vertices) > 0:
        export_dual_points_as_ply(dual_vertices, dual_points_out)

    # ---------------------------------------------------------
    # Return everything useful for next step:
    # face stitching / perimeter extraction / visualization
    # ---------------------------------------------------------
    return {
        "print_model": print_model,
        "substrate": substrate,
        "layer_zero_surface": layer_zero_surface,
        "debug_roi_box": debug_roi_box,
        "core_mesh": core_mesh,
        "field_evaluator": field_evaluator,
        "roi_min": roi_min,
        "roi_max": roi_max,
        "all_cubes": all_cubes,
        "active_cubes": active_cubes,
        "dual_vertices": dual_vertices,
        "cube_map": cube_map,
        "normal_map": normal_map,
    }


if __name__ == "__main__":
    base_dir = r"C:\Users\bb237\PycharmProjects\b_repp_offsetting\stl_files\c"
    model_file = os.path.join(base_dir, r"UA - Part4^UA-1.stl")
    substrate_file = os.path.join(base_dir, r"UA - kickoff-1 10cm_dia_substrate_updated v3.step-1.stl")

    result = main(
        model_file=model_file,
        substrate_file=substrate_file,
        layer_height=0.8,
        voxel_size=0.4,
        export_dual_points=True,
        dual_points_out=os.path.join(base_dir, "layer1_dual_points.ply"),
    )

    print("\n[Done] Main pipeline completed.")
    print(f"Active cubes : {len(result['active_cubes'])}")
    print(f"Dual vertices: {len(result['dual_vertices'])}")

    show_dc_solution(result)