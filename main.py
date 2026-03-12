import os
import pickle
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
from cache_utils import save_cache, load_cache, cache_exists

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

def show_dc_solution(
    result,
    point_size=10.0,
    cube_center_size=5.0,
    show_print_model=True,
    show_substrate=True,
    show_layer_zero=True,
    show_debug_roi=True,
    show_dc_roi=True,
):
    plotter = pv.Plotter()
    plotter.set_background("white")

    print_model = result.get("print_model", None)
    substrate = result.get("substrate", None)
    layer_zero_surface = result.get("layer_zero_surface", None)
    debug_roi_box = result.get("debug_roi_box", None)

    roi_min = result.get("roi_min", None)
    roi_max = result.get("roi_max", None)

    active_cubes = result.get("active_cubes", [])
    dual_vertices = result.get("dual_vertices", [])

    # A. Print model
    if show_print_model and print_model is not None and len(print_model.faces) > 0:
        plotter.add_mesh(
            pv.wrap(print_model),
            color="#3070B3",
            opacity=0.35,
            show_edges=True,
            edge_color="black",
            label="Print Model"
        )

    # B. Substrate
    if show_substrate and substrate is not None and len(substrate.faces) > 0:
        plotter.add_mesh(
            pv.wrap(substrate),
            color="lightgray",
            opacity=0.18,
            show_edges=True,
            edge_color="gray",
            label="Substrate"
        )

    # C. Layer-zero surface
    if show_layer_zero and layer_zero_surface is not None and len(layer_zero_surface.faces) > 0:
        plotter.add_mesh(
            pv.wrap(layer_zero_surface),
            color="#2ECC71",
            opacity=0.85,
            show_edges=True,
            edge_color="darkgreen",
            label="Layer 0 Surface"
        )

    # D. Original debug ROI box
    if show_debug_roi and debug_roi_box is not None:
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

    # E. Tight DC ROI box
    if show_dc_roi and roi_min is not None and roi_max is not None:
        roi_box = pv_box_from_bounds(np.asarray(roi_min), np.asarray(roi_max))
        plotter.add_mesh(
            roi_box,
            color="magenta",
            style="wireframe",
            opacity=0.35,
            line_width=2,
            label="DC ROI"
        )

    # F. Active cube centers
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

    # G. Dual contour solution points
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
def make_cache_dir(base_dir):
    cache_dir = os.path.join(base_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def make_layer_zero_cache_path(cache_dir, layer_height, voxel_size):
    return os.path.join(
        cache_dir,
        f"layer_zero_h{layer_height:.3f}_vox{voxel_size:.3f}.pkl"
    )


def make_dc_cache_path(cache_dir, layer_height, voxel_size, mode_name="real"):
    return os.path.join(
        cache_dir,
        f"dc_{mode_name}_h{layer_height:.3f}_vox{voxel_size:.3f}.pkl"
    )
def main(
    model_file,
    substrate_file,
    layer_height=0.4,
    voxel_size=0.2,
    roi_extra_margin=0.0,
    export_dual_points=False,
    dual_points_out="layer1_dual_points.ply",
    use_plane_test=False,
    use_tilted_plane_test=False,
    use_cache=True,
    rebuild_cache=False,
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
    base_dir = os.path.dirname(model_file)
    cache_dir = make_cache_dir(base_dir)

    layer_zero_cache_file = make_layer_zero_cache_path(
        cache_dir=cache_dir,
        layer_height=layer_height,
        voxel_size=voxel_size,
    )

    print("[2/6] Getting layer_zero_surface...")

    if use_cache and (not rebuild_cache) and cache_exists(layer_zero_cache_file):
        cached_lz = load_cache(layer_zero_cache_file)
        layer_zero_surface = cached_lz["layer_zero_surface"]
        debug_roi_box = cached_lz["debug_roi_box"]
        core_mesh = cached_lz["core_mesh"]
    else:
        print("       Computing layer_zero_surface from scratch...")
        layer_zero_surface, debug_roi_box, core_mesh = get_layer_zero(print_model, substrate)

        if layer_zero_surface is None or len(layer_zero_surface.faces) == 0:
            raise RuntimeError("get_layer_zero returned an empty layer_zero_surface.")

        if use_cache:
            save_cache(
                {
                    "layer_zero_surface": layer_zero_surface,
                    "debug_roi_box": debug_roi_box,
                    "core_mesh": core_mesh,
                },
                layer_zero_cache_file
            )

    if layer_zero_surface is None or len(layer_zero_surface.faces) == 0:
        raise RuntimeError("layer_zero_surface is empty.")
    print(f"       Layer-zero faces : {len(layer_zero_surface.faces)}")
    if core_mesh is not None:
        print(f"       Core faces       : {len(core_mesh.faces)}")

    # ---------------------------------------------------------
    # C. Build field evaluator for Layer 1
    # ---------------------------------------------------------
    print("[3/6] Building conformal field evaluator for Layer 1...")
    if use_plane_test:
        print("[3/6] Using PlaneFieldEvaluator for fast DC testing...")
        field_evaluator = PlaneFieldEvaluator(target_height=layer_height)

    elif use_tilted_plane_test:
        print("[3/6] Using TiltedPlaneFieldEvaluator for fast DC testing...")
        field_evaluator = TiltedPlaneFieldEvaluator(
            offset=0.8,  # target plane position
            ax=0.2,
            ay=0.0,
            az=1.0
        )

    else:
        print("[3/6] Building conformal field evaluator for Layer 1...")
        field_evaluator = ConformalLayerFieldEvaluator(
            layer_zero_surface=layer_zero_surface,
            print_model_vertices=print_model.vertices,
            print_model_faces=print_model.faces,
            target_height=layer_height,
            patch_margin=3.0 * layer_height,
            model_band=2 * layer_height,
        )

    if use_plane_test:
        dc_mode_name = "plane"
    elif use_tilted_plane_test:
        dc_mode_name = "tilted_plane"
    else:
        dc_mode_name = "real"
    # ---------------------------------------------------------
    # D. Build voxel ROI and cube centers
    # ---------------------------------------------------------
    print("[4/6] Building ROI voxel grid...")
    roi_min, roi_max = build_tight_roi_bounds_from_layer_zero(
        layer_zero_surface=layer_zero_surface,
        layer_height=layer_height,
        voxel_size=voxel_size,
        xy_margin=max(voxel_size, 0.8 * layer_height),
        z_margin=max(voxel_size, 1.2 * layer_height),
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
    dc_cache_file = make_dc_cache_path(
        cache_dir=cache_dir,
        layer_height=layer_height,
        voxel_size=voxel_size,
        mode_name=dc_mode_name,
    )

    print("[6/6] Solving QEF for active cubes...")

    if use_cache and (not rebuild_cache) and cache_exists(dc_cache_file):
        cached_dc = load_cache(dc_cache_file)
        dual_vertices = cached_dc["dual_vertices"]
        cube_map = cached_dc["cube_map"]
        normal_map = cached_dc["normal_map"]
    else:
        dual_vertices, cube_map, normal_map = qef_solution(active_cubes, field_evaluator)

        if use_cache:
            save_cache(
                {
                    "dual_vertices": dual_vertices,
                    "cube_map": cube_map,
                    "normal_map": normal_map,
                },
                dc_cache_file
            )

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

class PlaneFieldEvaluator:
    def __init__(self, target_height=0.4):
        self.target_height = target_height

    def evaluate(self, points):
        pts = np.asarray(points, dtype=float)

        # zero iso-surface is plane z = target_height
        values = pts[:, 2] - self.target_height

        normals = np.tile(np.array([0.0, 0.0, 1.0]), (len(pts), 1))
        valid = np.ones(len(pts), dtype=bool)

        return values, normals, valid

def debug_dc_result(result, n_show_cubes=5, n_show_dual=10):
    """
    Print a few useful debug values for DC sanity-checking.
    Best for plane-test or early field debugging.
    """
    active_cubes = result.get("active_cubes", [])
    dual_vertices = result.get("dual_vertices", [])

    print("\n========== DC DEBUG SUMMARY ==========")
    print(f"Number of active cubes : {len(active_cubes)}")
    print(f"Number of dual vertices: {len(dual_vertices)}")

    # -------------------------------------------------
    # A. Show a few active cubes and their corner values
    # -------------------------------------------------
    if len(active_cubes) > 0:
        c0 = active_cubes[0]
        print("\n--- First active cube: real field detail ---")
        print("center:", np.round(c0.center, 6))
        print("corner values:", np.round(c0.sdf_values, 6))
        print("corner valid :", c0.corner_valid.astype(int) if c0.corner_valid is not None else None)

        if c0._intersection_points is not None:
            print("intersection z-range:",
                  np.round(c0._intersection_points[:, 2].min(), 6),
                  np.round(c0._intersection_points[:, 2].max(), 6))

    # -------------------------------------------------
    # B. Show dual vertex z-stats
    # -------------------------------------------------
    if len(dual_vertices) > 0:
        dual_pts = np.asarray(dual_vertices, dtype=float)

        print("\n--- Dual vertex statistics ---")
        print("x min/max:", np.round(dual_pts[:, 0].min(), 6), np.round(dual_pts[:, 0].max(), 6))
        print("y min/max:", np.round(dual_pts[:, 1].min(), 6), np.round(dual_pts[:, 1].max(), 6))
        print("z min/max:", np.round(dual_pts[:, 2].min(), 6), np.round(dual_pts[:, 2].max(), 6))
        print("z mean   :", np.round(dual_pts[:, 2].mean(), 6))

        print(f"\n--- First {min(n_show_dual, len(dual_pts))} dual vertices ---")
        print(np.round(dual_pts[:n_show_dual], 6))

    print("======================================\n")

    if len(dual_vertices) > 0:
        dual_pts = np.asarray(dual_vertices, dtype=np.float64)

        finite_rows = np.all(np.isfinite(dual_pts), axis=1)
        n_bad = np.sum(~finite_rows)

        print("\n--- Dual vertex finiteness check ---")
        print(f"finite rows : {np.sum(finite_rows)}")
        print(f"bad rows    : {n_bad}")

        if n_bad > 0:
            bad_idx = np.where(~finite_rows)[0][:10]
            print("first bad indices:", bad_idx)
            print("bad rows:")
            print(dual_pts[bad_idx])

    field_evaluator = result.get("field_evaluator", None)
    if field_evaluator is not None and len(dual_vertices) > 0:
        dual_pts = np.asarray(dual_vertices, dtype=float)
        try:
            vals, _, _ = field_evaluator.evaluate(dual_pts)
            print("\n--- Dual vertex field residuals ---")
            print("value min/max:", np.round(vals.min(), 8), np.round(vals.max(), 8))
            print("value mean   :", np.round(vals.mean(), 8))
        except Exception as e:
            print(f"[WARN] Could not evaluate field residuals on dual points: {e}")

def debug_active_cube_histograms(result):
    active_cubes = result.get("active_cubes", [])

    valid_corner_counts = []
    intersection_counts = []

    for cube in active_cubes:
        if cube.corner_valid is not None:
            valid_corner_counts.append(int(np.sum(cube.corner_valid)))
        else:
            valid_corner_counts.append(0)

        if cube._intersection_points is not None:
            intersection_counts.append(len(cube._intersection_points))
        else:
            intersection_counts.append(0)

    print("\n--- Active cube valid-corner histogram ---")
    for k in range(9):
        print(f"{k} valid corners :", sum(v == k for v in valid_corner_counts))

    print("\n--- Active cube intersection-count histogram ---")
    max_i = max(intersection_counts) if len(intersection_counts) > 0 else 0
    for k in range(max_i + 1):
        print(f"{k} intersections :", sum(v == k for v in intersection_counts))


class TiltedPlaneFieldEvaluator:
    def __init__(self, offset=0.8, ax=0.2, ay=0.0, az=1.0, debug=False):
        self.offset = float(offset)
        self.debug = debug

        n = np.array([ax, ay, az], dtype=np.float64)
        L = np.linalg.norm(n)
        if L < 1e-12:
            raise ValueError("Normal vector magnitude is too small.")

        self.normal = np.ascontiguousarray(n / L, dtype=np.float64)

        if self.debug:
            print("[TiltedPlaneFieldEvaluator]")
            print("  normal:", self.normal)
            print("  offset:", self.offset)

    def evaluate(self, points):
        pts = np.asarray(points, dtype=np.float64)

        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected points shape (N,3), got {pts.shape}")

        # Check field normal itself
        if not np.all(np.isfinite(self.normal)):
            raise ValueError(f"Field normal is not finite: {self.normal}")

        finite_mask = np.all(np.isfinite(pts), axis=1)

        values = np.full(len(pts), np.nan, dtype=np.float64)
        normals = np.tile(self.normal, (len(pts), 1))
        valid = finite_mask.copy()

        if np.any(finite_mask):
            good_pts = np.ascontiguousarray(pts[finite_mask], dtype=np.float64)

            # extra debug checks
            if self.debug:
                max_abs = np.max(np.abs(good_pts))
                if not np.isfinite(max_abs):
                    print("[WARN] Non-finite max_abs in good_pts")
                elif max_abs > 1e6:
                    print(f"[WARN] Very large coordinates detected: max_abs={max_abs}")

            with np.errstate(divide='raise', over='raise', invalid='raise'):
                try:
                    values_good = np.dot(good_pts, self.normal) - self.offset
                except FloatingPointError:
                    print("\n[ERROR] Floating-point issue inside TiltedPlaneFieldEvaluator.evaluate()")
                    print("points shape:", good_pts.shape)
                    print("normal:", self.normal)
                    print("offset:", self.offset)
                    print("points min:", np.nanmin(good_pts, axis=0))
                    print("points max:", np.nanmax(good_pts, axis=0))
                    print("sample points:")
                    print(good_pts[:10])
                    raise

            values[finite_mask] = values_good

        return values, normals, valid
if __name__ == "__main__":
    base_dir = r"/Users/bipendrabasnet/PycharmProjects/b_rep_confromal/stl_files/c"
    model_file = os.path.join(base_dir, r"UA - Part4^UA-1.stl")
    substrate_file = os.path.join(base_dir, r"UA - kickoff-1 10cm_dia_substrate_updated v3.step-1.stl")

    result = main(
        model_file=model_file,
        substrate_file=substrate_file,
        layer_height=0.4,
        voxel_size=0.2,
        export_dual_points=True,
        dual_points_out=os.path.join(base_dir, "layer1_dual_points.ply"),
        use_plane_test=False,
        use_tilted_plane_test=False,
        use_cache=True,
        rebuild_cache=False,   # set True when you want to force recompute
    )

    print("\n[Done] Main pipeline completed.")
    print(f"Active cubes : {len(result['active_cubes'])}")
    print(f"Dual vertices: {len(result['dual_vertices'])}")

    debug_dc_result(result, n_show_cubes=5, n_show_dual=10)
    debug_active_cube_histograms(result)

    show_dc_solution(
        result,
        show_print_model=True,
        show_substrate=True,
        show_layer_zero=True,
        show_debug_roi=False,
        show_dc_roi=True,
    )