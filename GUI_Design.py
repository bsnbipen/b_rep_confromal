import tkinter as tk
from tkinter import ttk
import pyvista as pv
import numpy as np
import trimesh
import trimesh.transformations as tf
import threading
from trimesh.proximity import closest_point
from scratch import *

def ensure_trimesh(obj):
    if isinstance(obj, str):
        return trimesh.load(obj)
    return obj

def get_user_orientation_gui(print_model_in, substrate_in):
    # --- 1. INITIAL GEOMETRY SETUP ---
    try:
        tm_pm_orig = ensure_trimesh(print_model_in)
        tm_sub_orig = ensure_trimesh(substrate_in)

        all_pts = np.vstack([tm_pm_orig.vertices, tm_sub_orig.vertices])
        model_size = np.ptp(all_pts, axis=0).max()
        GRID_STEP = 5.0 if model_size < 75 else 10.0

        max_v = (np.ceil(np.abs(all_pts).max() / GRID_STEP) * GRID_STEP) + GRID_STEP
        fixed_bounds = [-max_v, max_v, -max_v, max_v, -max_v, max_v]
    except Exception as e:
        print(f"Loading Error: {e}")
        return print_model_in, substrate_in, (0, 0, 0)

    # UI State
    state = {'x': 0, 'y': 0, 'z': 0}
    ui_settings = {
        'show_model_right': True,
        'show_core_right': True,
        'show_normals_view3': True,
    }

    # Computed geometry assets
    slice_assets = {
        "skin_tm": None,        # green substrate envelope
        "skin_pv": None,

        "core_tm": None,        # red strict contact core
        "core_pv": None,

        "roi_pv": None,         # yellow/black ROI box

        "patch_tm": None,       # blue model-side first-layer patch
        "patch_pv": None,

        "perimeter_loops": [],  # ordered loops for View 4
        "debug": None
    }

    def clear_slice_assets():
        slice_assets["skin_tm"] = None
        slice_assets["skin_pv"] = None
        slice_assets["core_tm"] = None
        slice_assets["core_pv"] = None
        slice_assets["roi_pv"] = None
        slice_assets["patch_tm"] = None
        slice_assets["patch_pv"] = None
        slice_assets["perimeter_loops"] = []
        slice_assets["debug"] = None

    p = pv.Plotter(shape=(2, 2), window_size=[1200, 900])
    p.set_background('white', all_renderers=True)

    def get_rotated_trimeshes():
        """
        Simulates a 5-axis rotary table by pivoting all rotations
        around the substrate's centroid.
        """
        rx, ry, rz = np.radians(state['x']), np.radians(state['y']), np.radians(state['z'])

        pm = tm_pm_orig.copy()
        sub = tm_sub_orig.copy()

        pivot = tm_sub_orig.centroid

        matrix_x = trimesh.transformations.rotation_matrix(rx, [1, 0, 0], point=pivot)
        matrix_y = trimesh.transformations.rotation_matrix(ry, [0, 1, 0], point=pivot)
        matrix_z = trimesh.transformations.rotation_matrix(rz, [0, 0, 1], point=pivot)

        full_matrix = trimesh.transformations.concatenate_matrices(matrix_z, matrix_y, matrix_x)

        pm.apply_transform(full_matrix)
        sub.apply_transform(full_matrix)

        return pm, sub

    def cmd_generate_slice():
        clear_slice_assets()

        tm_pm, tm_sub = get_rotated_trimeshes()
        layer_h = float(layer_height_var.get())

        # 1. Get substrate-side working surface
        skin, roi_box, core_mesh = get_layer_zero(tm_pm, tm_sub)

        if skin is not None and len(skin.faces) > 0:
            slice_assets["skin_tm"] = skin
            slice_assets["skin_pv"] = pv.wrap(skin)

            # 2. Get model-side first-layer patch + perimeter
            patch_mesh, perimeter_loops, debug = extract_first_layer_intersection_perimeter(
                layer_zero_surface=skin,
                print_model=tm_pm,
                layer_height=layer_h,
                gap_tol=0.10,
                euclid_tol=layer_h + 0.25,
                min_component_faces=20,
                keep_largest_only=True,
            )

            if patch_mesh is not None and len(patch_mesh.faces) > 0:
                slice_assets["patch_tm"] = patch_mesh
                slice_assets["patch_pv"] = pv.wrap(patch_mesh)

            slice_assets["perimeter_loops"] = perimeter_loops
            slice_assets["debug"] = debug

        if core_mesh is not None and len(core_mesh.faces) > 0:
            slice_assets["core_tm"] = core_mesh
            slice_assets["core_pv"] = pv.wrap(core_mesh)

        if roi_box is not None and len(roi_box.faces) > 0:
            slice_assets["roi_pv"] = pv.wrap(roi_box)

        update_3d_view()

    def update_3d_view():
        tm_pm, tm_sub = get_rotated_trimeshes()
        pm_pv, sub_pv = pv.wrap(tm_pm), pv.wrap(tm_sub)

        # =====================================================
        # VIEW 1: TOP-LEFT (Machine Config)
        # =====================================================
        p.subplot(0, 0)
        p.add_mesh(
            pm_pv, color='#3070B3', name='pm_l',
            show_edges=True, edge_color='black', line_width=0.5,
            reset_camera=False
        )
        p.add_mesh(
            sub_pv, color='#D0D0D0', opacity=0.3, name='sub_l',
            show_edges=True, edge_color='gray', line_width=0.5,
            reset_camera=False
        )
        p.add_text("1. Machine View", font_size=10, color='gray',
                   position='upper_left', name='txt_l')
        p.show_grid(color='#AAAAAA', bounds=fixed_bounds)

        # =====================================================
        # VIEW 2: TOP-RIGHT (Slice Preview)
        # =====================================================
        p.subplot(0, 1)

        legend_entries = []
        model_opacity = 1.0 if ui_settings['show_model_right'] else 0.0

        p.add_mesh(
            pm_pv, color='#3070B3', name='pm_r',
            opacity=model_opacity, show_edges=True, edge_color='black',
            reset_camera=False
        )
        legend_entries.append(["Print Model", "#3070B3"])

        if slice_assets['skin_pv'] is not None:
            p.add_mesh(
                slice_assets['skin_pv'], color='#2ECC71', name='skin_r',
                show_edges=True, edge_color='darkgreen',
                reset_camera=False
            )
            legend_entries.append(["Layer Zero Envelope", "#2ECC71"])
        else:
            try:
                p.remove_actor('skin_r', render=False)
            except Exception:
                pass

        if slice_assets['core_pv'] is not None:
            core_opacity = 1.0 if ui_settings['show_core_right'] else 0.0
            p.add_mesh(
                slice_assets['core_pv'], color='red', name='core_r',
                opacity=core_opacity, reset_camera=False
            )
            if ui_settings['show_core_right']:
                legend_entries.append(["Contact Core", "red"])
        else:
            try:
                p.remove_actor('core_r', render=False)
            except Exception:
                pass

        if slice_assets['roi_pv'] is not None:
            p.add_mesh(
                slice_assets['roi_pv'], color='black', name='roi_r',
                style='wireframe', opacity=0.3, line_width=1,
                reset_camera=False
            )
            legend_entries.append(["ROI Boundary", "black"])
        else:
            try:
                p.remove_actor('roi_r', render=False)
            except Exception:
                pass

        if legend_entries:
            p.add_legend(
                legend_entries, bcolor='white', border=True,
                size=(0.24, 0.24), name='leg_r'
            )

        p.add_text("2. Interaction Preview", font_size=10, color='gray',
                   position='upper_left', name='txt_m')
        p.show_grid(color='#AAAAAA', bounds=fixed_bounds)

        # =====================================================
        # VIEW 3: BOTTOM-LEFT (Isolated Detail)
        # =====================================================
        p.subplot(1, 0)

        # Clear old normals actor if needed
        try:
            p.remove_actor('skin_normals_iso', render=False)
        except Exception:
            pass

        if slice_assets['skin_pv'] is not None and slice_assets['skin_tm'] is not None:
            p.add_mesh(
                slice_assets['skin_pv'],
                color='#2ECC71',
                name='skin_iso',
                show_edges=True,
                edge_color='darkgreen'
            )
            p.add_text(
                "3. Surface Detail", font_size=10, color='darkgreen',
                position='upper_left', name='txt_iso'
            )

            if ui_settings.get('show_normals_view3', True):
                skin_tm = slice_assets['skin_tm']

                # Triangle centers and face normals
                centers = skin_tm.triangles_center
                normals = skin_tm.face_normals

                # Build a PyVista point cloud at triangle centers
                normals_pd = pv.PolyData(centers)
                normals_pd["normals"] = normals

                # Create arrow glyphs
                arrows = normals_pd.glyph(
                    orient="normals",
                    scale=False,
                    factor=2.0  # adjust this size as needed
                )

                p.add_mesh(
                    arrows,
                    color='red',
                    name='skin_normals_iso'
                )

            p.reset_camera(render=False)
        else:
            try:
                p.remove_actor('skin_iso', render=False)
            except Exception:
                pass
            try:
                p.remove_actor('txt_iso', render=False)
            except Exception:
                pass
            try:
                p.remove_actor('skin_normals_iso', render=False)
            except Exception:
                pass
        # =====================================================
        # VIEW 4: BOTTOM-RIGHT (First-Layer Perimeter / Toolpath)
        # =====================================================
        p.subplot(1, 1)

        # Clear old dynamic actors in View 4
        for actor_name in ['v4_skin', 'v4_patch', 'txt_stats', 'txt_v4_err', 'txt_v4_title']:
            try:
                p.remove_actor(actor_name, render=False)
            except Exception:
                pass

        for i in range(100):
            for prefix in ['perim_pts_', 'perim_norms_']:
                try:
                    p.remove_actor(f'{prefix}{i}', render=False)
                except Exception:
                    pass

        p.add_text(
            "4. First-Layer Perimeter", font_size=10, color='gray',
            position='upper_left', name='txt_v4_title'
        )

        if slice_assets['skin_pv'] is not None:
            p.add_mesh(
                slice_assets['skin_pv'], color='#2ECC71', name='v4_skin',
                opacity=0.20, show_edges=True, edge_color='darkgreen',
                reset_camera=False
            )

        if slice_assets['patch_pv'] is not None:
            p.add_mesh(
                slice_assets['patch_pv'], color='#3498DB', name='v4_patch',
                opacity=0.40, show_edges=True, edge_color='navy',
                reset_camera=False
            )

            path_data_list = slice_assets.get('perimeter_loops', [])

            if path_data_list:
                total_len = 0.0

                for i, path_data in enumerate(path_data_list):
                    pts = path_data['points']
                    norms = path_data['normals']

                    loop_pv = pv.PolyData(pts)
                    loop_pv.point_data["normals"] = norms

                    # Yellow perimeter points
                    p.add_mesh(
                        loop_pv, color='yellow', point_size=10.0,
                        render_points_as_spheres=True, name=f'perim_pts_{i}',
                        reset_camera=False
                    )

                    # Red nozzle normal arrows
                    arrows = loop_pv.glyph(orient="normals", scale=False, factor=3.0)
                    p.add_mesh(
                        arrows, color='red', name=f'perim_norms_{i}',
                        reset_camera=False
                    )

                    if len(pts) > 1:
                        total_len += np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
                        if path_data.get('is_closed', True):
                            total_len += np.linalg.norm(pts[-1] - pts[0])

                p.add_text(
                    f"Total Path: {total_len:.2f} mm",
                    position='lower_right', font_size=9, name='txt_stats'
                )
                p.reset_camera(render=False)
            else:
                p.add_text(
                    "No Boundary Found", position='upper_left',
                    color='orange', name='txt_v4_err'
                )
        else:
            p.add_text(
                "Generate Slice First", position='upper_left',
                color='gray', name='txt_v4_err'
            )

        p.render()

    # =====================================================
    # TKINTER UI
    # =====================================================
    root = tk.Tk()
    root.title("5-Axis Master Slicer | B-Rep Pipeline")
    root.geometry("380x880")
    root.attributes('-topmost', True)

    frame = ttk.Frame(root, padding="20")
    frame.pack(expand=True, fill="both")

    ttk.Label(frame, text="GEOMETRIC EXTRACTION", font=('Arial', 10, 'bold')).pack(pady=(5, 5))

    layer_height_var = tk.DoubleVar(value=0.40)

    ttk.Label(frame, text="Layer Height (mm)", font=('Arial', 9)).pack(pady=(5, 2))
    ttk.Entry(frame, textvariable=layer_height_var).pack(fill='x', pady=(0, 8))

    ttk.Button(frame, text="GENERATE SLICE (CURRENT ORIENT)", command=cmd_generate_slice).pack(fill='x', pady=5)

    ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=15)

    def create_axis_group(axis):
        ttk.Label(frame, text=f"{axis.upper()} AXIS", font=('Arial', 10, 'bold')).pack(pady=(10, 2))
        btn_f = ttk.Frame(frame)
        btn_f.pack(fill='x')

        def rot(v):
            state[axis] = (state[axis] + v) % 360
            clear_slice_assets()
            update_3d_view()

        ttk.Button(btn_f, text="+90°", command=lambda: rot(90)).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(btn_f, text="-90°", command=lambda: rot(-90)).pack(side='left', expand=True, fill='x', padx=2)

    for a in ['x', 'y', 'z']:
        create_axis_group(a)

    ttk.Label(frame, text="LAYER VISIBILITY", font=('Arial', 10, 'bold')).pack(pady=(15, 5))

    model_var = tk.BooleanVar(value=True)
    normals_var = tk.BooleanVar(value=True)

    def toggle_normals_view3():
        ui_settings['show_normals_view3'] = normals_var.get()
        update_3d_view()

    normals_f = ttk.Frame(frame)
    normals_f.pack(fill='x')
    ttk.Checkbutton(
        normals_f,
        text="Show Surface Normals (View 3)",
        variable=normals_var,
        command=toggle_normals_view3
    ).pack(side='left', padx=10)


    def toggle_model():
        ui_settings['show_model_right'] = model_var.get()
        update_3d_view()

    model_f = ttk.Frame(frame)
    model_f.pack(fill='x')
    ttk.Checkbutton(
        model_f, text="Show Print Model",
        variable=model_var, command=toggle_model
    ).pack(side='left', padx=10)

    core_var = tk.BooleanVar(value=True)

    def toggle_core():
        ui_settings['show_core_right'] = core_var.get()
        update_3d_view()

    core_f = ttk.Frame(frame)
    core_f.pack(fill='x')
    ttk.Checkbutton(
        core_f, text="Show Red Core",
        variable=core_var, command=toggle_core
    ).pack(side='left', padx=10)

    final_res = {"pm": None, "sub": None}

    def on_finish():
        pm_f, sub_f = get_rotated_trimeshes()
        final_res["pm"], final_res["sub"] = pm_f, sub_f
        root.quit()
        root.destroy()
        p.close()

    ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=30)
    ttk.Button(frame, text="CONFIRM & START DUAL CONTOURING", command=on_finish).pack(fill='x', pady=10)

    def launch():
        # Initialize all four quadrants
        active_quadrants = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for r, c in active_quadrants:
            p.subplot(r, c)
            p.show_grid(bounds=fixed_bounds)
            p.reset_camera(bounds=fixed_bounds)

        update_3d_view()
        p.show(interactive_update=True)

    root.after(100, launch)
    root.mainloop()

    return final_res["pm"], final_res["sub"], (state['x'], state['y'], state['z'])

def orient_mesh(mesh, axis, angle_degrees, pivot=[0, 0, 0]):
    """
    Permanently rotates a trimesh object around a specific global axis (X, Y, or Z).
    """
    print(f"Applying mathematical rotation: {angle_degrees} degrees around the {axis.upper()}-axis...")

    # 1. Convert human degrees (from your GUI slider) to mathematical radians
    angle_rad = np.radians(angle_degrees)

    # 2. Define the mathematical vector for the chosen axis
    if axis.upper() == 'X':
        direction = [1, 0, 0]
    elif axis.upper() == 'Y':
        direction = [0, 1, 0]
    elif axis.upper() == 'Z':
        direction = [0, 0, 1]
    else:
        print("Error: Axis must be 'X', 'Y', or 'Z'.")
        return mesh

    # 3. Generate the 4x4 Transformation Matrix
    # This matrix tells every single vertex in the mesh exactly where to move
    matrix = tf.rotation_matrix(angle_rad, direction, point=pivot)

    # 4. Apply the rotation directly to the mesh data
    mesh.apply_transform(matrix)

    return mesh