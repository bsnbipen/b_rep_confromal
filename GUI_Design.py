import tkinter as tk
from tkinter import ttk
import pyvista as pv
import numpy as np
import trimesh
import trimesh.transformations as tf
import threading
from trimesh.proximity import closest_point
from scratch import *


def get_user_orientation_gui(print_model_in, substrate_in):
    # --- 1. INITIAL GEOMETRY SETUP ---
    def ensure_trimesh(obj):
        if isinstance(obj, str): return trimesh.load(obj)
        return obj

    try:
        # Store original "Raw" meshes to prevent rotation drift
        tm_pm_orig = ensure_trimesh(print_model_in)
        tm_sub_orig = ensure_trimesh(substrate_in)

        # Dynamic Grid Scaling (Fits the grid to the model size)
        all_pts = np.vstack([tm_pm_orig.vertices, tm_sub_orig.vertices])
        model_size = np.ptp(all_pts, axis=0).max()
        GRID_STEP = 5.0 if model_size < 75 else 10.0

        max_v = (np.ceil(np.abs(all_pts).max() / GRID_STEP) * GRID_STEP) + GRID_STEP
        fixed_bounds = [-max_v, max_v, -max_v, max_v, -max_v, max_v]
    except Exception as e:
        print(f"Loading Error: {e}");
        return print_model_in, substrate_in, (0, 0, 0)

    # UI State tracking
    state = {'x': 0, 'y': 0, 'z': 0}
    ui_settings = {
        'show_model_right': True,
        'show_seeds_right': True  # <-- Add this line
    }
    slice_assets = {
        'skin_pv': None,  # The Green Surface
        'seeds_pv': None,  # The Red Surface
        'roi_pv': None  # The Yellow ROI Box
    }

    p = pv.Plotter(shape=(2, 2), window_size=[1200, 900])
    p.set_background('white', all_renderers=True)

    def get_rotated_trimeshes():
        """
        Simulates a 5-axis rotary table by pivoting all rotations
        around the substrate's centroid.
        """
        # Convert UI Degrees to Math Radians
        rx, ry, rz = np.radians(state['x']), np.radians(state['y']), np.radians(state['z'])

        pm = tm_pm_orig.copy()
        sub = tm_sub_orig.copy()

        # --- THE KEY FIX: Shared Pivot Point ---
        # Calculate the pivot once from the original substrate to avoid drift
        pivot = tm_sub_orig.centroid

        # Create individual rotation matrices around the pivot
        # We use concatenation to apply X, then Y, then Z in one clean step
        matrix_x = trimesh.transformations.rotation_matrix(rx, [1, 0, 0], point=pivot)
        matrix_y = trimesh.transformations.rotation_matrix(ry, [0, 1, 0], point=pivot)
        matrix_z = trimesh.transformations.rotation_matrix(rz, [0, 0, 1], point=pivot)

        # Combined Matrix: Z * Y * X
        full_matrix = trimesh.transformations.concatenate_matrices(matrix_z, matrix_y, matrix_x)

        # Apply the SAME transformation to BOTH meshes
        pm.apply_transform(full_matrix)
        sub.apply_transform(full_matrix)

        return pm, sub

    def cmd_generate_slice():
        # Get live meshes based on the slider/button rotations
        tm_pm, tm_sub = get_rotated_trimeshes()

        # Call the bulletproof function we refined
        skin, roi_box, seed_pts = get_layer_zero(tm_pm, tm_sub)

        # Wrap them for PyVista
        if skin:
            slice_assets['skin_pv'] = pv.wrap(skin)
        if seed_pts:
            slice_assets['seeds_pv'] = pv.wrap(seed_pts)
        if roi_box:
            slice_assets['roi_pv'] = pv.wrap(roi_box)

        update_3d_view()

    def update_3d_view():
        tm_pm, tm_sub = get_rotated_trimeshes()
        pm_pv, sub_pv = pv.wrap(tm_pm), pv.wrap(tm_sub)

        # --- VIEW 1: TOP-LEFT (Machine Config) ---
        p.subplot(0, 0)
        p.add_mesh(pm_pv, color='#3070B3', name='pm_l', show_edges=True, edge_color='black', line_width=0.5,
                   reset_camera=False)
        p.add_mesh(sub_pv, color='#D0D0D0', opacity=0.3, name='sub_l', show_edges=True, edge_color='gray',
                   line_width=0.5, reset_camera=False)
        p.add_text("1. Machine View", font_size=10, color='gray', position='upper_left', name='txt_l')
        p.show_grid(color='#AAAAAA', bounds=fixed_bounds)

        # --- VIEW 2: TOP-RIGHT (Slice Preview) ---
        p.subplot(0, 1)
        legend_entries = []
        model_opacity = 1.0 if ui_settings['show_model_right'] else 0.0
        p.add_mesh(pm_pv, color='#3070B3', name='pm_r', opacity=model_opacity, show_edges=True, edge_color='black',
                   reset_camera=False)
        legend_entries.append(["Print Model", "#3070B3"])

        if slice_assets['skin_pv'] is not None:
            p.add_mesh(slice_assets['skin_pv'], color='#2ECC71', name='skin_r', show_edges=True, edge_color='darkgreen',
                       reset_camera=False)
            legend_entries.append(["Unified Layer Zero", "#2ECC71"])
        else:
            try:
                p.remove_actor('skin_r', render=False)
            except:
                pass

        if slice_assets['seeds_pv'] is not None:
            seed_opacity = 1.0 if ui_settings['show_seeds_right'] else 0.0
            p.add_mesh(slice_assets['seeds_pv'], color='red', name='seeds_r', opacity=seed_opacity, reset_camera=False)
            if ui_settings['show_seeds_right']: legend_entries.append(["Contact Zone", "red"])
        else:
            try:
                p.remove_actor('seeds_r', render=False)
            except:
                pass

        if slice_assets['roi_pv'] is not None:
            p.add_mesh(slice_assets['roi_pv'], color='black', name='roi_r', style='wireframe', opacity=0.3,
                       line_width=1, reset_camera=False)
            legend_entries.append(["ROI Boundary", "black"])
        else:
            try:
                p.remove_actor('roi_r', render=False)
            except:
                pass

        if legend_entries:
            p.add_legend(legend_entries, bcolor='white', border=True, size=(0.2, 0.2), name='leg_r')

        p.add_text("2. Interaction Preview", font_size=10, color='gray', position='upper_left', name='txt_m')
        p.show_grid(color='#AAAAAA', bounds=fixed_bounds)

        # --- VIEW 3: BOTTOM-LEFT (Isolated Detail - BELOW VIEW 1) ---
        p.subplot(1, 0)  # Row 1, Column 0
        if slice_assets['skin_pv'] is not None:
            p.add_mesh(slice_assets['skin_pv'], color='#2ECC71', name='skin_iso', show_edges=True,
                       edge_color='darkgreen')
            p.add_text("3. Surface Detail", font_size=10, color='darkgreen', position='upper_left', name='txt_iso')
            p.reset_camera(render=False)  # Zoom in specifically on the layer
        else:
            try:
                p.remove_actor('skin_iso', render=False)
                p.remove_actor('txt_iso', render=False)
            except:
                pass

        # --- VIEW 4: BOTTOM-RIGHT (Final Integrated Logic) ---
        p.subplot(1, 1)

        try:
            from trimesh.proximity import closest_point

            # 1. Logic remains the same to find the contact area
            _, distances, _ = closest_point(tm_pm, tm_sub.triangles_center)
            contact_indices = np.where(distances < 0.05)[0]

            if len(contact_indices) > 0:
                interface_mesh = tm_sub.submesh([contact_indices], append=True)

                # --- IMPROVEMENT: Keep patch visible but faint to verify intersection ---
                p.add_mesh(pv.wrap(interface_mesh), color='#3498DB', name='diw_patch', opacity=0.3)

                # 2. Extract ALL Perimeters (Returns a LIST of dicts)
                path_data_list = extract_ordered_perimeter(interface_mesh)

                if path_data_list and isinstance(path_data_list, list):
                    total_len = 0

                    for i, path_data in enumerate(path_data_list):
                        pts = path_data['points']
                        norms = path_data['normals']

                        # Create PolyData for THIS specific loop
                        loop_pv = pv.PolyData(pts)
                        loop_pv.point_data["normals"] = norms

                        # Unique names prevent the inner ring from deleting the outer ring
                        name_pts = f'diw_pts_{i}'
                        name_norms = f'diw_norms_{i}'

                        # Render as points
                        p.add_mesh(loop_pv,
                                   color='yellow',
                                   point_size=12.0,  # Increased for visibility
                                   render_points_as_spheres=True,
                                   name=name_pts)

                        # Render the red arrows
                        arrows = loop_pv.glyph(orient="normals", scale=False, factor=3.0)
                        p.add_mesh(arrows, color='red', name=name_norms)

                        # Calculate path length including loop closure
                        if len(pts) > 1:
                            segment_dist = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
                            closure_dist = np.linalg.norm(pts[-1] - pts[0])
                            total_len += segment_dist + closure_dist

                    p.add_text(f"Total Path: {total_len:.2f} mm", position='lower_right', font_size=9, name='txt_stats')

                    # --- IMPROVEMENT: Force camera to find the new points ---
                    p.reset_camera()
                    try:
                        p.remove_actor('txt_v4_err', render=False)
                    except:
                        pass
                else:
                    # If patch exists but no loops found, it might be a closed manifold
                    p.add_text("Patch found, but no open boundaries.", position='upper_left', color='orange',
                               name='txt_v4_err')
            else:
                # Cleanup
                for i in range(10):
                    p.remove_actor(f'diw_pts_{i}', render=False)
                    p.remove_actor(f'diw_norms_{i}', render=False)
                p.remove_actor('diw_patch', render=False)
                p.add_text("No Contact Detected", position='upper_left', color='red', name='txt_v4_err')

        except Exception as e:
            print(f"Interface Generation Error: {e}")

        p.render()

    # --- 3. TKINTER UI ---
    root = tk.Tk()
    root.title(f"5-Axis Master Slicer | B-Rep Pipeline")
    root.geometry("380x850")
    root.attributes('-topmost', True)
    frame = ttk.Frame(root, padding="20")
    frame.pack(expand=True, fill="both")

    ttk.Label(frame, text="GEOMETRIC EXTRACTION", font=('Arial', 10, 'bold')).pack(pady=(5, 5))
    ttk.Button(frame, text="GENERATE SLICE (CURRENT ORIENT)", command=cmd_generate_slice).pack(fill='x', pady=5)

    ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=15)

    def create_axis_group(axis):
        ttk.Label(frame, text=f"{axis.upper()} AXIS", font=('Arial', 10, 'bold')).pack(pady=(10, 2))
        btn_f = ttk.Frame(frame)
        btn_f.pack(fill='x')

        def rot(v):
            state[axis] = (state[axis] + v) % 360

            # --- THE FIX: Clear EVERYTHING stale ---
            slice_assets['skin_pv'] = None
            slice_assets['seeds_pv'] = None
            slice_assets['roi_pv'] = None

            update_3d_view()

        ttk.Button(btn_f, text="+90°", command=lambda: rot(90)).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(btn_f, text="-90°", command=lambda: rot(-90)).pack(side='left', expand=True, fill='x', padx=2)

    for a in ['x', 'y', 'z']: create_axis_group(a)

    view_var = tk.BooleanVar(value=True)

    # ... (After your 'for a in ['x', 'y', 'z']: create_axis_group(a)' loop) ...

    # --- NEW LAYER VISIBILITY SECTION ---
    ttk.Label(frame, text="LAYER VISIBILITY", font=('Arial', 10, 'bold')).pack(pady=(15, 5))

    # 1. Model Visibility Toggle
    view_var = tk.BooleanVar(value=True)

    def toggle_model():
        ui_settings['show_model_right'] = view_var.get()
        update_3d_view()

    model_f = ttk.Frame(frame)
    model_f.pack(fill='x')
    ttk.Checkbutton(model_f, text="Show Print Model", variable=view_var,
                    command=toggle_model).pack(side='left', padx=10)

    # 2. Seed Visibility Toggle (The code you asked about)
    seeds_var = tk.BooleanVar(value=True)

    def toggle_seeds():
        ui_settings['show_seeds_right'] = seeds_var.get()
        update_3d_view()

    seed_f = ttk.Frame(frame)
    seed_f.pack(fill='x')
    ttk.Checkbutton(seed_f, text="Show Red Seeds", variable=seeds_var,
                    command=toggle_seeds).pack(side='left', padx=10)

    # ... (Before your 'on_finish' and 'CONFIRM' button) ...
    # --- FINAL RETURN DATA ---
    final_res = {"pm": None, "sub": None}

    def on_finish():
        pm_f, sub_f = get_rotated_trimeshes()
        final_res["pm"], final_res["sub"] = pm_f, sub_f
        root.quit() # Stop the loop first
        root.destroy()
        p.close()

    ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=30)
    ttk.Button(frame, text="CONFIRM & START DUAL CONTOURING", command=on_finish).pack(fill='x', pady=10)

    def launch():
        # Setup the three active quadrants (leaving 1,1 empty for now)
        active_quadrants = [(0, 0), (0, 1), (1, 0)]
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