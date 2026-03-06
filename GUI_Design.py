import tkinter as tk
from tkinter import ttk
import pyvista as pv
import numpy as np
import trimesh
import trimesh.transformations as tf
import threading
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
    ui_settings = {'show_model_right': True}
    slice_assets = {'skin_pv': None}

    p = pv.Plotter(shape=(1, 2), window_size=[1400, 700])
    p.set_background('white', all_renderers=True)

    def get_rotated_trimeshes():
        """Applies current GUI rotations to fresh copies of the original meshes."""
        # Convert UI Degrees to Math Radians
        rx, ry, rz = np.radians(state['x']), np.radians(state['y']), np.radians(state['z'])

        pm = tm_pm_orig.copy()
        sub = tm_sub_orig.copy()

        # Apply rotations sequentially around the world origin (0,0,0)
        # Order: X -> Y -> Z (Standard Euler sequence for 5-axis simulation)
        for angle, axis in zip([rx, ry, rz], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
            if angle != 0:
                matrix = trimesh.transformations.rotation_matrix(angle, axis)
                pm.apply_transform(matrix)
                sub.apply_transform(matrix)
        return pm, sub

    # --- 2. ORIENTATION-AWARE SLICING LOGIC ---
    def cmd_generate_slice():
        # 1. Get meshes exactly as they are oriented on screen
        tm_pm_rotated, tm_sub_rotated = get_rotated_trimeshes()

        # 2. Run the extraction
        # The 'upward' check now finds the skin relative to the NEW orientation
        result = get_master_slicing_surface(tm_pm_rotated, tm_sub_rotated, padding=5)

        if result:
            master_skin, _, _ = result
            # This master_skin is now the 'Layer 0' for your Dual Contouring
            slice_assets['skin_pv'] = pv.wrap(master_skin)
            update_3d_view()

    def update_3d_view():
        tm_pm, tm_sub = get_rotated_trimeshes()
        pm_pv, sub_pv = pv.wrap(tm_pm), pv.wrap(tm_sub)

        # --- LEFT VIEW: CONFIG ---
        p.subplot(0, 0)
        p.add_mesh(pm_pv, color='#3070B3', name='pm_l', show_edges=True, edge_color='black', line_width=0.5,
                   reset_camera=False)
        p.add_mesh(sub_pv, color='#D0D0D0', opacity=0.3, name='sub_l', show_edges=True, edge_color='gray',
                   line_width=0.5, reset_camera=False)
        p.show_grid(color='#AAAAAA', bounds=fixed_bounds, xtitle=f'X ({GRID_STEP}mm)')

        # --- RIGHT VIEW: PREVIEW + DYNAMIC SLICE ---
        p.subplot(0, 1)
        model_opacity = 1.0 if ui_settings['show_model_right'] else 0.0
        p.add_mesh(pm_pv, color='#3070B3', name='pm_r', opacity=model_opacity, show_edges=True, edge_color='black',
                   reset_camera=False)

        if slice_assets['skin_pv'] is not None:
            # Render the slice based on the orientation it was generated in
            p.add_mesh(slice_assets['skin_pv'], color='#2ECC71', name='skin_r', show_edges=True, edge_color='darkgreen',
                       reset_camera=False)
        else:
            try:
                p.remove_actor('skin_r', render=False)
            except:
                pass

        p.show_grid(color='#AAAAAA', bounds=fixed_bounds)
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
        btn_f = ttk.Frame(frame);
        btn_f.pack(fill='x')

        def rot(v):
            state[axis] = (state[axis] + v) % 360
            slice_assets['skin_pv'] = None  # Auto-clear stale slices on move
            update_3d_view()

        ttk.Button(btn_f, text="+90°", command=lambda: rot(90)).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(btn_f, text="-90°", command=lambda: rot(-90)).pack(side='left', expand=True, fill='x', padx=2)

    for a in ['x', 'y', 'z']: create_axis_group(a)

    view_var = tk.BooleanVar(value=True)

    def toggle():
        ui_settings['show_model_right'] = view_var.get(); update_3d_view()

    ttk.Label(frame, text="PREVIEW OPTIONS", font=('Arial', 10, 'bold')).pack(pady=(15, 5))
    rb_f = ttk.Frame(frame);
    rb_f.pack(fill='x')
    ttk.Radiobutton(rb_f, text="Show Model", variable=view_var, value=True, command=toggle).pack(side='left', padx=10)
    ttk.Radiobutton(rb_f, text="Hide Model", variable=view_var, value=False, command=toggle).pack(side='left', padx=10)

    # --- FINAL RETURN DATA ---
    final_res = {"pm": None, "sub": None}

    def on_finish():
        pm_f, sub_f = get_rotated_trimeshes()
        final_res["pm"], final_res["sub"] = pm_f, sub_f
        p.close();
        root.destroy()

    ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=30)
    ttk.Button(frame, text="CONFIRM & START DUAL CONTOURING", command=on_finish).pack(fill='x', pady=10)

    def launch():
        for i in range(2):
            p.subplot(0, i);
            p.show_grid(bounds=fixed_bounds);
            p.reset_camera(bounds=fixed_bounds)
        update_3d_view();
        p.show(interactive_update=True)

    root.after(100, launch);
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