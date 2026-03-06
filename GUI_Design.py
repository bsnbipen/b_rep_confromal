import tkinter as tk
from tkinter import ttk
import pyvista as pv
import numpy as np
import trimesh
import trimesh.transformations as tf
import threading


def get_user_orientation_gui(print_model, substrate):
    # --- CONFIGURATION: GRID SCALE SETTINGS ---
    GRID_STEP = 10.0  # Unit increment (e.g., 10mm)

    def ensure_pyvista(obj):
        if isinstance(obj, str): return pv.read(obj)
        return pv.wrap(obj)

    try:
        pm_orig = ensure_pyvista(print_model).copy()
        sub_orig = ensure_pyvista(substrate).copy()

        all_points = np.vstack([pm_orig.points, sub_orig.points])
        max_val = (np.ceil(np.abs(all_points).max() / GRID_STEP) * GRID_STEP) + (2 * GRID_STEP)
        fixed_bounds = [-max_val, max_val, -max_val, max_val, -max_val, max_val]
    except Exception as e:
        print(f"Loading Error: {e}")
        return print_model, substrate, (0, 0, 0)

    state = {'x': 0, 'y': 0, 'z': 0}
    ui_settings = {'show_model_right': True}

    p = pv.Plotter(shape=(1, 2), window_size=[1200, 600])
    p.set_background('white', all_renderers=True)

    def get_rotated_meshes():
        p_rot = pm_orig.copy().rotate_x(state['x'], point=(0, 0, 0)).rotate_y(state['y'], point=(0, 0, 0)).rotate_z(
            state['z'], point=(0, 0, 0))
        s_rot = sub_orig.copy().rotate_x(state['x'], point=(0, 0, 0)).rotate_y(state['y'], point=(0, 0, 0)).rotate_z(
            state['z'], point=(0, 0, 0))
        return p_rot, s_rot

    def update_3d_view():
        pm_rot, sub_rot = get_rotated_meshes()

        p.subplot(0, 0)
        p.add_mesh(pm_rot, color='#3070B3', name='pm_mesh', show_edges=True, reset_camera=False)
        p.add_mesh(sub_rot, color='#D0D0D0', opacity=0.3, name='sub_mesh', reset_camera=False)
        p.show_grid(color='#AAAAAA', bounds=fixed_bounds, xtitle=f'X Axis ({GRID_STEP}mm units)',
                    ytitle='Y Axis', ztitle='Z Axis', grid='both', location='outer')

        p.subplot(0, 1)
        target_opacity = 1.0 if ui_settings['show_model_right'] else 0.0
        p.add_mesh(pm_rot, color='#3070B3', name='pm_preview',
                   opacity=target_opacity, show_edges=(target_opacity > 0), reset_camera=False)
        p.show_grid(color='#AAAAAA', bounds=fixed_bounds, grid='both')

        p.add_text(f"ROTATION: X={state['x']} | Y={state['y']} | Z={state['z']}",
                   position='upper_left', font_size=10, name='stat', color='black')
        p.render()

    # --- Tkinter Control UI ---
    root = tk.Tk()
    root.title(f"5-Axis Console | {GRID_STEP}mm Scale")
    root.geometry("380x800")  # Increased height for new button
    root.attributes('-topmost', True)

    frame = ttk.Frame(root, padding="20")
    frame.pack(expand=True, fill="both")

    # --- NEW: Slicing Preview Button ---
    def cmd_generate_slice():
        # Placeholder for your slicing surface logic
        print("[ACTION] Generate Slice triggered. Orientation is currently:", state)
        # update_3d_view() could be called here once you add skin data

    ttk.Label(frame, text="GEOMETRIC EXTRACTION", font=('Arial', 10, 'bold')).pack(pady=(5, 5))
    slice_btn = ttk.Button(frame, text="GENERATE SLICE", command=cmd_generate_slice)
    slice_btn.pack(fill='x', pady=5)

    ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=15)

    def create_axis_group(axis):
        ttk.Label(frame, text=f"{axis.upper()} AXIS CONTROL", font=('Arial', 10, 'bold')).pack(pady=(10, 2))
        btn_frame = ttk.Frame(frame);
        btn_frame.pack(fill='x')

        def rot(v):
            state[axis] = (state[axis] + v) % 360
            update_3d_view()

        ttk.Button(btn_frame, text="+90°", command=lambda: rot(90)).pack(side='left', expand=True, fill='x', padx=2)
        ttk.Button(btn_frame, text="-90°", command=lambda: rot(-90)).pack(side='left', expand=True, fill='x', padx=2)

    view_var = tk.BooleanVar(value=True)

    def on_toggle():
        ui_settings['show_model_right'] = view_var.get()
        update_3d_view()

    ttk.Label(frame, text="MODEL VISIBILITY (RIGHT)", font=('Arial', 10, 'bold')).pack(pady=(15, 5))
    rb_f = ttk.Frame(frame);
    rb_f.pack(fill='x')
    ttk.Radiobutton(rb_f, text="Show", variable=view_var, value=True, command=on_toggle).pack(side='left', padx=10)
    ttk.Radiobutton(rb_f, text="Hide", variable=view_var, value=False, command=on_toggle).pack(side='left', padx=10)

    for a in ['x', 'y', 'z']: create_axis_group(a)

    final_res = {"pm": None, "sub": None}

    def on_finish():
        pm_f, sub_f = get_rotated_meshes()
        final_res["pm"] = trimesh.Trimesh(vertices=pm_f.points, faces=pm_f.faces.reshape(-1, 4)[:, 1:])
        final_res["sub"] = trimesh.Trimesh(vertices=sub_f.points, faces=sub_f.faces.reshape(-1, 4)[:, 1:])
        p.close();
        root.destroy()

    ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=30)
    ttk.Button(frame, text="CONFIRM & START SLICING", command=on_finish).pack(fill='x', pady=10)

    def launch():
        for i in range(2):
            p.subplot(0, i)
            p.show_grid(bounds=fixed_bounds, grid='both')
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