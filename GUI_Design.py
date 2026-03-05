import pyvista as pv
import trimesh
import numpy as np
import trimesh.transformations as tf
from scratch import *
import pyvista as pv
import numpy as np


def get_user_orientation_gui(print_model, substrate):
    print("Launching Precision-UI: X-Axis Calibration Mode...")

    p = pv.Plotter(shape=(1, 2), window_size=[1600, 800], border=False)
    pm_orig_pv = pv.wrap(print_model).copy()
    sub_orig_pv = pv.wrap(substrate).copy()
    angles = {'x': 0.0, 'y': 0.0, 'z': 0.0}

    # =========================================================
    # 1. LOGIC & HELPERS
    # =========================================================

    def draw_sleek_triad(plotter, length=12):
        l = float(length)
        coords = {
            'x': np.array([[l + 2.0, 0.0, 0.0]], dtype=np.float32),
            'y': np.array([[0.0, l + 2.0, 0.0]], dtype=np.float32),
            'z': np.array([[0.0, 0.0, l + 2.0]], dtype=np.float32)
        }
        plotter.add_mesh(pv.Arrow(start=(0, 0, 0), direction=(l, 0, 0), tip_radius=0.08, shaft_radius=0.02),
                         color='red')
        plotter.add_point_labels(coords['x'], ["X"], font_size=14, text_color='red', shadow=False, shape=None)
        plotter.add_mesh(pv.Arrow(start=(0, 0, 0), direction=(0, l, 0), tip_radius=0.08, shaft_radius=0.02),
                         color='green')
        plotter.add_point_labels(coords['y'], ["Y"], font_size=14, text_color='green', shadow=False, shape=None)
        plotter.add_mesh(pv.Arrow(start=(0, 0, 0), direction=(0, 0, l), tip_radius=0.08, shaft_radius=0.02),
                         color='blue')
        plotter.add_point_labels(coords['z'], ["Z"], font_size=14, text_color='blue', shadow=False, shape=None)

    def redraw_scene():
        pm_new = pm_orig_pv.copy()
        sub_new = sub_orig_pv.copy()
        for mesh in [pm_new, sub_new]:
            mesh.rotate_x(angles['x'], inplace=True, point=[0, 0, 0])
            mesh.rotate_y(angles['y'], inplace=True, point=[0, 0, 0])
            mesh.rotate_z(angles['z'], inplace=True, point=[0, 0, 0])
        p.subplot(0, 0)
        p.add_mesh(pm_new, color='#3070B3', opacity=0.8, show_edges=True, name='pm')
        p.add_mesh(sub_new, color='#D0D0D0', opacity=0.4, show_edges=True, name='sub')

        # Readout positioned higher to clear the buttons
        readout = f"ORIENT STATE >> X: {int(angles['x'])}° | Y: {int(angles['y'])}° | Z: {int(angles['z'])}°"
        p.add_text(readout, position=(0.02, 0.22), font_size=9, color='#2C3E50', name='readout')
        p.render()

    def rotate_step(axis, increment):
        angles[axis] = (angles[axis] + increment) % 360
        redraw_scene()

    # =========================================================
    # 2. UI ASSEMBLY
    # =========================================================

    p.subplot(0, 0)
    p.set_background('white')
    p.add_text("WCS CONFIGURATION", position=(0.02, 0.95), font_size=10, color='#333333')
    p.show_grid(color='#AAAAAA')
    draw_sleek_triad(p, length=12)

    # --- X-AXIS MODULE (CALIBRATION v2) ---
    X_START = 0.05

    # 1. Label: Moved significantly higher to 0.18
    p.add_text("ROT X",
               position=(X_START, 0.18),
               font_size=10,
               color='red',
               viewport=True,
               name='label_x_axis')

    # 2. Button: Moved lower to 0.05
    # Increased size to 35 for a better "hit box"
    p.add_checkbox_button_widget(
        lambda s: rotate_step('x', 90),
        position=(X_START, 0.05),
        size=35,
        color_on='red',
        color_off='#f0f0f0'
    )

    # --- RIGHT SIDE ---
    p.subplot(0, 1)
    p.set_background('white')
    p.add_text("ENGINEERING PREVIEW", position=(0.02, 0.95), font_size=10, color='#333333')
    p.show_grid(color='#AAAAAA')
    draw_sleek_triad(p, length=12)

    redraw_scene()
    p.show(title="B-Rep Conformal Slicer | X-Axis Calibration")
    return angles['x'], angles['y'], angles['z']

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