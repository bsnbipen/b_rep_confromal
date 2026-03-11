import numpy as np
import open3d as o3d
import trimesh.proximity
import trimesh
from numpy.ma.core import reshape

from DC import *

import igl

class Cube:
    def __init__(self, center, voxel_size, min_bounds, max_bounds, scene=None):
        self.center = np.asarray(center, dtype=float)
        self.voxel_size = float(voxel_size)
        self.half_size = voxel_size / 2.0
        self.min_bounds = np.asarray(min_bounds, dtype=float)
        self.max_bounds = np.asarray(max_bounds, dtype=float)

        self.voxel_index = None
        self.corners = None

        # sampled field values / validity / normals at corners
        self.sdf_values = None
        self.corner_valid = None
        self.corner_normals = None

        self.has_zerocrossing = False
        self.sign_change_edges = None
        self._intersection_points = None
        self.intersection_normals = None

        self.scene = scene

        # kept only for compatibility if other parts still attach them
        self.vertices = None
        self.faces = None

        self._corner_template = np.array([
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ], dtype=float)

        self._scaling_matrix = np.array([
            [voxel_size, 0, 0],
            [0, voxel_size, 0],
            [0, 0, voxel_size],
        ], dtype=float)

        self.get_voxel_index()
        self.compute_corners()

    def get_voxel_index(self):
        if self.voxel_index is None:
            self.voxel_index = np.floor((self.center - self.min_bounds) / self.voxel_size).astype(int)
        return self.voxel_index

    def compute_corners(self):
        if self.corners is None:
            scaled_corners = np.dot(self._corner_template, self._scaling_matrix.T)
            scaled_center = np.mean(scaled_corners, axis=0)
            translation_vector = self.center - scaled_center

            corners_homogeneous = np.hstack([
                scaled_corners,
                np.ones((self._corner_template.shape[0], 1))
            ])

            translation_matrix = np.array([
                [1, 0, 0, translation_vector[0]],
                [0, 1, 0, translation_vector[1]],
                [0, 0, 1, translation_vector[2]],
                [0, 0, 0, 1]
            ], dtype=float)

            translated_corners_homogeneous = np.dot(corners_homogeneous, translation_matrix.T)
            self.corners = np.array(translated_corners_homogeneous[:, :3], dtype=float)

        return self.corners

    def sample_corners(self, field_evaluator):
        """
        Ask the field evaluator for scalar values / normals / valid flags
        at the 8 cube corners.
        """
        query_points = self.compute_corners()
        values, normals, valid = field_evaluator.evaluate(query_points)

        self.sdf_values = np.asarray(values, dtype=float)
        self.corner_valid = np.asarray(valid, dtype=bool)

        if normals is not None:
            normals = np.asarray(normals, dtype=float)
            L = np.linalg.norm(normals, axis=1, keepdims=True)
            out = np.zeros_like(normals)
            good = L.squeeze() > 0
            out[good] = normals[good] / L[good]
            self.corner_normals = out
        else:
            self.corner_normals = None

        return self.sdf_values, self.corner_normals, self.corner_valid

    def interpolate_zero_on_edge(self, v1, v2, sdf1, sdf2, iso_level=0.0, eps=1e-12):
        v1 = np.asarray(v1, float)
        v2 = np.asarray(v2, float)

        f1 = float(sdf1 - iso_level)
        f2 = float(sdf2 - iso_level)

        # exact hits
        if abs(f1) <= eps:
            return v1
        if abs(f2) <= eps:
            return v2

        # no crossing
        if (f1 > 0) == (f2 > 0):
            return None

        t = f1 / (f1 - f2)
        t = min(1.0, max(0.0, t))
        return v1 + t * (v2 - v1)

    def compute_intersection_points(self, field_evaluator, iso_level=0.0):
        """
        Compute edge intersections for the target iso-level.
        """
        edges = [
            (0, 1), (3, 2), (4, 5), (7, 6),  # X-axis
            (0, 3), (1, 2), (4, 7), (5, 6),  # Y-axis
            (0, 4), (1, 5), (2, 6), (3, 7)   # Z-axis
        ]

        if self.sdf_values is None:
            raise ValueError("Call sample_corners(field_evaluator) before compute_intersection_points().")

        _sign_change_edges = []
        _intersection_points = []

        for edge in edges:
            v0, v1 = edge

            # skip if either endpoint is invalid
            if self.corner_valid is not None:
                if not (self.corner_valid[v0] and self.corner_valid[v1]):
                    continue

            sdf0, sdf1 = self.sdf_values[v0], self.sdf_values[v1]

            point = self.interpolate_zero_on_edge(
                self.corners[v0],
                self.corners[v1],
                sdf0,
                sdf1,
                iso_level=iso_level
            )

            if point is not None:
                self.has_zerocrossing = True
                _sign_change_edges.append(edge)
                _intersection_points.append(point)

        if _intersection_points:
            _intersection_points = np.array(_intersection_points, dtype=float)

            # sample normals at intersection points from the same field
            _, inter_normals, inter_valid = field_evaluator.evaluate(_intersection_points)

            if inter_normals is not None:
                inter_normals = np.asarray(inter_normals, dtype=float)
                L = np.linalg.norm(inter_normals, axis=1, keepdims=True)
                out = np.zeros_like(inter_normals)
                good = L.squeeze() > 0
                out[good] = inter_normals[good] / L[good]
                _normal_points = out
            else:
                _normal_points = None
        else:
            _normal_points = None

        self.sign_change_edges = np.array(_sign_change_edges, dtype=int) if _sign_change_edges else None
        self._intersection_points = np.array(_intersection_points, dtype=float) if _intersection_points else None
        self.intersection_normals = np.array(_normal_points, dtype=float) if _normal_points is not None else None

        return self._intersection_points, self.intersection_normals

    def get_cube_data(self):
        return {
            "center": self.center,
            "voxel_index": self.get_voxel_index(),
            "corners": self.corners,
            "sdf_values": self.sdf_values,
            "corner_valid": self.corner_valid,
            "has_zerocrossing": self.has_zerocrossing,
            "intersection_points": self._intersection_points,
            "intersection_normals": self.intersection_normals,
            "sign_change_edges": self.sign_change_edges,
        }

class BaseFieldEvaluator:
    """
    Abstract interface for a scalar field evaluator used by DC.
    """

    def evaluate(self, points):
        """
        Parameters
        ----------
        points : (N,3) array

        Returns
        -------
        values : (N,) float
            Scalar field values to contour (DC extracts zero).
        normals : (N,3) float
            Approximate field normals / gradients at those points.
        valid : (N,) bool
            Whether each point is valid for contouring.
        """
        raise NotImplementedError


class ConformalLayerFieldEvaluator(BaseFieldEvaluator):
    """
    Field evaluator for conformal slicing layers.

    Field definition:
        phi(x) = dot(x - p(x), n(x))
    where:
        p(x) = closest point on layer_zero_surface
        n(x) = local normal on layer_zero_surface

    DC field:
        psi(x) = phi(x) - target_height

    A point is valid only if it lies in a trusted model-domain region.
    """

    def __init__(
        self,
        layer_zero_surface,
        print_model_vertices,
        print_model_faces,
        target_height,
        patch_margin=None,
        model_band=None,
        use_inside_test=True,
        signed_distance_type=igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
    ):
        self.layer_zero_surface = layer_zero_surface.copy()
        self.print_model_vertices = np.asarray(print_model_vertices, dtype=float)
        self.print_model_faces = np.asarray(print_model_faces, dtype=int)
        self.target_height = float(target_height)

        # How far from layer_zero_surface we still trust the field
        self.patch_margin = target_height * 2.0 if patch_margin is None else float(patch_margin)

        # How far from print-model surface we still allow points if not strictly inside
        self.model_band = target_height * 1.5 if model_band is None else float(model_band)

        self.use_inside_test = use_inside_test
        self.signed_distance_type = signed_distance_type

        try:
            self.layer_zero_surface.fix_normals()
        except Exception:
            pass

        self._lz_face_normals = self.layer_zero_surface.face_normals

    def _closest_patch_data(self, points):
        """
        Closest point and patch-face normal from layer_zero_surface.
        """
        pts = np.asarray(points, dtype=float)

        closest_pts, distances, face_ids = trimesh.proximity.closest_point(
            self.layer_zero_surface, pts
        )

        valid_patch = (face_ids >= 0) & (face_ids < len(self.layer_zero_surface.faces))

        patch_normals = np.zeros_like(pts)
        patch_normals[valid_patch] = self._lz_face_normals[face_ids[valid_patch]]

        # normalize normals just in case
        L = np.linalg.norm(patch_normals, axis=1, keepdims=True)
        good = L.squeeze() > 0
        patch_normals[good] = patch_normals[good] / L[good]

        return closest_pts, distances, face_ids, patch_normals, valid_patch

    def _model_signed_distance(self, points):
        """
        Signed distance and closest-surface normals from print model.
        Uses libigl.
        """
        pts = np.asarray(points, dtype=np.float64)

        sdf, _, _, normals = igl.signed_distance(
            pts,
            self.print_model_vertices,
            self.print_model_faces,
            self.signed_distance_type
        )

        normals = np.asarray(normals, dtype=float)
        L = np.linalg.norm(normals, axis=1, keepdims=True)
        good = L.squeeze() > 0
        out = np.zeros_like(normals)
        out[good] = normals[good] / L[good]

        return np.asarray(sdf, dtype=float), out

    def _infer_inside_negative(self):
        """
        Optional calibration helper if you later want to robustly detect
        whether libigl returns negative inside for this mesh.
        For first version, we assume standard convention: inside <= 0.
        """
        return True

    def evaluate(self, points):
        pts = np.asarray(points, dtype=float)

        # -------------------------------------------------
        # 1. Closest-point data on layer_zero_surface
        # -------------------------------------------------
        closest_pts, patch_dist, face_ids, patch_normals, valid_patch = self._closest_patch_data(pts)

        # Directional field relative to the patch
        gap_vec = pts - closest_pts
        phi = np.einsum("ij,ij->i", gap_vec, patch_normals)

        # Fix sign convention locally if needed:
        # We want positive phi to mean moving away from substrate toward the model.
        # Heuristic: near the patch, if average phi is negative, flip.
        near_patch = valid_patch & (patch_dist <= (self.target_height + self.patch_margin))
        if np.any(near_patch):
            if np.mean(phi[near_patch]) < 0:
                phi = -phi
                patch_normals = -patch_normals

        # Shift field so DC extracts zero for this layer
        values = phi - self.target_height

        # -------------------------------------------------
        # 2. Model-domain restriction
        # -------------------------------------------------
        model_sdf, model_normals = self._model_signed_distance(pts)

        # Assume standard convention: inside <= 0
        inside_negative = self._infer_inside_negative()

        if inside_negative:
            inside_model = model_sdf <= 0.0
        else:
            inside_model = model_sdf >= 0.0

        near_model = np.abs(model_sdf) <= self.model_band

        # Keep points that are either inside model or sufficiently near its surface
        valid_model = inside_model | near_model

        # -------------------------------------------------
        # 3. Patch trust restriction
        # -------------------------------------------------
        # Avoid trusting field too far from the layer_zero patch
        valid_patch_band = valid_patch & (patch_dist <= (self.target_height + self.patch_margin))

        # Final validity
        valid = valid_patch_band & valid_model

        # -------------------------------------------------
        # 4. Normals for DC / Hermite use
        # -------------------------------------------------
        # For the first version, use patch normals as field normals.
        # This is consistent with the directional field phi.
        normals = patch_normals.copy()

        return values, normals, valid

class NeighborCube:
    def __init__(self, base_cube, direction, field_evaluator):
        """
        direction is the neighbor voxel index (i,j,k), same logical use as before.
        """
        if not isinstance(base_cube, Cube):
            raise ValueError("Base Cube must be a Cube Object")

        self.base_cube = base_cube
        self.voxel_size = self.base_cube.voxel_size
        self.min_bounds = self.base_cube.min_bounds
        self.max_bounds = self.base_cube.max_bounds
        self.scene = self.base_cube.scene

        if isinstance(direction, (list, tuple, np.ndarray)):
            direction = np.array(direction, dtype=int)
            if direction.shape != (3,):
                raise ValueError("Direction vector must be a 3D vector")
        else:
            raise ValueError("Direction must be a 3D vector (list, tuple, or numpy array)")

        new_center = ((direction + 0.5) * self.voxel_size) + self.min_bounds

        self.neighbor_cube = Cube(
            center=new_center,
            voxel_size=self.voxel_size,
            min_bounds=self.min_bounds,
            max_bounds=self.max_bounds,
            scene=self.scene
        )

        # 1) sample field at corners
        values, _, valid = self.neighbor_cube.sample_corners(field_evaluator)
        query_points = self.neighbor_cube.compute_corners()

        # 2) compute iso-surface edge intersections
        self.intersection_mat = self.neighbor_cube.compute_intersection_points(
            field_evaluator=field_evaluator,
            iso_level=0.0
        )

        self.has_zerocrossing = self.neighbor_cube.has_zerocrossing

        # 3) solve QEF only if active
        if self.has_zerocrossing and self.intersection_mat[0] is not None and len(self.intersection_mat[0]) > 0:
            self.x_vert, self.normal = qef_solution_for_cube(
                new_center,
                values,
                self.intersection_mat,
                query_points
            )
        else:
            self.x_vert = None
            self.normal = None