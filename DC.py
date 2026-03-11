import math
import trimesh.proximity
import trimesh
import igl
import numpy as np
from numpy import dtype

#from QEF import *


def set_symmetric_matrix(a00,a01,a02,a11,a12,a22):
    sym_mat=np.array([[a00,a01,a02],[a01,a11,a12],[a02,a12,a22]])
    return sym_mat

def givens_rotation(mat_A, i,k):
    "Perform Given rotation for 5x4 matrix: A': G.T@A"

    mat_A_new = mat_A.astype(float).copy()

    a=mat_A[i,i]  #pivot
    b=mat_A[k,i]  #eleminating element

    if abs(b)<1e-10:
        c=1;s=0
    else:
        if abs(b)>abs(a):
            tau=-(a/b)
            s=1/math.sqrt(1+(tau**2))
            c=s*tau
        else:
            tau=-(b/a)
            c=1/math.sqrt(1+(tau**2))
            s=c*tau

    for index_col in range(mat_A.shape[1]):
        val_1=mat_A[i,index_col]
        val_2=mat_A[k,index_col]
        mat_A_new[i,index_col]=(c*val_1)-(s*val_2)
        mat_A_new[k,index_col]=(s*val_1)+(c*val_2)
    return mat_A_new

def rotational_mat(A,V, i: int, k: int):
    B = A.astype(float).copy()
    #print("Input matrix for Tau:\n",B)
    a_pp = B[i, i]
    a_qq = B[k, k]
    a_qp = B[k, i]


    if a_qp==0:
        #print("A pq=",a_qp)
        c=1
        s=0
    else:
        tau = (a_qq - a_pp) / (2 * a_qp)

        #print("tau =", tau)
        stt=np.sqrt(1+(tau**2))

        if tau>=0:
            tan_ang=1/(tau+stt)
        else:
            tan_ang=1/(tau-stt)

        c = 1 / np.sqrt(1 + (tan_ang**2))
        s = tan_ang * c

    cc=c**2
    ss=s**2
    sc=s*c
    mix=2*c*s*B[i][k]

    G = np.eye(A.shape[0])
    G[i, i] = c; G[i, k] = s
    G[k, i] = -s; G[k, k] = c

    V_new=np.array([ [V[0][0]*G[0][0] + V[0][1]*G[1][0] + V[0][2]*G[2][0],   # m11
         V[0][0]*G[0][1] + V[0][1]*G[1][1] + V[0][2]*G[2][1],   # m12
         V[0][0]*G[0][2] + V[0][1]*G[1][2] + V[0][2]*G[2][2]],  # m13

        [V[1][0]*G[0][0] + V[1][1]*G[1][0] + V[1][2]*G[2][0],   # m21
         V[1][0]*G[0][1] + V[1][1]*G[1][1] + V[1][2]*G[2][1],   # m22
         V[1][0]*G[0][2] + V[1][1]*G[1][2] + V[1][2]*G[2][2]],  # m23

        [V[2][0]*G[0][0] + V[2][1]*G[1][0] + V[2][2]*G[2][0],   # m31
         V[2][0]*G[0][1] + V[2][1]*G[1][1] + V[2][2]*G[2][1],   # m32
         V[2][0]*G[0][2] + V[2][1]*G[1][2] + V[2][2]*G[2][2]]])

    #print("Rotation Matrix is:\n",G)
    #B_return=G.T@ B @ G
    if i==0 and k==1:
        B_return=set_symmetric_matrix(cc * B[0, 0] - mix + ss * B[1, 1],cc*B[0,1]-ss*B[1,0]+sc*B[0,0]-sc*B[1,1],c * B[0, 2] - s * B[1, 2],
    ss * B[0, 0] + mix + cc * B[1, 1],s * B[0, 2] + c * B[1, 2],B[2, 2])
    elif i==0 and k==2:
        B_return = set_symmetric_matrix(cc*B[0,0]+ss*B[2,2]-mix, c*B[0,1]-s*B[1,2],cc*B[0,2]-ss*B[2,0]+sc*B[0,0]-sc*B[2,2],B[1,1],s*B[1,0]+c*B[1,2],ss*B[0,0]+mix+cc*B[2,2])
    elif i==1 and k==2:
        B_return=set_symmetric_matrix(B[0, 0],c * B[0, 1] - s * B[0, 2],s * B[0, 1] + c * B[0, 2],
    cc * B[1, 1] - mix + ss * B[2, 2],cc*B[1,2]-ss*B[2,1]+sc*B[1,1]-sc*B[2,2],ss * B[1, 1] + mix + cc * B[2, 2])

    return B_return,V_new, G

def off_norm(A):
    return math.sqrt(np.sum(np.tril(A, -1)**2 + np.triu(A, 1)**2))

def jacobi_sweep(A, tol, max_sweep):
    if not np.allclose(A, A.T):
        raise ValueError("Input matrix must be symmetric")

    B = A.astype(float).copy()
    V = np.eye(A.shape[0])

    delta_tol = tol * off_norm(B)
    sweep = 0

    n = A.shape[0]

    while sweep < max_sweep and off_norm(B) > delta_tol:
            B,V,G = rotational_mat(B,V,0, 1)
            B, V, G = rotational_mat(B,V, 0, 2)
            B, V, G = rotational_mat(B,V, 1, 2)
            #A = (A + A.T) / 2  # Enforce symmetry

            sweep += 1

    return np.diag(B), V, B

def pseudo_inv(A,V,tol):
    eigenvalues=np.diag(A)
    inv_eigen=[]

    max_eigen=np.max(np.abs(eigenvalues))+ 1e-16
    dynamic_tol=max_eigen*tol




    for eigen in eigenvalues:
        if abs(eigen)<dynamic_tol:
            inv_eigen.append(0)
        else:
            inv_eigen.append(1/eigen)

    A_inv=np.array([[inv_eigen[0]*V[0][0]*V[0][0] + inv_eigen[1]*V[0][1]*V[0][1] + inv_eigen[2]*V[0][2]*V[0][2],
        inv_eigen[0]*V[0][0]*V[1][0] + inv_eigen[1]*V[0][1]*V[1][1] + inv_eigen[2]*V[0][2]*V[1][2],
        inv_eigen[0]*V[0][0]*V[2][0] + inv_eigen[1]*V[0][1]*V[2][1] + inv_eigen[2]*V[0][2]*V[2][2]],

        [inv_eigen[0] * V[1][0] * V[0][0] + inv_eigen[1] * V[1][1] * V[0][1] + inv_eigen[2] * V[1][2] * V[0][2],
        inv_eigen[0] * V[1][0] * V[1][0] + inv_eigen[1] * V[1][1] * V[1][1] + inv_eigen[2] * V[1][2] * V[1][2],
        inv_eigen[0] * V[1][0] * V[2][0] + inv_eigen[1] * V[1][1] * V[2][1] + inv_eigen[2] * V[1][2] * V[2][2]],

        [inv_eigen[0] * V[2][0] * V[0][0] + inv_eigen[1] * V[2][1] * V[0][1] + inv_eigen[2] * V[2][2] * V[0][2],
        inv_eigen[0] * V[2][0] * V[1][0] + inv_eigen[1] * V[2][1] * V[1][1] + inv_eigen[2] * V[2][2] * V[1][2],
        inv_eigen[0] * V[2][0] * V[2][0] + inv_eigen[1] * V[2][1] * V[2][1] + inv_eigen[2] * V[2][2] * V[2][2]]])
    return A_inv

def QR_decompose(mat_A):
    m,n=mat_A.shape
    R = np.zeros((n,n))
    for i in range(mat_A.shape[0]): #loop for number of rows in R
        R=R[0:4,0:4]
        R=np.vstack((R,mat_A[i]))
        for col in range(n):
            R=givens_rotation(R,col,4)
    #print(R)
    mat_A_dot=R[0:3,0:3]
    mat_B_dot=R[0:3,3]
    r_dot=R[3,3]
    return mat_A_dot, mat_B_dot, r_dot


def mat_vec_mult(A,b):
    """
       Return B_output = A @ B   using explicit Python loops.

       Parameters
       ----------
       A : (m, n) array_like
       b : (n, 1)

       Returns Output: B_output : (m,1)

       """

    A_rows,A_cols=A.shape
    B_output=np.zeros(A_rows)
    for i in range(A_rows):
        for j in range(A_cols):
            B_output[i]+=A[i,j]*b[j]
    return B_output

def cal_mass_point(point_array):
    #takes an of all point intersection from a voxel
    col_avg=np.mean(point_array,axis=0)
    return col_avg

def minimizer(ata_inv, a_aug, b_aug,mass_point):
    """
        Return c = (A^T A)^{-1} A^T (b - A p)

        Parameters
        ----------
        ata_inv    : (3, 3) array_like   # precomputed inverse of A^T A
        a_aug      : (3, 3) array_like
        b_aug      : (3, 1) array_like
        mass_point : (3, 1) array_like   # p

        Returns
        -------
        c_minimizer : (3, 1) array
        """

    a_p=mat_vec_mult(a_aug,mass_point)
    resid=b_aug-a_p
    rhs=mat_vec_mult(a_aug.T,resid)
    c_minimizer=mat_vec_mult(ata_inv,rhs)
    return c_minimizer

def qef_solver(intersect_mat,normal_mat,b_mat,tol, max_sweep):

    b_mat = b_mat.reshape(-1, 1)

    mass_point=cal_mass_point(intersect_mat)
    #print("Mass Point:\n",mass_point)

    sol_mat=np.hstack((normal_mat,b_mat))
    A_aug,b_aug,r=QR_decompose(sol_mat)
    #print("Augmented A form QR Decompose:\n",A_aug)
    #print("Augmented b form QR Decompose:\n", b_aug)

    N = A_aug.T @ A_aug
    eigenvalues,eigenvectors,N_diag=jacobi_sweep(N,tol,max_sweep)

    Ata_inv=pseudo_inv(N_diag,eigenvectors,tol)

    c_minimizer=minimizer(Ata_inv,A_aug,b_aug,mass_point)
    #print("C Minimizer:\n",c_minimizer)
    x_vec=c_minimizer+mass_point
    #print("x_vec:\n",x_vec)
    return (x_vec,c_minimizer)

def qef_solution(cubes, field_evaluator):
    """
    Parameters
    ----------
    cubes : list[Cube]
        Active candidate cubes in ROI
    field_evaluator : BaseFieldEvaluator
        Supplies scalar field values / normals / validity

    Returns
    -------
    vert_array : list
        Dual contour solution points
    cube_map : dict
        key   -> voxel index (i,j,k)
        value -> dual solution point
    normal_map : dict
        key   -> voxel index (i,j,k)
        value -> averaged normal at that voxel solution
    """
    vert_array = []
    cube_map = {}
    normal_map = {}

    print("Number of cubes passed to qef_solution:", len(cubes))

    for i, cube in enumerate(cubes):
        cube_info = cube.get_cube_data()

        # 1) sample field at corners
        sdf_values, _, corner_valid = cube.sample_corners(field_evaluator)

        # 2) compute zero-crossing intersections
        intersection_matrix = cube.compute_intersection_points(
            field_evaluator=field_evaluator,
            iso_level=0.0
        )

        if intersection_matrix[0] is None or len(intersection_matrix[0]) == 0:
            continue

        intersect_mat = np.array(intersection_matrix[0], dtype=float)
        normal_mat = np.array(intersection_matrix[1], dtype=float)

        # sanity checks
        if intersect_mat.ndim != 2 or intersect_mat.shape[1] != 3:
            continue
        if normal_mat.ndim != 2 or normal_mat.shape[1] != 3:
            continue
        if len(intersect_mat) < 3:
            continue

        # rhs for QEF
        b_matrix = np.einsum("ij,ij->i", normal_mat, intersect_mat)

        max_sweep = 10000
        tol = 1e-6

        x_vector, _minimizer = qef_solver(intersect_mat, normal_mat, b_matrix, tol, max_sweep)

        # clamp to cube bounds
        corner = cube.compute_corners()
        min_bound = np.min(corner, axis=0)
        max_bound = np.max(corner, axis=0)
        x_vector_clamped = np.clip(x_vector, min_bound, max_bound)

        voxel_index = tuple(cube_info["voxel_index"])
        cube_map[voxel_index] = x_vector_clamped

        n = normal_mat.mean(axis=0)
        L = np.linalg.norm(n)
        normal_map[voxel_index] = (n / L) if L > 0 else n

        vert_array.append(x_vector_clamped)

    return vert_array, cube_map, normal_map

def is_point_within_cube(cube_corners, point):
    """
    Check if the point is within the 8 corners of a cube.

    Parameters:
    - cube_corners: A list of tuples or a 2D array of shape (8, 3) representing the 8 corners of the cube.
    - point: A tuple or array representing the point to check.

    Returns:
    - True if the point is within the 8 corners of the cube, False otherwise.
    """

    # Ensure the cube corners are a numpy array for easier manipulation
    cube_corners = np.array(cube_corners)


    # Check if the point lies within the cube's bounding box
    min_bound = np.min(cube_corners, axis=0)
    max_bound = np.max(cube_corners, axis=0)


    # Compare point's coordinates with the bounding box
    within_x = min_bound[0] <= point[0] <= max_bound[0]
    within_y = min_bound[1] <= point[1] <= max_bound[1]
    within_z = min_bound[2] <= point[2] <= max_bound[2]

    return (within_x, within_y, within_z)


def reorder_voxel_offset_ccw(voxel_dual_ctr_dct, axis, trusted_normal=None):
    """
    Reorder voxel offsets in CCW order around the specified axis.

    Parameters
    ----------
    voxel_dual_ctr_dct : dict
        keys   -> local voxel offsets, e.g. (0,0,0), (0,-1,0), ...
        values -> dual contour solution points
    axis : str
        'x', 'y', or 'z'
    trusted_normal : dict
        keys   -> same local voxel offsets
        values -> normals for winding consistency

    Returns
    -------
    sorted_voxel_dual_ctr : dict
        same dictionary, but keys returned in sorted CCW order
    """
    voxel_lst = list(voxel_dual_ctr_dct.keys())

    if len(voxel_lst) != 4:
        raise ValueError(f"Expected exactly 4 voxel offsets, got {len(voxel_lst)}")

    points_3d = np.array([voxel_dual_ctr_dct[k] for k in voxel_lst], dtype=float)
    local_centroid = np.mean(points_3d, axis=0)

    if trusted_normal is None:
        raise ValueError("trusted_normal must be provided")

    normals = [np.array(trusted_normal[v], dtype=float) for v in voxel_lst]
    avg_normal = np.mean(normals, axis=0)
    avg_norm = np.linalg.norm(avg_normal)
    if avg_norm > 0:
        avg_normal = avg_normal / avg_norm

    # use list of (angle, voxel_key) instead of dict to avoid silent overwrite
    angle_pairs = []

    for k in voxel_lst:
        sol_pt = np.array(voxel_dual_ctr_dct[k], dtype=float)
        diff = sol_pt - local_centroid

        if axis == 'z':
            angle = math.atan2(diff[1], diff[0])   # XY plane
        elif axis == 'y':
            angle = math.atan2(diff[2], diff[0])   # XZ plane
        elif axis == 'x':
            angle = math.atan2(diff[2], diff[1])   # YZ plane
        else:
            raise ValueError("Axis must be x, y, or z")

        angle_pairs.append((angle, k))

    angle_pairs.sort(key=lambda x: x[0])
    sorted_voxels = [k for _, k in angle_pairs]

    # verify winding against average normal
    p0 = np.array(voxel_dual_ctr_dct[sorted_voxels[0]], dtype=float)
    p1 = np.array(voxel_dual_ctr_dct[sorted_voxels[1]], dtype=float)
    p2 = np.array(voxel_dual_ctr_dct[sorted_voxels[2]], dtype=float)

    edge_1 = p1 - p0
    edge_2 = p2 - p0
    geo_normal = np.cross(edge_1, edge_2)

    if np.dot(geo_normal, avg_normal) < 0:
        sorted_voxels.reverse()

    sorted_voxel_dual_ctr = {}
    for voxel_srt in sorted_voxels:
        sorted_voxel_dual_ctr[voxel_srt] = voxel_dual_ctr_dct[voxel_srt]

    return sorted_voxel_dual_ctr


def compute_edge_neighbor(cube, edge, solution_numbers, normals, field_evaluator):
    """
    Parameters
    ----------
    cube : Cube
        Active voxel
    edge : tuple
        Intersecting local edge, e.g. (0,1)
    solution_numbers : dict
        key   -> voxel index (i,j,k)
        value -> dual contour solution point
    normals : dict
        key   -> voxel index (i,j,k)
        value -> average normal at that voxel's dual point
    field_evaluator : BaseFieldEvaluator
        Field provider for missing neighbor cubes

    Returns
    -------
    sorted_normal_sol, voxel_edge_processed, axis
        or (None, None, None) if the face is incomplete / invalid
    """
    edge = tuple(int(x) for x in edge)

    edge_to_axis = {
        (0, 1): 'x', (3, 2): 'x', (4, 5): 'x', (7, 6): 'x',
        (0, 3): 'y', (1, 2): 'y', (4, 7): 'y', (5, 6): 'y',
        (0, 4): 'z', (1, 5): 'z', (2, 6): 'z', (3, 7): 'z'
    }

    edge_to_cube_offsets = {
        (0, 1): [(0, 0, 0), (0, -1, 0), (0, 0, -1), (0, -1, -1)],
        (3, 2): [(0, 0, 0), (0, 1, 0), (0, 0, -1), (0, 1, -1)],
        (4, 5): [(0, 0, 0), (0, -1, 0), (0, 0, 1), (0, -1, 1)],
        (7, 6): [(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1)],
        (0, 3): [(0, 0, 0), (-1, 0, 0), (0, 0, -1), (-1, 0, -1)],
        (1, 2): [(0, 0, 0), (1, 0, 0), (0, 0, -1), (1, 0, -1)],
        (4, 7): [(0, 0, 0), (-1, 0, 0), (0, 0, 1), (-1, 0, 1)],
        (5, 6): [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)],
        (0, 4): [(0, 0, 0), (-1, 0, 0), (0, -1, 0), (-1, -1, 0)],
        (1, 5): [(0, 0, 0), (1, 0, 0), (0, -1, 0), (1, -1, 0)],
        (2, 6): [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],
        (3, 7): [(0, 0, 0), (-1, 0, 0), (0, 1, 0), (-1, 1, 0)],
    }

    edge_shared_local = {
        (0, 1): [(0, 1), (3, 2), (4, 5), (7, 6)],
        (3, 2): [(3, 2), (0, 1), (7, 6), (4, 5)],
        (4, 5): [(4, 5), (7, 6), (0, 1), (3, 2)],
        (7, 6): [(7, 6), (4, 5), (3, 2), (0, 1)],
        (0, 3): [(0, 3), (1, 2), (4, 7), (5, 6)],
        (1, 2): [(1, 2), (0, 3), (5, 6), (4, 7)],
        (4, 7): [(4, 7), (5, 6), (0, 3), (1, 2)],
        (5, 6): [(5, 6), (4, 7), (1, 2), (0, 3)],
        (0, 4): [(0, 4), (1, 5), (3, 7), (2, 6)],
        (1, 5): [(1, 5), (0, 4), (2, 6), (3, 7)],
        (2, 6): [(2, 6), (3, 7), (1, 5), (0, 4)],
        (3, 7): [(3, 7), (2, 6), (0, 4), (1, 5)]
    }

    if edge not in edge_to_cube_offsets:
        raise ValueError(f"Invalid edge: {edge}")

    axis = edge_to_axis[edge]
    voxel_dual_cntr_dct = {}
    normal_dct = {}
    voxel_edge_processed = {}

    voxel_coord = tuple(cube.get_voxel_index())
    vx, vy, vz = voxel_coord

    offsets = edge_to_cube_offsets[edge]
    common_edges = edge_shared_local[edge]

    incomplete_face = False

    for i, coord in enumerate(offsets):
        cube_voxel_coord = (int(vx + coord[0]), int(vy + coord[1]), int(vz + coord[2]))
        voxel_edge_processed[cube_voxel_coord] = common_edges[i]

        if cube_voxel_coord in solution_numbers:
            dual_pt = solution_numbers[cube_voxel_coord]
            dual_n = normals[cube_voxel_coord]
        else:
            neighbor = NeighborCube(cube, cube_voxel_coord, field_evaluator)

            # skip incomplete neighbor solutions
            if neighbor.x_vert is None or neighbor.normal is None:
                incomplete_face = True
                break

            dual_pt = neighbor.x_vert
            dual_n = neighbor.normal

        voxel_dual_cntr_dct[coord] = dual_pt
        normal_dct[coord] = dual_n

    if incomplete_face or len(voxel_dual_cntr_dct) != 4:
        return None, None, None

    voxel_indices_dct = reorder_voxel_offset_ccw(voxel_dual_cntr_dct, axis, normal_dct)

    sorted_voxels_index = list(voxel_indices_dct.keys())
    sorted_normal_sol = {}
    for voxel_index in sorted_voxels_index:
        sorted_normal_sol[voxel_index] = (
            voxel_indices_dct[voxel_index],
            normal_dct[voxel_index]
        )

    return sorted_normal_sol, voxel_edge_processed, axis