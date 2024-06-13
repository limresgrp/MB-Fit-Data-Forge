import numpy as np


def get_bonds(pos: np.ndarray, bond_idcs: np.ndarray):
    p = pos[:, bond_idcs]
    p0 = p[:, :, 0]
    p1 = p[:, :, 1]

    b0 = p1 - p0
    return np.linalg.norm(b0, axis=2, keepdims=False)

def get_angles_from_vectors(b0: np.ndarray, b1: np.ndarray, return_cos: bool = False):
    b0n = np.linalg.norm(b0, axis=2, keepdims=False)
    b1n = np.linalg.norm(b1, axis=2, keepdims=False)
    angles = np.sum(b0 * b1, axis=-1) / b0n / b1n
    clamped_cos = np.clip(angles, a_min=-1., a_max=1.)
    if return_cos:
        return clamped_cos
    return np.arccos(clamped_cos)

def get_angles(pos, angle_idcs):
    dist_vectors = pos[:, angle_idcs]
    b0 = -1.0 * (dist_vectors[:, :, 1] - dist_vectors[:, :, 0])
    b1 = (dist_vectors[:, :, 2] - dist_vectors[:, :, 1])
    return get_angles_from_vectors(b0, b1)

def get_dihedrals(pos: np.ndarray, dihedral_idcs: np.ndarray):
    p = pos[:, dihedral_idcs]
    p0 = p[..., 0, :]
    p1 = p[..., 1, :]
    p2 = p[..., 2, :]
    p3 = p[..., 3, :]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1, axis=-1, keepdims=True)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.einsum('ijk,ikj->ij', b0, np.transpose(b1, (0, 2, 1)))[..., None] * b1
    w = b2 - np.einsum('ijk,ikj->ij', b2, np.transpose(b1, (0, 2, 1)))[..., None] * b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.einsum('ijk,ikj->ij', v, np.transpose(w, (0, 2, 1)))
    y = np.einsum('ijk,ikj->ij', np.cross(b1, v), np.transpose(w, (0, 2, 1)))
    return np.arctan2(y, x)

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size) 
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

def dist_from_line(points, line_point_A, line_point_B):
    """
        Distance of point to line passing through line_point_A and line_point_B
    """
    if len(points.shape) == 3 and len(line_point_A.shape) == 2:
        assert len(points) == len(line_point_A)
        assert len(points) == len(line_point_B)
        line_point_A = line_point_A[:, None, :]
        line_point_B = line_point_B[:, None, :]
    a = points - line_point_A # Translate system so that point A is the origin
    b = line_point_B - line_point_A # ---------------------------------------
    
    s = np.sum(a*b, axis=-1, keepdims=True) / np.sum(b*b, axis=-1, keepdims=True)
    return np.linalg.norm(a - s*b, axis=-1)

def rad_of_gyration(points, masses, line_point_A, line_point_B):
    """
        Radius of Gyration of system composed by 'points' w.r.t. the axis of rotation
        passing through 'line_point_A' and 'line_point_B'
    """
    dists = dist_from_line(points, line_point_A, line_point_B)
    return np.sqrt(np.sum(masses * dists * dists, axis=-1) / np.sum(masses))

def rad_of_gyration_compactness(points, masses, point_P):
    """
        Radius of Gyration of system composed by 'points' w.r.t. center 'point_P'
    """
    if len(points.shape) == 3 and len(point_P.shape) == 2:
        assert len(points) == len(point_P)
        point_P = point_P[:, None, :]
    dists = np.linalg.norm(points - point_P, axis=-1)
    return np.sqrt(np.sum(masses * dists * dists, axis=-1) / np.sum(masses))