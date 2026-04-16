import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """3x3 skew-symmetric matrix."""
    return np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],  0.0,  -v[0]],
        [-v[1], v[0],  0.0],
    ])


def quat_mul(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w, x, y, z]."""
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = p
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ])


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quaternion) [w, x, y, z]."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion [w, x, y, z]."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([0.25/s, (R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s])
    if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """SO(3) exponential map."""
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3) + skew(phi)
    K = skew(phi / theta)
    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithm map."""
    cos_t = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if abs(theta) < 1e-10:
        return 0.5 * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    return (theta / (2*np.sin(theta))) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])


def small_angle_quat(dtheta: np.ndarray) -> np.ndarray:
    """Small rotation vector -> quaternion [w, x, y, z]."""
    half = dtheta * 0.5
    norm = np.linalg.norm(half)
    if norm < 1e-10:
        return np.array([1.0, half[0], half[1], half[2]])
    return np.array([np.cos(norm), *(np.sin(norm)/norm * half)])


def rot_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation that maps unit vector a to unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = np.dot(a, b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    if s < 1e-10:
        return np.eye(3) if c > 0 else -np.eye(3)
    K = skew(v / s)
    return np.eye(3) + s*K + (1 - c)*(K @ K)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)
