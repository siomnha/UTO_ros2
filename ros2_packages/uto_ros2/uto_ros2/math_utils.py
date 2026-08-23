import numpy as np


def skew(v):
    x, y, z = np.asarray(v, float)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def so3_exp(v):
    v = np.asarray(v, float)
    a = np.linalg.norm(v)
    K = skew(v)
    if a < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + np.sin(a) / a * K + (1 - np.cos(a)) / a**2 * K @ K


def so3_log(R):
    R = np.asarray(R, float)
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    q = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return 0.5 * q if a < 1e-8 else a * q / (2 * np.sin(a))


def quat_to_rot(q):
    x, y, z, w = np.asarray(q, float)
    n = np.linalg.norm([x, y, z, w])
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("invalid quaternion")
    x, y, z, w = np.array([x, y, z, w]) / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_euler(R):
    return np.array(
        [
            np.arctan2(R[2, 1], R[2, 2]),
            np.arcsin(np.clip(-R[2, 0], -1, 1)),
            np.arctan2(R[1, 0], R[0, 0]),
        ]
    )


def enu_to_ned(v):
    v = np.asarray(v, float)
    return np.array([v[1], v[0], -v[2]])


def yaw_enu_to_ned(yaw):
    return np.pi / 2 - yaw


def ned_to_enu(v):
    v = np.asarray(v, float)
    return np.array([v[1], v[0], -v[2]])


def euler_to_quat(e):
    r, p, y = np.asarray(e, float) / 2
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )


def euler_to_rot(euler):
    """ZYX roll-pitch-yaw rotation matrix."""
    roll, pitch, yaw = np.asarray(euler, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def align_enu_velocity(velocity, mode="identity", yaw_offset=0.0):
    """Align PX4 ENU velocity with the planning map by a configured yaw."""
    velocity = np.asarray(velocity, dtype=float)
    if mode == "identity":
        return velocity.copy()
    if mode != "yaw_offset":
        raise ValueError("velocity frame alignment is unconfirmed")
    c, s = np.cos(yaw_offset), np.sin(yaw_offset)
    return np.array(
        [
            c * velocity[0] - s * velocity[1],
            s * velocity[0] + c * velocity[1],
            velocity[2],
        ]
    )
