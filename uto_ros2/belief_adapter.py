"""FAST-LIO pose belief conversion with SO(3) tangent uncertainty."""

from dataclasses import dataclass
import numpy as np
from .math_utils import quat_to_rot, so3_exp, so3_log, rot_to_euler


def pose_covariance_to_tangent(pose_covariance):
    """ROS pose covariance ordering is already [xyz, rotation-about-xyz]; retain cross blocks."""
    return np.asarray(pose_covariance, float).reshape(6, 6).copy()


def sanitize_covariance(cov, eigen_floor=1e-9, inflation=1.0, reject_zero=True):
    c = np.asarray(cov, float).reshape(6, 6)
    if not np.all(np.isfinite(c)):
        raise ValueError("non-finite covariance")
    c = (c + c.T) / 2
    if reject_zero and np.max(np.abs(c)) < eigen_floor:
        raise ValueError("all-zero covariance")
    values, vectors = np.linalg.eigh(c)
    if values.min() < -max(1e-7, 100 * eigen_floor):
        raise ValueError("covariance is not PSD")
    return ((vectors * np.maximum(values, eigen_floor)) @ vectors.T) * float(inflation)


def simplex_sigma_points(mean, cov):
    mean = np.asarray(mean, float)
    cov = np.asarray(cov, float)
    d = len(mean)
    n = d + 1
    projection = np.eye(n) - np.ones((n, n)) / n
    values, vectors = np.linalg.eigh(projection)
    simplex = np.sqrt(n) * vectors[:, values > 0][:, :d].T
    return mean[:, None] + np.linalg.cholesky(cov) @ simplex, np.full(n, 1 / n)


def sigma_states(position, R_mean, velocity, cov):
    deviations, weights = simplex_sigma_points(np.zeros(6), cov)
    states = []
    for i in range(7):
        states.append(
            np.r_[
                np.asarray(position) + deviations[:3, i],
                velocity,
                rot_to_euler(R_mean @ so3_exp(deviations[3:, i])),
            ]
        )
    return np.asarray(states).T, weights


def attitude_sigma_tangent_mean(rotations, iterations=10):
    mean = np.asarray(rotations[0], float)
    for _ in range(iterations):
        correction = sum(so3_log(mean.T @ R) for R in rotations) / len(rotations)
        mean = mean @ so3_exp(correction)
        if np.linalg.norm(correction) < 1e-12:
            break
    return mean


@dataclass
class Belief:
    stamp: float
    frame_id: str
    position: np.ndarray
    rotation: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    sigma_states: np.ndarray
    mean_state: np.ndarray
    generation: int


@dataclass
class StabilityConfig:
    samples: int = 5
    timeout: float = 0.3
    position_trace: float = 0.2
    attitude_trace: float = 0.05
    mean_delta: float = 0.3
    covariance_delta: float = 0.1


class BeliefStableDetector:
    """Consecutive stability test using SO(3) geodesic attitude change."""

    def __init__(self, cfg=StabilityConfig()):
        self.cfg = cfg
        self.count = 0
        self.last = None

    def invalidate(self):
        self.count = 0
        self.last = None

    def update(self, stamp, now, position, rotation, cov, frame_ok=True, velocity_ok=True):
        position = np.asarray(position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        cov = np.asarray(cov, dtype=float)
        ok = (
            0.0 <= now - stamp <= self.cfg.timeout
            and frame_ok
            and velocity_ok
            and np.all(np.isfinite(position))
            and np.all(np.isfinite(rotation))
            and np.all(np.isfinite(cov))
            and np.trace(cov[:3, :3]) <= self.cfg.position_trace
            and np.trace(cov[3:, 3:]) <= self.cfg.attitude_trace
            and np.max(np.abs(cov)) > 0
        )
        if ok and self.last is not None:
            position_delta = np.linalg.norm(position - self.last[0])
            attitude_delta = np.linalg.norm(so3_log(self.last[1].T @ rotation))
            covariance_delta = np.linalg.norm(cov - self.last[2])
            ok = (
                position_delta <= self.cfg.mean_delta
                and attitude_delta <= self.cfg.mean_delta
                and covariance_delta <= self.cfg.covariance_delta
            )
        self.count = self.count + 1 if ok else 0
        self.last = (position.copy(), rotation.copy(), cov.copy()) if ok else None
        return self.count >= self.cfg.samples


class BeliefAdapter:
    def __init__(self, eigen_floor=1e-9, inflation=1.0, stability=StabilityConfig()):
        self.floor = eigen_floor
        self.inflation = inflation
        self.detector = BeliefStableDetector(stability)
        self.generation = 0

    def convert(
        self,
        stamp,
        frame_id,
        position,
        quaternion_xyzw,
        pose_covariance,
        velocity,
        now,
        frame_ok=True,
        velocity_ok=True,
    ):
        velocity = np.asarray(velocity, float)
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)) or not velocity_ok:
            raise ValueError("velocity unavailable or stale")
        R = quat_to_rot(quaternion_xyzw)
        cov = sanitize_covariance(
            pose_covariance_to_tangent(pose_covariance), self.floor, self.inflation
        )
        states, _ = sigma_states(position, R, velocity, cov)
        mean = np.r_[position, velocity, rot_to_euler(R)]
        stable = self.detector.update(stamp, now, position, R, cov, frame_ok, velocity_ok)
        self.generation += 1
        return (
            Belief(
                stamp,
                frame_id,
                np.asarray(position),
                R,
                velocity,
                cov,
                states,
                mean,
                self.generation,
            ),
            stable,
        )


def reconstruct_belief_from_sigma(sigma_states):
    """Recover a state mean and 6-D position/SO(3)-tangent covariance."""
    from .math_utils import euler_to_rot

    sigma = np.asarray(sigma_states, dtype=float)
    if sigma.shape != (9, 7):
        raise ValueError("sigma_states must have shape [9,7]")
    position = sigma[:3].mean(axis=1)
    velocity = sigma[3:6].mean(axis=1)
    rotations = [euler_to_rot(sigma[6:9, index]) for index in range(7)]
    rotation = attitude_sigma_tangent_mean(rotations)
    errors = np.empty((6, 7))
    errors[:3] = sigma[:3] - position[:, None]
    for index, sample_rotation in enumerate(rotations):
        errors[3:, index] = so3_log(rotation.T @ sample_rotation)
    covariance = errors @ errors.T / 7.0
    mean = np.r_[position, velocity, rot_to_euler(rotation)]
    return mean, rotation, (covariance + covariance.T) / 2.0


def resample_sigma_states(mean_state, rotation, covariance):
    """Generate seven position/SO(3) sigma states about a reconstructed belief."""
    return sigma_states(
        mean_state[:3],
        rotation,
        mean_state[3:6],
        sanitize_covariance(covariance, reject_zero=False),
    )[0]


def reconstruct_joint_tangent_from_sigma(sigma_states):
    """Return SO(3) mean and full [p,v,attitude-tangent] covariance/deviations."""
    from .math_utils import euler_to_rot

    sigma = np.asarray(sigma_states, dtype=float)
    mean, rotation, _ = reconstruct_belief_from_sigma(sigma)
    deviations = np.empty((9, sigma.shape[1]))
    deviations[:6] = sigma[:6] - mean[:6, None]
    for index in range(sigma.shape[1]):
        deviations[6:9, index] = so3_log(rotation.T @ euler_to_rot(sigma[6:9, index]))
    deviations -= deviations.mean(axis=1, keepdims=True)
    covariance = deviations @ deviations.T / sigma.shape[1]
    return mean, rotation, (covariance + covariance.T) / 2.0, deviations


def joint_sigma_process_update(sigma_states, pose_process_covariance):
    """Add pose noise through a joint rank-six tangent update preserving vertex identity."""

    sigma = np.asarray(sigma_states, dtype=float)
    process = np.asarray(pose_process_covariance, dtype=float).reshape(6, 6)
    mean, rotation, covariance, deviations = reconstruct_joint_tangent_from_sigma(sigma)
    if np.max(np.abs(process)) == 0.0:
        return sigma.copy(), mean, rotation, covariance
    target = covariance.copy()
    pose_indices = np.array([0, 1, 2, 6, 7, 8])
    target[np.ix_(pose_indices, pose_indices)] += process
    target = (target + target.T) / 2.0
    values, vectors = np.linalg.eigh(target)
    order = np.argsort(values)[::-1][:6]
    values = np.maximum(values[order], 0.0)
    factor = vectors[:, order] * np.sqrt(values)
    # Right singular coordinates retain each propagated simplex vertex identity.
    _, _, right = np.linalg.svd(deviations, full_matrices=True)
    latent = np.sqrt(7.0) * right[:6]
    updated_deviations = factor @ latent
    updated_deviations -= updated_deviations.mean(axis=1, keepdims=True)
    updated = np.empty_like(sigma)
    updated[:3] = mean[:3, None] + updated_deviations[:3]
    updated[3:6] = mean[3:6, None] + updated_deviations[3:6]
    for index in range(7):
        updated[6:9, index] = rot_to_euler(rotation @ so3_exp(updated_deviations[6:9, index]))
    new_mean, new_rotation, new_covariance, _ = reconstruct_joint_tangent_from_sigma(updated)
    return updated, new_mean, new_rotation, new_covariance
