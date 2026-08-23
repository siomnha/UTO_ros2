import numpy as np


def dynamics(x, u, g=9.81, tau=0.35, drag=0.12):
    p = x[6]
    t = x[7]
    y = x[8]
    cp, sp, ct, st, cy, sy = np.cos(p), np.sin(p), np.cos(t), np.sin(t), np.cos(y), np.sin(y)
    R = np.array(
        [
            [cy * ct, cy * st * sp - sy * cp, cy * st * cp + sy * sp],
            [sy * ct, sy * st * sp + cy * cp, sy * st * cp - cy * sp],
            [-st, ct * sp, ct * cp],
        ]
    )
    a = R @ np.array([0, 0, u[0]]) - np.array([0, 0, g]) - drag * x[3:6]
    return np.r_[x[3:6], a, (u[1] - p) / tau, (u[2] - t) / tau, u[3]]


def rk4(x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + k1 * dt / 2, u)
    k3 = dynamics(x + k2 * dt / 2, u)
    k4 = dynamics(x + k3 * dt, u)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
