"""Reusable normalised 2-region LGR UTO graph ported from the MATLAB model."""

from dataclasses import dataclass
import importlib
import time
import numpy as np
from .lgr import control_check_grid, lgr_operators, quadrature_weights, interpolation_matrix


@dataclass(frozen=True)
class UTOConfig:
    regions: int = 2
    nodes: int = 5
    sigma: int = 7
    references: int = 10
    gravity: float = 9.81
    attitude_tau: float = 0.35
    drag: float = 0.12
    state_scale: tuple = (3, 1, 1.2, 4, 4, 4, 0.6, 0.6, 0.6)
    control_scale: tuple = (9.81, 0.48, 0.48, 1.2)
    control_min: tuple = (0, -0.48, -0.48, -1.2)
    control_max: tuple = (18, 0.48, 0.48, 1.2)
    velocity_max: float = 4.0
    angle_max: float = 0.6
    terminal_position_tolerance: float = 0.25
    control_check_points: int = 31
    max_iter: int = 900
    tolerance: float = 2e-5
    acceptable_tolerance: float = 1e-4


class UTONLP:
    def __init__(self, cfg=UTOConfig()):
        if (cfg.regions, cfg.nodes, cfg.sigma) != (2, 5, 7):
            # Tests may use smaller K, but flight defaults are the mandated dimensions.
            if min(cfg.regions, cfg.nodes, cfg.sigma) < 1:
                raise ValueError("invalid startup dimensions")
        self.cfg = cfg
        self.opti = None
        self.build_count = 0
        self.build_time = 0.0
        self.parameter_update_time = 0.0
        self.solve_time = 0.0
        self.extraction_time = 0.0
        self.tau, self.D = lgr_operators(cfg.nodes)
        self.quad = quadrature_weights(self.tau)
        self.endpoint = interpolation_matrix(self.tau, [1.0])[:, 0]
        self.control_checks = control_check_grid(self.tau, cfg.control_check_points)
        self.control_interpolation = interpolation_matrix(self.tau, self.control_checks)

    def build(self):
        if self.opti is not None:
            return self
        if importlib.util.find_spec("casadi") is None:
            raise RuntimeError("CasADi with IPOPT is required")
        ca = importlib.import_module("casadi")
        start = time.perf_counter()
        c = self.cfg
        o = ca.Opti()
        self.opti = o
        sx = ca.DM(c.state_scale)
        su = ca.DM(c.control_scale)
        self._sx = np.asarray(c.state_scale, float)
        self._su = np.asarray(c.control_scale, float)
        self.p_x0 = o.parameter(9, c.sigma)
        self.p_ref = o.parameter(3, c.references)
        self.p_h = o.parameter()
        self.p_vref = o.parameter(3)
        self.p_vlo = o.parameter(3)
        self.p_vhi = o.parameter(3)
        self.p_mode = o.parameter()
        self.p_weights = o.parameter(6)
        self.p_ptol = o.parameter()
        self.U = [o.variable(4, c.nodes) for _ in range(c.regions)]
        self.X = [[o.variable(9, c.nodes + 1) for _ in range(c.regions)] for _ in range(c.sigma)]
        duration = self.p_h / c.regions
        D = ca.DM(self.D)
        q = ca.DM(self.quad)
        endpoint = ca.DM(self.endpoint)
        lower = ca.DM(c.control_min) / su
        upper = ca.DM(c.control_max) / su
        effort = 0
        smooth = 0
        path = 0
        for region in range(c.regions):
            checked_control = self.U[region] @ ca.DM(self.control_interpolation)
            check_count = len(self.control_checks)
            o.subject_to(
                o.bounded(
                    ca.repmat(lower, 1, check_count),
                    checked_control,
                    ca.repmat(upper, 1, check_count),
                )
            )
            up = ca.diag(su) @ self.U[region]
            # Derivative matrix of the K-node control interpolant at its nodes.
            dc = np.column_stack([self._control_derivative_column(i) for i in range(c.nodes)])
            rate = (2 / duration) * up @ ca.DM(dc)
            for k in range(c.nodes):
                du = up[:, k] - ca.DM([c.gravity, 0, 0, 0])
                effort += (duration / 2) * q[k] * ca.dot(du, ca.DM([0.002, 0.15, 0.15, 0.04]) * du)
                smooth += (
                    (duration / 2)
                    * q[k]
                    * ca.dot(rate[:, k], ca.DM([0.01, 0.3, 0.3, 0.08]) * rate[:, k])
                )
            if region + 1 < c.regions:
                o.subject_to(self.U[region] @ endpoint == self.U[region + 1][:, 0])
            for sigma in range(c.sigma):
                xblock = self.X[sigma][region]
                o.subject_to(
                    xblock[:, 0]
                    == (
                        self.p_x0[:, sigma] / sx
                        if region == 0
                        else self.X[sigma][region - 1][:, -1]
                    )
                )
                velocity_limit = ca.repmat(
                    ca.DM(c.velocity_max / np.asarray(c.state_scale[3:6])), 1, c.nodes + 1
                )
                angle_limit = ca.repmat(
                    ca.DM(c.angle_max / np.asarray(c.state_scale[6:8])), 1, c.nodes + 1
                )
                o.subject_to(o.bounded(-velocity_limit, xblock[3:6, :], velocity_limit))
                o.subject_to(o.bounded(-angle_limit, xblock[6:8, :], angle_limit))
                for k in range(c.nodes):
                    physical = ca.diag(sx) @ xblock[:, k]
                    control = ca.diag(su) @ self.U[region][:, k]
                    o.subject_to(
                        xblock @ D[k, :].T
                        == (duration / 2) * (self._dynamics(ca, physical, control) / sx)
                    )
            for k in range(c.nodes):
                mean = sum(ca.diag(sx) @ self.X[s][region][:, k] for s in range(c.sigma)) / c.sigma
                ref_index = min(region * c.nodes + k, c.references - 1)
                error = mean[:3] - self.p_ref[:, ref_index]
                path += (duration / 2) * q[k] * ca.dot(error, error)
        terminal_states = [ca.diag(sx) @ self.X[s][-1][:, -1] for s in range(c.sigma)]
        self.terminal_mean = sum(terminal_states) / c.sigma
        self.terminal_cov = (
            sum((x - self.terminal_mean) @ (x - self.terminal_mean).T for x in terminal_states)
            / c.sigma
        )
        goal = self.p_ref[:, -1]
        position_error = self.terminal_mean[:3] - goal
        o.subject_to(o.bounded(-self.p_ptol, position_error, self.p_ptol))
        o.subject_to(o.bounded(self.p_vlo, self.terminal_mean[3:6], self.p_vhi))
        velocity_error = self.terminal_mean[3:6] - self.p_vref
        terminal_velocity = (1 - self.p_mode) * ca.dot(
            velocity_error, velocity_error
        ) + self.p_mode * ca.dot(self.terminal_mean[3:6], self.terminal_mean[3:6])
        cov_cost = ca.trace(self.terminal_cov[:3, :3])
        terminal_position = ca.dot(position_error, position_error)
        objective = (
            self.p_weights[0] * path
            + self.p_weights[1] * terminal_position
            + self.p_weights[2] * cov_cost
            + self.p_weights[3] * terminal_velocity
            + self.p_weights[4] * effort
            + self.p_weights[5] * smooth
        )
        o.minimize(objective)
        self.objective = objective
        o.solver(
            "ipopt",
            {"expand": True, "print_time": False},
            {
                "print_level": 0,
                "max_iter": c.max_iter,
                "tol": c.tolerance,
                "acceptable_tol": c.acceptable_tolerance,
                "nlp_scaling_method": "none",
            },
        )
        self.build_count = 1
        self.build_time = time.perf_counter() - start
        return self

    def _control_derivative_column(self, index):
        from .lgr import barycentric_weights, derivative_at_node

        return derivative_at_node(self.tau, barycentric_weights(self.tau), index)

    def _dynamics(self, ca, x, u):
        c = self.cfg
        phi, theta, psi = x[6], x[7], x[8]
        R = ca.vertcat(
            ca.horzcat(
                ca.cos(psi) * ca.cos(theta),
                ca.cos(psi) * ca.sin(theta) * ca.sin(phi) - ca.sin(psi) * ca.cos(phi),
                ca.cos(psi) * ca.sin(theta) * ca.cos(phi) + ca.sin(psi) * ca.sin(phi),
            ),
            ca.horzcat(
                ca.sin(psi) * ca.cos(theta),
                ca.sin(psi) * ca.sin(theta) * ca.sin(phi) + ca.cos(psi) * ca.cos(phi),
                ca.sin(psi) * ca.sin(theta) * ca.cos(phi) - ca.cos(psi) * ca.sin(phi),
            ),
            ca.horzcat(-ca.sin(theta), ca.cos(theta) * ca.sin(phi), ca.cos(theta) * ca.cos(phi)),
        )
        acceleration = R @ ca.vertcat(0, 0, u[0]) - ca.vertcat(0, 0, c.gravity) - c.drag * x[3:6]
        return ca.vertcat(
            x[3:6],
            acceleration,
            (u[1] - phi) / c.attitude_tau,
            (u[2] - theta) / c.attitude_tau,
            u[3],
        )

    def set_parameters(
        self,
        x0,
        references,
        horizon,
        vref,
        vlo,
        vhi,
        mode,
        weights,
        terminal_position_tolerance=None,
    ):
        start = time.perf_counter()
        pairs = [
            (self.p_x0, np.asarray(x0)),
            (self.p_ref, np.asarray(references).T),
            (self.p_h, horizon),
            (self.p_vref, vref),
            (self.p_vlo, vlo),
            (self.p_vhi, vhi),
            (self.p_mode, mode),
            (self.p_weights, weights),
            (
                self.p_ptol,
                (
                    self.cfg.terminal_position_tolerance
                    if terminal_position_tolerance is None
                    else terminal_position_tolerance
                ),
            ),
        ]
        if pairs[0][1].shape != (9, self.cfg.sigma) or pairs[1][1].shape != (
            3,
            self.cfg.references,
        ):
            raise ValueError("parameter dimensions would change fixed graph")
        for parameter, value in pairs:
            self.opti.set_value(parameter, value)
        self._initial_guess(np.asarray(x0), np.asarray(references)[-1], float(horizon))
        self.parameter_update_time = time.perf_counter() - start

    def _initial_guess(self, x0, goal, horizon):
        for r in range(self.cfg.regions):
            self.opti.set_initial(
                self.U[r], np.repeat(np.array([[1], [0], [0], [0.0]]), self.cfg.nodes, axis=1)
            )
            for s in range(self.cfg.sigma):
                block = np.empty((9, self.cfg.nodes + 1))
                for k, tau in enumerate(np.r_[self.tau, 1.0]):
                    q = (r + (tau + 1) / 2) / self.cfg.regions
                    block[:, k] = x0[:, s]
                    block[:3, k] = (1 - q) * x0[:3, s] + q * goal
                    block[3:6, k] = (goal - x0[:3, s]) / max(horizon, 0.1)
                    block[6:8, k] *= 1 - q
                self.opti.set_initial(self.X[s][r], block / self._sx[:, None])

    def solve(self):

        started = time.perf_counter()
        solution = self.opti.solve()
        self.solve_time = time.perf_counter() - started
        extract = time.perf_counter()
        c = self.cfg
        sigma_samples = []
        controls = []
        times = []
        normalized_blocks = [[None for _ in range(c.regions)] for _ in range(c.sigma)]
        physical_controls = []
        for region in range(c.regions):
            normalized_control = np.asarray(solution.value(self.U[region]), dtype=float)
            physical_control = self._su[:, None] * normalized_control
            physical_controls.append(physical_control)
            for sigma in range(c.sigma):
                normalized_blocks[sigma][region] = np.asarray(
                    solution.value(self.X[sigma][region]), dtype=float
                )
            local_times = (
                (region + (self.tau + 1.0) / 2.0) * float(self.opti.value(self.p_h)) / c.regions
            )
            for node, sample_time in enumerate(local_times):
                times.append(sample_time)
                controls.append(physical_control[:, node])
                sigma_samples.append(
                    np.stack(
                        [
                            self._sx * normalized_blocks[sigma][region][:, node]
                            for sigma in range(c.sigma)
                        ]
                    )
                )
        times.append(float(self.opti.value(self.p_h)))
        controls.append(controls[-1])
        sigma_samples.append(
            np.stack([self._sx * normalized_blocks[sigma][-1][:, -1] for sigma in range(c.sigma)])
        )
        sigma_samples = np.asarray(sigma_samples)
        from .belief_adapter import reconstruct_belief_from_sigma
        from .math_utils import euler_to_rot, so3_log

        mean_rows = []
        covariance_rows = []
        for samples in sigma_samples:
            sample_mean, mean_rotation, _ = reconstruct_belief_from_sigma(samples.T)
            errors = samples - sample_mean[None, :]
            for index, sample in enumerate(samples):
                errors[index, 6:9] = so3_log(mean_rotation.T @ euler_to_rot(sample[6:9]))
            mean_rows.append(sample_mean)
            covariance_rows.append(errors.T @ errors / c.sigma)
        mean = np.asarray(mean_rows)
        covariance = np.asarray(covariance_rows)
        maximum_residual = self._calculate_residual(
            normalized_blocks, physical_controls, float(self.opti.value(self.p_h))
        )
        self.extraction_time = time.perf_counter() - extract
        stats = solution.stats()
        return {
            "times": np.asarray(times),
            "states_physical": mean,
            "sigma_states_physical": sigma_samples,
            "controls_physical": np.asarray(controls),
            "mean_covariances": covariance,
            "terminal_covariance": covariance[-1],
            "objective": float(solution.value(self.objective)),
            "stats": stats,
            "iterations": int(stats.get("iter_count", -1)),
            "build_count": self.build_count,
            "max_lgr_dynamics_residual": maximum_residual,
            "normalized_state_blocks": normalized_blocks,
            "physical_control_blocks": physical_controls,
            "region_endpoint_sigma_physical": [
                np.stack(
                    [self._sx * normalized_blocks[sigma][region][:, -1] for sigma in range(c.sigma)]
                )
                for region in range(c.regions)
            ],
            "lgr_nodes": self.tau.copy(),
            "horizon": float(self.opti.value(self.p_h)),
            "regions": c.regions,
        }

    def _calculate_residual(self, normalized_blocks, physical_controls, horizon):
        from .dynamics import dynamics

        duration = horizon / self.cfg.regions
        maximum = 0.0
        for sigma in range(self.cfg.sigma):
            for region in range(self.cfg.regions):
                block = normalized_blocks[sigma][region]
                for node in range(self.cfg.nodes):
                    physical_state = self._sx * block[:, node]
                    physical_control = physical_controls[region][:, node]
                    residual = block @ self.D[node].T
                    residual -= (duration / 2.0) * (
                        dynamics(physical_state, physical_control) / self._sx
                    )
                    maximum = max(maximum, float(np.max(np.abs(residual))))
        return maximum

    def compute_residual(self, result):
        """Recompute the LGR residual from extracted state/control blocks."""
        required = ("normalized_state_blocks", "physical_control_blocks", "times")
        if not all(name in result for name in required):
            raise ValueError("result lacks LGR state/control blocks")
        horizon = float(np.asarray(result["times"])[-1])
        return self._calculate_residual(
            result["normalized_state_blocks"],
            result["physical_control_blocks"],
            horizon,
        )
