"""Paired-seed SITL experiment planning and safely scoped process execution."""

from dataclasses import dataclass
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import yaml


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    planner_mode: str
    seed: int
    sample_index: int
    initial_error: tuple


READINESS_STAGES = (
    "clock_advancing",
    "px4_connected",
    "px4_hold_ready",
    "belief_stable",
    "ifds_path_valid",
    "trajectory_committed",
    "trajectory_finished",
    "goal_reached",
)


def readiness_failure(snapshot: dict) -> str:
    """Return the first unmet event-driven readiness stage."""
    for stage in READINESS_STAGES:
        if not snapshot.get(stage, False):
            return f"READINESS_{stage.upper()}_TIMEOUT"
    return ""


def paired_trial_specs(sample_count, base_seed, methods, standard_deviation):
    """Generate adjacent DTO/UTO trials sharing the exact sampled six-vector."""
    if sample_count < 1 or set(methods) - {"deterministic", "uto"}:
        raise ValueError("invalid paired experiment configuration")
    standard_deviation = np.asarray(standard_deviation, dtype=float)
    if standard_deviation.shape != (6,) or np.any(standard_deviation < 0):
        raise ValueError("initial error standard deviation must contain six nonnegative values")
    trials = []
    for sample_index in range(1, sample_count + 1):
        seed = base_seed + sample_index - 1
        error = np.random.default_rng(seed).normal(np.zeros(6), standard_deviation)
        for mode in methods:
            trials.append(TrialSpec(f"seed_{seed:03d}_{mode}", mode, seed, sample_index, tuple(error)))
    return trials


class InitialStateInjector:
    """Explicit injection boundary; default implementation never claims success."""

    def inject(self, trial: TrialSpec) -> bool:
        return False


class ManualInjector(InitialStateInjector):
    def __init__(self, confirmation_callback):
        self.confirmation_callback = confirmation_callback

    def inject(self, trial):
        return bool(self.confirmation_callback(trial))


class ManagedProcess:
    """Own exactly one process group and never signal unrelated ROS processes."""

    def __init__(self, command, environment=None):
        self.command = list(command)
        self.environment = environment
        self.process = None

    def start(self):
        self.process = subprocess.Popen(self.command, env=self.environment, start_new_session=True)

    def stop(self, timeout=10.0):
        if self.process is None or self.process.poll() is not None:
            return
        group = os.getpgid(self.process.pid)
        os.killpg(group, signal.SIGINT)
        try:
            self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(group, signal.SIGTERM)
            self.process.wait(timeout=timeout)


class ExperimentRunner:
    """Execute scoped full-restart trials only after explicit injection confirmation."""

    def __init__(self, process_factory, injector, readiness, completion):
        self.process_factory = process_factory
        self.injector = injector
        self.readiness = readiness
        self.completion = completion

    def run_trial(self, trial, readiness_timeout, execution_timeout):
        process = self.process_factory(trial)
        process.start()
        try:
            if not self.injector.inject(trial):
                return False, "INITIAL_STATE_INJECTION_UNCONFIRMED"
            ready, reason = wait_for(self.readiness, readiness_timeout)
            if not ready:
                return False, reason
            complete, reason = wait_for(self.completion, execution_timeout)
            return complete, reason
        finally:
            process.stop()


def wait_for(predicate, timeout, clock=time.monotonic, poll_period=0.05):
    deadline = clock() + timeout
    while clock() < deadline:
        if predicate():
            return True, ""
        time.sleep(poll_period)
    return False, "READINESS_TIMEOUT"


def load_experiment(path):
    with Path(path).open() as stream:
        return yaml.safe_load(stream)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--base-seed", type=int)
    args = parser.parse_args(argv)
    config = load_experiment(args.config)
    if args.sample_count is not None:
        config["sample_count"] = args.sample_count
    if args.base_seed is not None:
        config["base_seed"] = args.base_seed
    if not config.get("simulation_only", True) or config.get("allow_vehicle_commands", False):
        raise SystemExit("runner refuses non-simulation or vehicle-command-enabled configuration")
    std = config["initial_error_std"]["position"] + config["initial_error_std"]["attitude"]
    trials = paired_trial_specs(config["sample_count"], config["base_seed"], config["methods"], std)
    output = Path(config.get("validation_output_directory", "~/uto_validation_results")).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "experiment_plan.json"
    temporary = plan_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps([trial.__dict__ for trial in trials], indent=2))
    os.replace(temporary, plan_path)
    if args.dry_run or config.get("dry_run", False):
        print(json.dumps([trial.__dict__ for trial in trials], indent=2))
        return
    # No verified set-entity-state interface ships in this repository. Fail closed.
    for trial in trials:
        print(json.dumps({
            "trial_id": trial.trial_id, "planner_mode": trial.planner_mode, "seed": trial.seed,
            "sample_index": trial.sample_index, "initial_error": trial.initial_error,
            "status": "aborted", "reason": "INITIAL_STATE_INJECTION_UNCONFIRMED",
        }))
