"""ROS-independent validation storage and DTO/UTO statistical summaries."""

from dataclasses import asdict, dataclass, fields
import csv
import json
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np


@dataclass
class TrialRecord:
    trial_id: str
    planner_mode: str
    seed: int
    sample_index: int
    success: bool
    failure_reason: str
    goal_x: float
    goal_y: float
    goal_z: float
    initial_error_px: float
    initial_error_py: float
    initial_error_pz: float
    initial_error_roll: float
    initial_error_pitch: float
    initial_error_yaw: float
    final_x: float = np.nan
    final_y: float = np.nan
    final_z: float = np.nan
    error_x: float = np.nan
    error_y: float = np.nan
    error_z: float = np.nan
    terminal_error_norm: float = np.nan
    path_tracking_rmse: float = np.nan
    maximum_path_error: float = np.nan
    maximum_roll: float = np.nan
    maximum_pitch: float = np.nan
    attitude_rate_rms: float = np.nan
    trajectory_duration: float = np.nan
    cold_build_time: float = np.nan
    parameter_update_time: float = np.nan
    solve_time: float = np.nan
    extraction_time: float = np.nan
    worker_total_time: float = np.nan
    predicted_terminal_covariance_trace: float = np.nan


FIELD_NAMES = [field.name for field in fields(TrialRecord)]


def _atomic_replace(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_records(csv_path) -> list[TrialRecord]:
    path = Path(csv_path).expanduser()
    if not path.exists():
        return []
    records = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            values = {}
            for name in FIELD_NAMES:
                value = row[name]
                if name in ("trial_id", "planner_mode", "failure_reason"):
                    values[name] = value
                elif name == "success":
                    values[name] = value.lower() == "true"
                elif name in ("seed", "sample_index"):
                    values[name] = int(value)
                else:
                    values[name] = float(value)
            records.append(TrialRecord(**values))
    return records


def method_statistics(records: list[TrialRecord], mode: str) -> dict:
    all_mode = [record for record in records if record.planner_mode == mode]
    valid = [record for record in all_mode if record.success and np.isfinite(record.terminal_error_norm)]
    positions = np.asarray([[r.final_x, r.final_y, r.final_z] for r in valid], dtype=float).reshape(-1, 3)
    errors = np.asarray([r.terminal_error_norm for r in valid], dtype=float)
    if len(valid) >= 2:
        covariance = np.cov(positions, rowvar=False, ddof=1)
        covariance_json = covariance.tolist()
        covariance_trace = float(np.trace(covariance))
    else:
        covariance = np.full((3, 3), np.nan)
        covariance_json = None
        covariance_trace = None
        warnings.warn(f"{mode}: fewer than two successful trials; sample covariance unavailable")
    goals = np.asarray([[r.goal_x, r.goal_y, r.goal_z] for r in valid], dtype=float).reshape(-1, 3)
    mean = positions.mean(axis=0) if len(valid) else np.full(3, np.nan)
    bias = (positions - goals).mean(axis=0) if len(valid) else np.full(3, np.nan)
    solve = np.asarray([r.solve_time for r in valid], dtype=float)
    return {
        "trial_count": len(all_mode),
        "successful_trial_count": len(valid),
        "success_rate": len(valid) / len(all_mode) if all_mode else 0.0,
        "terminal_mean": mean.tolist(),
        "mean_terminal_bias": bias.tolist(),
        "terminal_rmse": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        "terminal_covariance": covariance_json,
        "terminal_covariance_trace": covariance_trace,
        "terminal_error_p95": float(np.percentile(errors, 95)) if len(errors) else None,
        "mean_solve_time": float(np.nanmean(solve)) if len(solve) else None,
        "median_solve_time": float(np.nanmedian(solve)) if len(solve) else None,
        "mean_path_tracking_rmse": _finite_mean([r.path_tracking_rmse for r in valid]),
        "mean_attitude_rate_rms": _finite_mean([r.attitude_rate_rms for r in valid]),
    }


def _finite_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None


def build_outputs(records: list[TrialRecord]):
    successful = {
        mode: [r for r in records if r.planner_mode == mode and r.success]
        for mode in ("deterministic", "uto")
    }
    by_key = {
        mode: {(r.seed, r.sample_index): r for r in rows}
        for mode, rows in successful.items()
    }
    paired_keys = sorted(set(by_key["deterministic"]) & set(by_key["uto"]))
    matrices = {}
    for label, mode in (("DTO", "deterministic"), ("UTO", "uto")):
        rows = successful[mode]
        positions = np.asarray([[r.final_x, r.final_y, r.final_z] for r in rows], dtype=float).reshape(-1, 3)
        errors = np.asarray([r.terminal_error_norm for r in rows], dtype=float)
        matrices[f"terminal_positions_{label}"] = positions
        matrices[f"terminal_errors_{label}"] = errors
        matrices[f"terminal_mean_{label}"] = positions.mean(axis=0) if len(rows) else np.full(3, np.nan)
        matrices[f"terminal_covariance_{label}"] = (
            np.cov(positions, rowvar=False, ddof=1) if len(rows) >= 2 else np.full((3, 3), np.nan)
        )
    matrices["paired_error_difference"] = np.asarray(
        [by_key["uto"][key].terminal_error_norm - by_key["deterministic"][key].terminal_error_norm for key in paired_keys]
    )
    summary = {mode: method_statistics(records, mode) for mode in ("deterministic", "uto")}
    summary["paired_trial_count"] = len(paired_keys)
    return summary, matrices


class ValidationStore:
    """Append-and-rewrite store whose three outputs remain recoverable after every trial."""

    def __init__(self, output_directory):
        self.directory = Path(output_directory).expanduser()
        self.csv_path = self.directory / "validation_runs.csv"

    def append(self, record: TrialRecord) -> None:
        records = read_records(self.csv_path)
        records = [row for row in records if row.trial_id != record.trial_id] + [record]
        self.write(records)

    def write(self, records: list[TrialRecord]) -> None:
        def csv_writer(binary):
            import io
            text = io.TextIOWrapper(binary, encoding="utf-8", newline="", write_through=True)
            writer = csv.DictWriter(text, fieldnames=FIELD_NAMES)
            writer.writeheader()
            writer.writerows(asdict(record) for record in records)
            text.detach()

        _atomic_replace(self.csv_path, csv_writer)
        summary, matrices = build_outputs(records)
        _atomic_replace(
            self.directory / "validation_summary.json",
            lambda stream: stream.write(json.dumps(summary, indent=2, allow_nan=True).encode()),
        )
        _atomic_replace(
            self.directory / "validation_matrices.npz",
            lambda stream: np.savez(stream, **matrices),
        )


def summarize_main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args(argv)
    path = Path(args.input).expanduser()
    ValidationStore(path.parent).write(read_records(path))
