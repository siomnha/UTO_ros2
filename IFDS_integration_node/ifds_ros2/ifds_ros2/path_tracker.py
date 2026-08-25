"""Stateful geometric path tracking helpers for IFDS setpoint generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Projection:
    """Projection of a position onto a polyline."""

    point: np.ndarray
    segment_index: int
    segment_fraction: float
    arc_length: float
    distance: float


class CarrotPathTracker:
    """Track progress and select a fixed-distance carrot on a 3-D polyline."""

    def __init__(self, lookahead_distance: float):
        self.lookahead_distance = max(float(lookahead_distance), 0.01)
        self.path: Optional[np.ndarray] = None
        self.cumulative: Optional[np.ndarray] = None
        self.progress_arc = 0.0

    @property
    def active(self) -> bool:
        return self.path is not None and len(self.path) >= 2

    def clear(self) -> None:
        self.path = None
        self.cumulative = None
        self.progress_arc = 0.0

    def replace_path(self, candidate: np.ndarray, position: np.ndarray) -> Tuple[bool, float]:
        """Immediately replace the tracked path with the latest complete IFDS plan.

        The AIAA IFDS baseline performs receding full-path replanning from the
        current UAV position to the unchanged global goal.  Therefore every
        successful complete path must become the active path, even if its carrot
        is far from the previous left/right or vertical branch.
        """

        candidate = np.asarray(candidate, dtype=float)
        position = np.asarray(position, dtype=float)
        if len(candidate) < 2:
            return False, float("inf")

        candidate_cumulative = self._cumulative_lengths(candidate)
        candidate_projection = self._project(candidate, candidate_cumulative, position, 0.0)
        deviation = 0.0
        if self.active:
            old_projection = self._project(self.path, self.cumulative, position, self.progress_arc)
            old_progress = max(self.progress_arc, old_projection.arc_length)
            old_carrot, _ = self._point_at_arc(self.path, self.cumulative, old_progress + self.lookahead_distance)
            new_carrot, _ = self._point_at_arc(
                candidate, candidate_cumulative, candidate_projection.arc_length + self.lookahead_distance
            )
            deviation = float(np.linalg.norm(new_carrot - old_carrot))

        self.path = candidate.copy()
        self.cumulative = candidate_cumulative
        self.progress_arc = candidate_projection.arc_length
        return True, deviation

    def consider_path(self, candidate: np.ndarray, position: np.ndarray) -> Tuple[bool, float]:
        """Backward-compatible alias for paper-faithful immediate replacement."""

        return self.replace_path(candidate, position)

    def carrot(self, position: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return carrot point, local unit tangent, and monotonic path progress."""

        if not self.active:
            raise RuntimeError("no active path")
        projection = self._project(self.path, self.cumulative, position, self.progress_arc)
        self.progress_arc = max(self.progress_arc, projection.arc_length)
        carrot, segment_index = self._point_at_arc(
            self.path,
            self.cumulative,
            self.progress_arc + self.lookahead_distance,
        )
        tangent = self.path[segment_index + 1] - self.path[segment_index]
        norm = float(np.linalg.norm(tangent))
        if norm > 1e-9:
            tangent = tangent / norm
        else:
            tangent = np.array([1.0, 0.0, 0.0])
        return carrot, tangent, self.progress_arc

    @staticmethod
    def _cumulative_lengths(path: np.ndarray) -> np.ndarray:
        lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        return np.concatenate(([0.0], np.cumsum(lengths)))

    @staticmethod
    def _point_at_arc(path: np.ndarray, cumulative: np.ndarray, arc: float) -> Tuple[np.ndarray, int]:
        arc = float(np.clip(arc, 0.0, cumulative[-1]))
        index = min(int(np.searchsorted(cumulative, arc, side="right") - 1), len(path) - 2)
        segment_length = cumulative[index + 1] - cumulative[index]
        fraction = 0.0 if segment_length < 1e-9 else (arc - cumulative[index]) / segment_length
        return path[index] + fraction * (path[index + 1] - path[index]), index

    @staticmethod
    def _project(
        path: np.ndarray,
        cumulative: np.ndarray,
        position: np.ndarray,
        minimum_arc: float,
    ) -> Projection:
        best: Optional[Projection] = None
        for index, (start, end) in enumerate(zip(path[:-1], path[1:])):
            segment = end - start
            length_squared = float(segment @ segment)
            if length_squared < 1e-12:
                continue
            fraction = float(np.clip(((position - start) @ segment) / length_squared, 0.0, 1.0))
            arc = float(cumulative[index] + fraction * np.sqrt(length_squared))
            if arc + 1e-9 < minimum_arc:
                continue
            point = start + fraction * segment
            distance = float(np.linalg.norm(position - point))
            if best is None or distance < best.distance:
                best = Projection(point, index, fraction, arc, distance)

        if best is not None:
            return best

        point, index = CarrotPathTracker._point_at_arc(path, cumulative, minimum_arc)
        return Projection(point, index, 0.0, minimum_arc, float(np.linalg.norm(position - point)))
