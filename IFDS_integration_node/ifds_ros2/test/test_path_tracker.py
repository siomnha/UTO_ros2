import numpy as np

from ifds_ros2.path_tracker import CarrotPathTracker


def test_carrot_uses_spatial_distance_and_progress_is_monotonic():
    tracker = CarrotPathTracker(lookahead_distance=2.5)
    path = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    accepted, _ = tracker.replace_path(path, np.array([0.0, 0.0, 0.0]))
    carrot, _, progress = tracker.carrot(np.array([0.0, 0.0, 0.0]))
    assert accepted
    assert np.allclose(carrot, [2.5, 0.0, 0.0])
    assert progress == 0.0

    tracker.carrot(np.array([5.0, 0.0, 0.0]))
    _, _, progress_after_backtracking = tracker.carrot(np.array([4.0, 0.0, 0.0]))
    assert progress_after_backtracking == 5.0


def test_successful_replan_replaces_path_even_with_large_carrot_jump():
    tracker = CarrotPathTracker(lookahead_distance=2.0)
    old_path = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    rerouted_path = np.array([[4.0, 0.0, 0.0], [4.0, 10.0, 5.0]])
    tracker.replace_path(old_path, np.array([0.0, 0.0, 0.0]))
    old_carrot, _, _ = tracker.carrot(np.array([4.0, 0.0, 0.0]))

    accepted, deviation = tracker.replace_path(rerouted_path, np.array([4.0, 0.0, 0.0]))
    carrot, _, progress = tracker.carrot(np.array([4.0, 0.0, 0.0]))

    assert accepted
    assert deviation > 1.0
    assert not np.allclose(carrot, old_carrot)
    assert np.allclose(carrot, [4.0, 2.0 / np.sqrt(1.25), 1.0 / np.sqrt(1.25)])
    assert progress == 0.0


def test_consider_path_alias_uses_immediate_replacement():
    tracker = CarrotPathTracker(lookahead_distance=2.0)
    old_path = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    divergent_path = np.array([[4.0, 0.0, 0.0], [4.0, 10.0, 0.0]])
    tracker.replace_path(old_path, np.array([0.0, 0.0, 0.0]))
    tracker.carrot(np.array([4.0, 0.0, 0.0]))

    accepted, deviation = tracker.consider_path(divergent_path, np.array([4.0, 0.0, 0.0]))
    carrot, _, _ = tracker.carrot(np.array([4.0, 0.0, 0.0]))
    assert accepted
    assert deviation > 1.0
    assert np.allclose(carrot, [4.0, 2.0, 0.0])
