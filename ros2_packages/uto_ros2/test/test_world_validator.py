"""Positive and negative tests for the bidirectional YAML/SDF contract."""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml

from uto_ros2.ifds_contract import validate_obstacle_world


PACKAGE = Path(__file__).resolve().parents[1]
STATIC_YAML = PACKAGE / "config/corridor_static_4_obstacles.yaml"
STATIC_SDF = PACKAGE / "worlds/my_rgl_corridor_static_4.sdf"
DYNAMIC_YAML = PACKAGE / "config/corridor_dynamic_4_obstacles.yaml"
DYNAMIC_SDF = PACKAGE / "worlds/my_rgl_corridor_dynamic_4.sdf"


def _files(tmp_path, data, root):
    yaml_path = tmp_path / "obstacles.yaml"
    sdf_path = tmp_path / "world.sdf"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    ET.ElementTree(root).write(sdf_path, encoding="unicode")
    return yaml_path, sdf_path


def _load(dynamic=False):
    data = yaml.safe_load((DYNAMIC_YAML if dynamic else STATIC_YAML).read_text())
    root = ET.parse(DYNAMIC_SDF if dynamic else STATIC_SDF).getroot()
    return data, root


def _assert_failure(tmp_path, data, root, fragment):
    valid, reasons = validate_obstacle_world(*_files(tmp_path, data, root))
    assert not valid
    assert any(fragment in reason for reason in reasons), reasons


def test_reference_corridor_worlds_validate_bidirectionally():
    assert validate_obstacle_world(STATIC_YAML, STATIC_SDF) == (True, [])
    assert validate_obstacle_world(DYNAMIC_YAML, DYNAMIC_SDF) == (True, [])


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    [
        ("wall_boundary", "wall boundary"),
        ("wall_thickness", "wall boundary"),
        ("wall_axis", "wall boundary"),
        ("wall_inside_sign", "invalid wall inside_sign"),
        ("yaml_missing", "SDF obstacle missing from YAML"),
        ("sdf_extra", "SDF obstacle missing from YAML"),
        ("center", "center mismatch"),
        ("axes", "axes/radii mismatch"),
        ("dynamic_static", "static/dynamic mismatch"),
        ("plugin_missing", "missing path plugin"),
        ("start", "start mismatch"),
        ("end", "end mismatch"),
        ("velocity", "velocity mismatch"),
        ("yaml_duplicate", "duplicate YAML obstacle name"),
        ("sdf_duplicate", "duplicate SDF model name"),
        ("frame", "planning frame mismatch"),
    ],
)
def test_world_validator_reports_specific_contract_failures(tmp_path, mutation, fragment):
    dynamic = mutation in {"dynamic_static", "plugin_missing", "start", "end", "velocity"}
    data, root = _load(dynamic)
    if mutation == "wall_boundary":
        data["obstacles"][0]["boundary"] += 0.2
    elif mutation == "wall_thickness":
        root.find(".//model[@name='corridor_wall_left']//box/size").text = "43.5 0.3 2"
    elif mutation == "wall_axis":
        data["obstacles"][0]["axis"] = "x"
    elif mutation == "wall_inside_sign":
        data["obstacles"][0]["inside_sign"] = 0
    elif mutation == "yaml_missing":
        data["obstacles"].pop()
    elif mutation == "sdf_extra":
        extra = deepcopy(root.find(".//model[@name='static_obstacle_1']"))
        extra.set("name", "static_obstacle_extra")
        root.find("world").append(extra)
    elif mutation == "center":
        data["obstacles"][4]["center"][0] += 1.0
    elif mutation == "axes":
        data["obstacles"][4]["axes"][0] += 0.1
    elif mutation == "dynamic_static":
        root.find(".//model[@name='dynamic_obstacle_3']/static").text = "true"
    elif mutation == "plugin_missing":
        model = root.find(".//model[@name='dynamic_obstacle_3']")
        model.remove(model.find("plugin"))
    elif mutation in {"start", "end", "velocity"}:
        plugin = root.find(".//model[@name='dynamic_obstacle_3']/plugin")
        plugin.find(mutation).text = "99 99 99" if mutation != "velocity" else "9.0"
    elif mutation == "yaml_duplicate":
        data["obstacles"].append(deepcopy(data["obstacles"][-1]))
    elif mutation == "sdf_duplicate":
        duplicate = deepcopy(root.find(".//model[@name='static_obstacle_1']"))
        root.find("world").append(duplicate)
    elif mutation == "frame":
        data["header"]["frame_id"] = "odom"
    _assert_failure(tmp_path, data, root, fragment)


def test_invalid_xml_and_missing_world_have_distinct_input_errors(tmp_path):
    bad = tmp_path / "bad.sdf"
    bad.write_text("<sdf><broken>")
    valid, reasons = validate_obstacle_world(STATIC_YAML, bad)
    assert not valid and "world validation input error" in reasons[0]
    valid, reasons = validate_obstacle_world(STATIC_YAML, tmp_path / "missing.sdf")
    assert not valid and "world validation input error" in reasons[0]
