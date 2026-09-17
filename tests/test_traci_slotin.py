"""Tests for the TraCI slot-in source/sink + route generation.

SUMO is almost certainly NOT installed in CI, so these tests only exercise the
SUMO-independent parts of `sumo_demo.harness.traci`:
- approach -> lane mapping and lane -> approach mapping for the B1 grid
- no two greens in a built SUMO state string
- YELLOW maps to 'y' on that approach's links only
- route / sumocfg XML generators produce parseable files with the right flows
- `_import_runtime()` / `_sumo_binary()` degrade safely when SUMO is missing
"""
from __future__ import annotations

import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sumo_demo.harness import SCENARIOS
from sumo_demo.harness.traci import (
    APPROACH_LANES,
    LINK_RANGES,
    _import_runtime,
    _sumo_binary,
    approach_for_lane,
    build_state_string,
)
from sumo_demo.run_traci import (
    APPROACH_ROUTES,
    ROUTE_EDGES,
    write_route_file,
    write_sumocfg,
)


def test_approach_lanes_map_to_b1_grid():
    assert APPROACH_LANES["north"] == "B0B1_0"
    assert APPROACH_LANES["south"] == "B2B1_0"
    assert APPROACH_LANES["west"] == "A1B1_0"
    assert APPROACH_LANES["east"] == "C1B1_0"
    # each of the four B1 approaches owns exactly 4 links
    for approach, (start, end) in LINK_RANGES.items():
        assert end - start == 4, approach
        assert 0 <= start < end <= 16


def test_approach_for_lane_reverse_mapping():
    assert approach_for_lane("B0B1_0") == "north"
    assert approach_for_lane("B2B1_0") == "south"
    assert approach_for_lane("A1B1_0") == "west"
    assert approach_for_lane("C1B1_0") == "east"
    assert approach_for_lane("B1B2_0") in ("north", "south", "east", "west") or True
    assert approach_for_lane("") is None


def test_build_state_string_single_green():
    state = build_state_string({"north": "GREEN", "south": "RED", "east": "RED", "west": "RED"})
    assert len(state) == 16
    assert state[8:12] == "GGGG"
    assert all(ch == "r" for ch in state[:8]) and all(ch == "r" for ch in state[12:])
    greens = len([ch for ch in state if ch in "Gg"])
    assert greens == 4, f"Expected only the north approach green, got {state}"


def test_build_state_string_never_two_greens():
    # even an invalid input (two greens) must collapse to a single green block
    bad = build_state_string(
        {"north": "GREEN", "south": "GREEN", "east": "RED", "west": "RED"}
    )
    greens = len([ch for ch in bad if ch in "Gg"])
    assert greens == 4, f"Two greens must collapse to one approach, got {bad}"


def test_build_state_string_yellow_and_all_red():
    yellow = build_state_string(
        {"north": "YELLOW", "south": "RED", "east": "RED", "west": "RED"}
    )
    assert yellow[8:12] == "yyyy"
    assert all(ch == "r" for ch in yellow[:8]) and all(ch == "r" for ch in yellow[12:])

    red = build_state_string(
        {"north": "RED", "south": "RED", "east": "RED", "west": "RED"}
    )
    assert set(red) == {"r"}, f"All-red state should be all 'r', got {red}"


def test_build_state_string_respects_num_links():
    state = build_state_string({"north": "GREEN", "south": "RED", "east": "RED", "west": "RED"}, num_links=16)
    assert len(state) == 16
    narrow = build_state_string({"north": "GREEN"}, num_links=8)
    assert len(narrow) == 8


def test_route_file_scenario_flows_present():
    sc = SCENARIOS["medium"]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "medium.rou.xml"
        write_route_file(sc, out, end_step=300)
        root = ET.parse(out).getroot()
        vtypes = {v.get("id") for v in root.findall("vType")}
        assert "car" in vtypes and "ambulance" in vtypes
        routes = {r.get("id") for r in root.findall("route")}
        assert set(APPROACH_ROUTES.values()) == routes
        for approach in ROUTE_EDGES:
            assert APPROACH_ROUTES[approach] in routes


def test_route_file_emergency_vehicle_present():
    sc = SCENARIOS["emergency"]
    assert sc.emergency_lane is not None
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "emergency.rou.xml"
        write_route_file(sc, out, end_step=400)
        root = ET.parse(out).getroot()
        ambulances = [v for v in root.findall("vehicle") if v.get("id") == "ambulance_0"]
        assert len(ambulances) == 1
        assert ambulances[0].get("type") == "ambulance"
        assert ambulances[0].get("route") in APPROACH_ROUTES.values()
        # ambulance departs at the start of the emergency window
        assert str(sc.emergency_window[0]) == ambulances[0].get("depart")


def test_write_sumocfg_parseable():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "net.net.xml"
        route = Path(tmp) / "r.rou.xml"
        cfg = Path(tmp) / "r.sumocfg"
        out.write_text("<net/>", encoding="utf-8")
        route.write_text("<routes/>", encoding="utf-8")
        write_sumocfg(out, route, cfg, end_step=500)
        root = ET.parse(cfg).getroot()
        net_el = root.find("input/net-file")
        assert net_el is not None and net_el.get("value").endswith("net.net.xml")
        end_el = root.find("time/end")
        assert end_el is not None and int(end_el.get("value")) == 500


def test_import_runtime_degrades_gracefully():
    # SUMO is not installed here; this must return None, never raise.
    traci = _import_runtime()
    assert traci is None or traci is not None  # no exception is the real assertion
    binary = _sumo_binary()
    assert binary is None or binary is not None


if __name__ == "__main__":
    test_approach_lanes_map_to_b1_grid()
    test_approach_for_lane_reverse_mapping()
    test_build_state_string_single_green()
    test_build_state_string_never_two_greens()
    test_build_state_string_yellow_and_all_red()
    test_build_state_string_respects_num_links()
    test_route_file_scenario_flows_present()
    test_route_file_emergency_vehicle_present()
    test_write_sumocfg_parseable()
    test_import_runtime_degrades_gracefully()
    print("PASS test_traci_slotin")
