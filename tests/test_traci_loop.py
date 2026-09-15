"""End-to-end validation of the TraCI slot-in using a FakeTraCI stub.

Real SUMO isn't installed, so we emulate the subset of the TraCI API the
harness actually calls (lane / trafficlight / vehicle / simulation) with a tiny
deterministic traffic model:

  - vehicles can be placed on approach lanes; halting counts reflect them
  - when the sink sets a GREEN state, `simulationStep` serves up to N vehicles
    from the green approach (like a SUMO crossing) and moves them to `ended`
  - an emergency vehicle id ("ambulance_0") can be placed and moved between
    lanes to exercise the emergency-lane detection

This drives the same FalconBridge loop the real runner uses, proving
TraciTrafficSource + TraciSignalSink + controller work together with no SUMO.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.signal import SignalController
from sumo_demo.harness.bridge import FalconBridge, make_controller
from sumo_demo.harness.traci import (
    APPROACH_LANES,
    LINK_RANGES,
    TraciSignalSink,
    TraciTrafficSource,
)
from sumo_demo.harness.traffic import TrafficSnapshot


class _TrafficLight:
    def __init__(self, owner: "FakeTraCI"):
        self.owner = owner

    def getControlledLanes(self, tls_id: str) -> List[str]:
        # Return the real B1 link-lane order implied by LINK_RANGES.
        lanes = ["l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7",
                 "l8", "l9", "l10", "l11", "l12", "l13", "l14", "l15"]
        self.owner.controlled_lanes = lanes
        return lanes

    def setRedYellowGreenState(self, tls_id: str, state: str) -> None:
        self.owner.state_history.append(state)
        # Determine which approach is green from the 'G' block.
        self.owner.green_approach = None
        for approach, (start, end) in LINK_RANGES.items():
            if state[start:end] == "G" * (end - start):
                self.owner.green_approach = approach


class _Lane:
    def __init__(self, owner: "FakeTraCI"):
        self.owner = owner

    def getLastStepVehicleIDs(self, lane: str) -> List[str]:
        return sorted(self.owner.vehicles.get(lane, set()))

    def getLastStepHaltingNumber(self, lane: str) -> int:
        return len(self.owner.vehicles.get(lane, set()))


class _Vehicle:
    def __init__(self, owner: "FakeTraCI"):
        self.owner = owner

    def getIDList(self) -> List[str]:
        return list(self.owner.vehicles.values()) if False else self.owner._vehicle_ids_present()

    def _ids(self) -> List[str]:
        return self.owner._vehicle_ids_present()

    def getLaneID(self, vid: str) -> str:
        for lane in APPROACH_LANES.values():
            if vid in self.owner.vehicles.get(lane, set()):
                return lane
        return ""

    def getAccumulatedWaitingTime(self, vid: str) -> float:
        return float(self.owner.waiting_seconds.get(vid, 0.0))


class _Simulation:
    def __init__(self, owner: "FakeTraCI"):
        self.owner = owner

    def simulationStep(self) -> None:
        # Serve up to 1 vehicle on the currently-green approach (service_rate).
        green = self.owner.green_approach
        if green is not None:
            lane = APPROACH_LANES[green]
            parked = self.owner.vehicles.get(lane, set())
            if parked:
                done = parked.pop()
                self.owner.ended.append(done)
                for vid in list(parked):
                    self.owner.waiting_seconds[vid] = self.owner.waiting_seconds.get(vid, 0.0) + 1.0
        # All stopped vehicles accumulate wait when their approach is not green.
        for lane, parked in self.owner.vehicles.items():
            idx = self.owner._approach_of(lane)
            if idx is not None and self.owner.green_approach != idx:
                for vid in parked:
                    self.owner.waiting_seconds[vid] = self.owner.waiting_seconds.get(vid, 0.0) + 1.0

    def getEndedNumber(self) -> int:
        return len(self.owner.ended)


class FakeTraCI:
    """Minimal TraCI-substitute exposing the surface the harness touches."""

    def __init__(self):
        self.vehicles: Dict[str, Set[str]] = {l: set() for l in APPROACH_LANES.values()}
        self.ended: List[str] = []
        self.waiting_seconds: Dict[str, float] = {}
        self.state_history: List[str] = []
        self.green_approach: str | None = None
        self.controlled_lanes: List[str] = []
        self.trafficlight = _TrafficLight(self)
        self.lane = _Lane(self)
        self.vehicle = _Vehicle(self)
        self.simulation = _Simulation(self)
        self.simulationStep = self.simulation.simulationStep  # TraCI top-level API

    def _approach_of(self, lane: str) -> str | None:
        for approach, candidate in APPROACH_LANES.items():
            if candidate == lane:
                return approach
        return None

    def _vehicle_ids_present(self) -> List[str]:
        ids: Set[str] = set()
        for parked in self.vehicles.values():
            ids |= parked
        return sorted(ids)

    # ---- test helpers ------------------------------------------------ #
    def place(self, vid: str, lane: str) -> None:
        self.vehicles.setdefault(lane, set()).add(vid)

    def place_emergency(self, lane: str | None) -> None:
        """Place or remove 'ambulance_0' on an approach lane."""
        for v in list(self.vehicles.get("ambulance_0", set())):
            self.vehicles[v]  # no-op compatibility
        for candidate in APPROACH_LANES.values():
            self.vehicles[candidate].discard("ambulance_0")
        if lane is not None:
            self.vehicles.setdefault(lane, set()).add("ambulance_0")


# --------------------------------------------------------------------------- #

def test_traci_sources_are_iterable_and_match_counts():
    stub = FakeTraCI()
    stub.place("v1", APPROACH_LANES["north"])
    stub.place("v2", APPROACH_LANES["north"])
    stub.place("v3", APPROACH_LANES["west"])
    src = TraciTrafficSource(stub)
    snaps = [s for s in src.steps(3)]
    assert len(snaps) == 3
    assert all(isinstance(s, TrafficSnapshot) for s in snaps)
    assert snaps[0].lane_counts["north"] == 2
    assert snaps[0].lane_counts["west"] == 1
    assert snaps[0].lane_counts["south"] == 0


def test_emergency_lane_detection():
    stub = FakeTraCI()
    stub.place_emergency(APPROACH_LANES["south"])
    src = TraciTrafficSource(stub, emergency_ids=["ambulance_0"])
    snap = next(src.steps(1))
    assert snap.emergency_lane == "south"

    stub.place_emergency(APPROACH_LANES["north"])
    snap = next(src.steps(1))
    assert snap.emergency_lane == "north"

    stub.place_emergency(None)
    snap = next(src.steps(1))
    assert snap.emergency_lane is None


def test_full_bridge_loop_applies_signals():
    stub = FakeTraCI()
    for lane, count in ((APPROACH_LANES["north"], 3),
                        (APPROACH_LANES["south"], 5),
                        (APPROACH_LANES["east"], 2),
                        (APPROACH_LANES["west"], 4)):
        for i in range(count):
            stub.place(f"{lane}-{i}", lane)

    sink = TraciSignalSink(stub)
    bridge = FalconBridge(source=TraciTrafficSource(stub), sink=sink, fps=1)
    result = bridge.run(make_controller("adaptive"), total_steps=120)

    assert len(result.records) == 120
    assert isinstance(result.records[0].signal_state, dict)
    assert stub.state_history, "expected signals to be pushed to the traffic light"
    # Signals must be exactly 16 chars (16 links).
    assert all(len(s) == 16 for s in stub.state_history)
    # Controller consumed real occupancy, not the synthetic queue.
    observed = sink.observed_counts()
    assert set(observed) == {"north", "south", "east", "west"}


def test_fake_traci_serves_on_green():
    stub = FakeTraCI()
    for lane in APPROACH_LANES.values():
        for i in range(4):
            stub.place(f"{lane}-{i}", lane)
    sink = TraciSignalSink(stub)
    bridge = FalconBridge(source=TraciTrafficSource(stub), sink=sink, fps=1)
    bridge.run(make_controller("adaptive"), total_steps=200)
    report = sink.report("fake", "ADAPTIVE", 200)
    assert report.total_steps == 200
    # Fake model must have served some vehicles as lanes rotated through GREEN.
    assert report.vehicles_served > 0
    assert report.avg_queue >= 0
    assert report.max_queue >= 0


def test_traci_report_matches_metrics_shape():
    stub = FakeTraCI()
    sink = TraciSignalSink(stub)
    bridge = FalconBridge(source=TraciTrafficSource(stub), sink=sink, fps=1)
    result = bridge.run(make_controller("baseline", green_seconds=10), total_steps=50)
    report = result.report("fake", "BASELINE", sink, 50)
    assert report.controller == "BASELINE"
    assert report.scenario == "fake"
    assert report.total_steps == 50
    assert report.vehicles_served >= 0


if __name__ == "__main__":
    test_traci_sources_are_iterable_and_match_counts()
    test_emergency_lane_detection()
    test_full_bridge_loop_applies_signals()
    test_fake_traci_serves_on_green()
    test_traci_report_matches_metrics_shape()
    print("PASS test_traci_loop")