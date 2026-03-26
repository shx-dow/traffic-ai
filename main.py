
from __future__ import annotations

import argparse
import json
import logging
import signal
import subprocess
import sys
import time
from pathlib import Path

import cv2

from config import (BASELINE_GREEN_SECONDS, CONFIG, DEFAULT_SIGNAL_MODE,
                    EMERGENCY_LATCH_SECONDS, MODEL_PATH)
from logic.baseline_signal import BaselineSignalController
from logic.counter import LaneCounter
from logic.live_metrics import LiveMetricsTracker
from logic.orchestrator import CorridorOrchestrator
from logic.roi import parse_rect_roi
from logic.runtime import select_corridor_lane
from logic.signal import SignalController
from logic.signal_state import (SignalStateSensor, parse_roi_arg,
                                resolve_signal_reading)
from logic.traffic_loop import build_signal_summary, is_balanced
from ui.overlay import TrafficOverlay
from utils.helpers import run_repo_pipeline
from vision.detector import VehicleDetector

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Traffic Management — main runtime")
    p.add_argument("--video-source", default=CONFIG["video_source"],
                   help="Video source (0 for webcam or path to video file)")
    p.add_argument("--model-path", default=CONFIG.get("model_path", MODEL_PATH),
                   help="Path to YOLO model weights")
    p.add_argument("--frame-width", type=int, default=CONFIG["frame_width"])
    p.add_argument("--frame-height", type=int, default=CONFIG["frame_height"])
    p.add_argument("--camera-lane", choices=("north", "south", "east", "west"), default=CONFIG.get("camera_lane", "north"), help="Lane represented by this camera in per-camera mode")
    p.add_argument("--top-down-view", action="store_true", help="Use center-split top-down lane assignment instead of per-camera mode")
    p.add_argument("--mode", choices=("adaptive", "baseline"), default=DEFAULT_SIGNAL_MODE)
    p.add_argument("--baseline-green-seconds", type=int, default=BASELINE_GREEN_SECONDS)
    p.add_argument("--headless", action="store_true",
                    help="Run without opening a display window (useful for CI)")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 means no limit)")
    p.add_argument("--save-output", action="store_true", default=CONFIG.get("save_output", False))
    p.add_argument("--output-path", default=CONFIG.get("output_path", "artifacts/demo_output.mp4"))
    p.add_argument("--run-pipeline", action="store_true", help="Run core repo validation checks before main loop")
    p.add_argument("--hide-kpi-hud", action="store_true", help="Disable the live KPI panel overlay")
    p.add_argument("--metrics-log-path", default=CONFIG.get("metrics_log_path", ""), help="Write live KPI snapshots to JSONL")
    p.add_argument("--metrics-log-every", type=int, default=int(CONFIG.get("metrics_log_every", 30)), help="Log KPIs every N frames")
    p.add_argument("--signal-state-source", choices=("none", "video", "api"), default=CONFIG.get("signal_state_source", "none"), help="External signal state source for credibility overlay")
    p.add_argument("--signal-state-api-url", default=CONFIG.get("signal_state_api_url", ""), help="API endpoint returning {state, confidence}")
    p.add_argument("--signal-state-roi", default=CONFIG.get("signal_state_roi", ""), help="Traffic light ROI as x1,y1,x2,y2 for video signal sensing")
    p.add_argument("--signal-state-api-timeout", type=float, default=float(CONFIG.get("signal_state_api_timeout", 0.4)), help="Signal state API timeout seconds")
    p.add_argument("--signal-state-fallback", choices=("none", "controller"), default="controller", help="Fallback source when signal sensing is unavailable")
    p.add_argument("--show-signal-roi", action="store_true", help="Draw the traffic-light ROI box when using video signal sensing")
    p.add_argument("--approach-roi", default=CONFIG.get("approach_roi", ""), help="Per-camera approach ROI x1,y1,x2,y2")
    p.add_argument("--queue-roi", default=CONFIG.get("queue_roi", ""), help="Per-camera stop-line queue ROI x1,y1,x2,y2")
    p.add_argument("--show-count-roi", action="store_true", help="Draw counting ROIs for approach and queue zones")
    p.add_argument("--ui-mode", choices=("demo", "debug"), default="demo", help="Overlay verbosity mode")
    p.add_argument("--emergency-source", choices=("vision", "gps", "fusion", "manual"), default="fusion", help="Emergency trigger mode")
    p.add_argument("--enable-orchestrator", action="store_true", help="Enable route-aware corridor pre-clear planning")
    p.add_argument("--orchestrator-route", default="int_a,int_b,int_c,int_d", help="Comma separated route node ids")
    p.add_argument("--orchestrator-node-id", default="int_a", help="Current node id for this runtime instance")
    p.add_argument("--orchestrator-preempt-hops", type=int, default=2, help="How many downstream intersections to pre-clear")
    p.add_argument(
        "--disable-gps",
        action="store_true",
        help="Do not start gps_server and do not run GPS integration tests",
    )
    return p.parse_args()


def open_capture(source) -> cv2.VideoCapture:
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except (TypeError, ValueError):
        cap = cv2.VideoCapture(str(source))
    return cap


def compute_lane_scores(signal_ctrl, lane_counts: dict[str, int]) -> dict[str, float]:
    if hasattr(signal_ctrl, "calculate_congestion_scores"):
        return signal_ctrl.calculate_congestion_scores(lane_counts)
    return {lane: float(count) for lane, count in lane_counts.items()}


def get_capture_dimensions(cap: cv2.VideoCapture, default_width: int, default_height: int) -> tuple[int, int]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0:
        width = default_width
    if height <= 0:
        height = default_height
    return width, height


class GracefulExit:
    def __init__(self):
        self.exit = False

    def __call__(self, signum, frame):
        LOGGER.info("Received signal %s, exiting gracefully", signum)
        self.exit = True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    gps_proc = None
    if args.run_pipeline:
        gps_proc = run_repo_pipeline(
            disable_gps=args.disable_gps,
        )

    # Setup graceful shutdown handlers
    ge = GracefulExit()
    signal.signal(signal.SIGINT, ge)
    try:
        signal.signal(signal.SIGTERM, ge)
    except (AttributeError, ValueError):
        pass

    cap = open_capture(args.video_source)
    if not cap.isOpened():
        LOGGER.error("Failed to open video source %s", args.video_source)
        sys.exit(1)

    frame_width, frame_height = get_capture_dimensions(cap, args.frame_width, args.frame_height)

    detector = VehicleDetector(model_path=args.model_path)
    counter_mode = "top_down" if args.top_down_view else str(CONFIG.get("counter_mode", "per_camera"))
    per_camera_mode = counter_mode == "per_camera"
    show_directional_counts = counter_mode == "top_down"
    approach_roi = parse_rect_roi(args.approach_roi)
    queue_roi = parse_rect_roi(args.queue_roi)
    counter = LaneCounter(
        frame_width,
        frame_height,
        mode=counter_mode,
        camera_lane=args.camera_lane,
        approach_roi=approach_roi,
        queue_roi=queue_roi,
    )
    overlay = TrafficOverlay()
    signal_sensor = SignalStateSensor(
        source=args.signal_state_source,
        api_url=args.signal_state_api_url,
        roi=parse_roi_arg(args.signal_state_roi),
        timeout_seconds=args.signal_state_api_timeout,
    )

    if args.mode == "baseline":
        signal_ctrl = BaselineSignalController(green_seconds=args.baseline_green_seconds)
    else:
        signal_ctrl = SignalController()

    lanes = ["north", "south", "east", "west"]
    active_lane = lanes[0]
    frame_counter = 0
    processed_frames = 0
    emergency_hold_frames = 0
    last_corridor_lane = active_lane
    manual_emergency = False
    emergency_progress_frames = 0

    route_nodes = [node.strip() for node in str(args.orchestrator_route).split(",") if node.strip()]
    orchestrator_enabled = bool(args.enable_orchestrator and route_nodes)
    if args.orchestrator_node_id not in route_nodes and route_nodes:
        route_nodes.insert(0, args.orchestrator_node_id)
    orchestrator = None
    if orchestrator_enabled:
        orchestrator = CorridorOrchestrator(
            route_nodes,
            preempt_hops=max(0, int(args.orchestrator_preempt_hops)),
            latch_frames=max(1, int(EMERGENCY_LATCH_SECONDS * 30)),
        )
    orchestrator_emergency_nodes = 0

    fps = 30
    metrics = LiveMetricsTracker(fps=fps)
    frame_delay_ms = int(1000 / fps)
    display_window = (not args.headless) and bool(CONFIG.get("display_window", True))
    show_kpi_hud = bool(CONFIG.get("show_kpi_hud", True)) and (not args.hide_kpi_hud)

    metrics_log_path = str(args.metrics_log_path or "").strip()
    metrics_log_every = max(1, int(args.metrics_log_every))
    metrics_log_file = None
    if metrics_log_path:
        log_path = Path(metrics_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_log_file = log_path.open("w", encoding="utf-8")

    writer = None
    if args.save_output:
        out_path = Path(args.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (frame_width, frame_height))
        if not writer.isOpened():
            LOGGER.warning("Failed to open video writer at %s", out_path)
            writer = None

    LOGGER.info("Starting main loop in %s mode. Press 'q' in window to quit.", args.mode)

    while not ge.exit and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            LOGGER.info("End of stream or cannot read frame")
            break

        detection = detector.detect(frame)

        lane_counts = counter.count_per_lane(detection["vehicles"])
        lane_scores = compute_lane_scores(signal_ctrl, lane_counts)
        sensed_signal_raw = signal_sensor.read(frame)
        emergency_seen = bool(detection.get("emergency"))
        gps_emergency = bool(detection.get("gps_emergency"))
        emergency_source = "none"

        key = cv2.waitKey(1) & 0xFF if display_window else 255
        if key == ord("e"):
            manual_emergency = not manual_emergency
            LOGGER.info("Manual emergency toggled: %s", manual_emergency)
            if manual_emergency:
                emergency_hold_frames = int(max(1, EMERGENCY_LATCH_SECONDS * fps))
        elif key == ord("q"):
            LOGGER.info("'q' pressed — exiting")
            break

        emergency_inputs = {
            "vision": emergency_seen,
            "gps": gps_emergency,
            "fusion": bool(emergency_seen or gps_emergency),
            "manual": manual_emergency,
        }
        emergency_triggered = emergency_inputs.get(args.emergency_source, False)
        if manual_emergency:
            emergency_source = "manual"
        elif emergency_seen and gps_emergency:
            emergency_source = "vision+gps"
        elif emergency_seen:
            emergency_source = "vision"
        elif gps_emergency:
            emergency_source = "gps"
        if emergency_seen:
            emergency_hold_frames = int(max(1, EMERGENCY_LATCH_SECONDS * fps))
        elif emergency_hold_frames > 0:
            emergency_hold_frames -= 1

        emergency_active = emergency_triggered or emergency_hold_frames > 0
        force_all_green = manual_emergency
        if emergency_active and not force_all_green:
            emergency_progress_frames += 1
        else:
            emergency_progress_frames = 0

        corridor = last_corridor_lane
        if emergency_active and not force_all_green:
            corridor = select_corridor_lane(
                vehicles=detection["vehicles"],
                lane_counts=lane_counts,
                lane_counter=counter,
                fallback_lane=active_lane,
                last_corridor_lane=last_corridor_lane,
            )

        if orchestrator is not None:
            if emergency_active and not force_all_green:
                position_index = min(len(route_nodes) - 1, emergency_progress_frames // max(1, fps * 2))
                plan = orchestrator.update(
                    route=route_nodes,
                    position_index=position_index,
                    ambulance_lane=corridor,
                )
            else:
                plan = orchestrator.update(route=None, position_index=None, ambulance_lane=None)
            node_plan = plan.get(args.orchestrator_node_id)
            if node_plan is not None and node_plan.mode == "EMERGENCY" and node_plan.corridor_lane:
                corridor = node_plan.corridor_lane
            orchestrator_emergency_nodes = sum(1 for p in plan.values() if p.mode == "EMERGENCY")

        if emergency_active and signal_ctrl.mode != "EMERGENCY":
            if force_all_green:
                signal_ctrl.override_all_green()
            else:
                last_corridor_lane = corridor
                signal_ctrl.override_for_emergency(corridor)

        if emergency_active and (not force_all_green) and signal_ctrl.mode == "EMERGENCY" and emergency_seen:
            if corridor != last_corridor_lane:
                last_corridor_lane = corridor
                signal_ctrl.override_for_emergency(corridor)

        if not emergency_active and signal_ctrl.mode == "EMERGENCY":
            signal_ctrl.resume_adaptive()
            frame_counter = 0

        if signal_ctrl.mode == "EMERGENCY":
            signal_states = signal_ctrl.current_state
            decision_reason = f"Emergency override ({emergency_source})"
        else:
            signal_states = signal_ctrl.get_current_signal_state(active_lane)
            signal_ctrl.record_cycle(active_lane, lane_counts)

            should_switch = signal_ctrl.should_switch_lane(
                active_lane=active_lane,
                lane_counts=lane_scores,
                frame_counter=frame_counter,
                fps=fps,
            )

            if should_switch:
                active_lane = signal_ctrl.choose_next_lane(active_lane, lane_scores)
                frame_counter = 0
                decision_reason = f"Switched to {active_lane} (higher congestion score)"
            else:
                frame_counter += 1
                balance_gap = float(getattr(signal_ctrl, "CONGESTION_BALANCE_GAP", 2.5))
                balanced = is_balanced(lane_scores.values(), balance_gap)
                decision_reason = (
                    f"Holding {active_lane} (balanced flow)"
                    if balanced
                    else f"Holding {active_lane} (min hold/score gap)"
                )

        sensed_signal = resolve_signal_reading(
            sensed_signal_raw,
            requested_source=args.signal_state_source,
            fallback_mode=args.signal_state_fallback,
            signal_states=signal_states,
            camera_lane=args.camera_lane,
        )

        green_lanes = [lane for lane, state in signal_states.items() if state == "GREEN"]
        metrics.update(lane_counts=lane_counts, green_lanes=green_lanes, mode=signal_ctrl.mode)
        kpi_snapshot = metrics.snapshot()

        if metrics_log_file is not None and (processed_frames % metrics_log_every == 0):
            row = {
                "frame": processed_frames,
                "mode": signal_ctrl.mode,
                "emergency_active": emergency_active,
                "active_lane": active_lane,
                "lane_counts": lane_counts,
                "lane_scores": lane_scores,
                "decision_reason": decision_reason,
                "emergency_source": emergency_source,
                "orchestrator_emergency_nodes": orchestrator_emergency_nodes,
                **kpi_snapshot,
            }
            gps_priority = detection.get("gps_priority")
            if isinstance(gps_priority, dict):
                row["gps_priority"] = gps_priority
            if sensed_signal_raw is not None:
                row["sensed_signal"] = {
                    "state": sensed_signal_raw.state,
                    "source": sensed_signal_raw.source,
                    "confidence": sensed_signal_raw.confidence,
                }
            if sensed_signal is not None and sensed_signal.source == "controller_fallback":
                row["sensed_signal_fallback"] = {
                    "state": sensed_signal.state,
                    "source": sensed_signal.source,
                    "confidence": sensed_signal.confidence,
                }
            metrics_log_file.write(json.dumps(row) + "\n")

        display = overlay.draw(
            frame.copy(),
            detection_result=detection,
            lane_counts=lane_counts,
            lane_scores=lane_scores,
            signal_states=signal_states,
            emergency_active=emergency_active,
            kpi_snapshot=(kpi_snapshot if show_kpi_hud else None),
            show_directional_counts=show_directional_counts,
            camera_lane=args.camera_lane,
            ui_mode=args.ui_mode,
            per_camera_mode=per_camera_mode,
        )
        y_text = 30
        cv2.putText(display, f"Mode: {signal_ctrl.mode}", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_text += 34
        if args.ui_mode == "debug" and show_directional_counts:
            cv2.putText(display, f"Counts: N{lane_counts['north']} S{lane_counts['south']} E{lane_counts['east']} W{lane_counts['west']}", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            approach_count = int(lane_counts.get(args.camera_lane, 0))
            score = float(lane_scores.get(args.camera_lane, 0.0))
            cv2.putText(display, f"Approach {args.camera_lane.title()} Count: {approach_count}  Score: {score:.1f}  Queue: {counter.last_queue_length}", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 0), 2)
        y_text += 36
        if sensed_signal is not None:
            if sensed_signal.source == "controller_fallback":
                cv2.putText(display, f"Observed signal: {sensed_signal.state} (fallback: controller)", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 230, 255), 2)
            else:
                cv2.putText(display, f"Observed signal: {sensed_signal.state} ({sensed_signal.source}, {sensed_signal.confidence:.2f})", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2)
        elif args.signal_state_source != "none":
            cv2.putText(display, f"Observed signal: UNKNOWN ({args.signal_state_source})", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 220, 255), 2)
        y_text += 32

        if emergency_active:
            cv2.putText(display, "EMERGENCY MODE ACTIVE", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            y_text += 30
            cv2.putText(display, f"Emergency source: {emergency_source}", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 255), 2)
            y_text += 28

        signal_summary = build_signal_summary(signal_states)
        cv2.putText(display, f"Signal output: {signal_summary}", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (215, 245, 255), 2)
        y_text += 28
        cv2.putText(display, f"Decision: {decision_reason}", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (215, 245, 255), 2)
        y_text += 32
        if orchestrator is not None:
            cv2.putText(display, f"Route pre-clear nodes: {orchestrator_emergency_nodes}", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 245, 220), 2)
            y_text += 28

        gps_priority = detection.get("gps_priority") if isinstance(detection, dict) else None
        if isinstance(gps_priority, dict) and gps_priority.get("emergency"):
            eta = gps_priority.get("eta_seconds")
            dist = gps_priority.get("distance_km")
            vid = gps_priority.get("vehicle_id")
            eta_text = f"{float(eta):.1f}s" if isinstance(eta, (int, float)) else "-"
            dist_text = f"{float(dist):.2f}km" if isinstance(dist, (int, float)) else "-"
            cv2.putText(display, f"GPS priority: {vid or 'unknown'} ETA {eta_text} Dist {dist_text}", (10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 230, 180), 2)

        if args.show_signal_roi and args.signal_state_source == "video":
            roi = signal_sensor.get_effective_roi(frame)
            overlay.draw_signal_roi(display, roi)

        if args.show_count_roi and counter_mode == "per_camera" and args.ui_mode == "debug":
            if counter.approach_roi is not None:
                overlay.draw_counting_roi(display, counter.approach_roi, "Approach ROI", (80, 220, 255))
            if counter.queue_roi is not None:
                overlay.draw_counting_roi(display, counter.queue_roi, "Queue ROI", (255, 100, 100))

        if writer is not None:
            writer.write(display)

        processed_frames += 1
        if args.max_frames > 0 and processed_frames >= args.max_frames:
            LOGGER.info("Reached max frames: %s", args.max_frames)
            break

        if display_window:
            cv2.imshow("Traffic System", display)

        if args.headless:
            time.sleep(1.0 / fps)

    cap.release()
    if writer is not None:
        writer.release()

    if metrics_log_file is not None:
        metrics_log_file.close()

    if display_window:
        cv2.destroyAllWindows()

    if gps_proc is not None:
        LOGGER.info("Stopping gps_server.py...")
        gps_proc.terminate()
        try:
            gps_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gps_proc.kill()


if __name__ == "__main__":
    main()
