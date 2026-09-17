"""validate_detector.py — ground-truth accuracy scoring for the detector.

Runs VehicleDetector over a recorded traffic video and compares the detected
vehicle count per frame (total, or per-approach via ROI labels) against
manually labelled ground truth.

Ground-truth labels file (JSON):
    {"0": {"north": 2, "south": 3, "east": 0, "west": 1}, "5": {...}, ...}
Frame keys are video frame indices; values are per-approach vehicle counts
(summed when no ROI/approach split is needed).

Usage:
    python scripts/validate_detector.py --video data/traffic.mp4 \
        --labels data/gt_labels.json --model yolov8n.pt \
        --out profiling/artifacts/detector_accuracy.json [--max-frames 300]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

APPROACHES = ("north", "south", "east", "west")


def _total(entry) -> int:
    if isinstance(entry, int):
        return entry
    return sum(int(entry.get(lane, 0)) for lane in APPROACHES)


def compute_accuracy(
    predicted: dict[int, int],
    ground_truth: dict[int, int],
    tolerance: int = 1,
) -> dict:
    """Pure metric computation (importable without cv2/ultralytics).

    - MAE / RMSE over all frames that exist in the labels
    - recall: share of labelled frames whose prediction is within `tolerance`
      vehicles of the labelled count
    - count bias: mean error (positive = over-counting)
    """
    frames = sorted(ground_truth)
    if not frames:
        return {"frames": 0, "mae": None, "rmse": None, "within_tolerance": None, "bias": None}

    errors = [predicted.get(f, 0) - ground_truth[f] for f in frames]
    abs_errors = [abs(e) for e in errors]
    mae = sum(abs_errors) / len(errors)
    rmse = (sum(e * e for e in errors) / len(errors)) ** 0.5
    within = sum(1 for e in errors if abs(e) <= tolerance) / len(errors)
    bias = sum(errors) / len(errors)
    return {
        "frames": len(frames),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "tolerance_veh": tolerance,
        "within_tolerance": round(within, 3),
        "bias": round(bias, 3),
    }


def load_labels(path: Path) -> dict[int, dict]:
    """Frame-index -> per-approach counts. Aggregate int entries are treated as
    a single virtual approach so the metric code stays uniform."""
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = {}
    for key, entry in data.items():
        frame = int(key)
        if isinstance(entry, int):
            labels[frame] = {"north": int(entry), "south": 0, "east": 0, "west": 0}
        else:
            labels[frame] = {lane: int(entry.get(lane, 0)) for lane in APPROACHES}
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the detector against ground truth")
    parser.add_argument("--video", required=True, help="recorded traffic video (BGR)")
    parser.add_argument("--labels", required=True, help="JSON ground-truth counts per frame")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights (if not already cached)")
    parser.add_argument("--max-frames", type=int, default=None, help="cap the number of frames processed")
    parser.add_argument("--out", default="profiling/artifacts/detector_accuracy.json")
    args = parser.parse_args()


    from vision.detector import VehicleDetector

    video_path = Path(args.video)
    labels_path = Path(args.labels)
    if not video_path.is_file():
        print(f"ERROR: video not found: {video_path}", file=sys.stderr)
        return 2
    if not labels_path.is_file():
        print(f"ERROR: labels not found: {labels_path}", file=sys.stderr)
        return 2

    labels = load_labels(labels_path)
    detector = VehicleDetector(args.model)

    import cv2

    cap = cv2.VideoCapture(str(video_path.resolve()))
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {video_path}", file=sys.stderr)
        return 2

    predicted: dict[int, int] = {}
    frame_index = 0
    while frame_index <= max((labels or {0}).keys()):
        if args.max_frames and frame_index >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index in labels:
            detections = detector.detect(frame)
            predicted[frame_index] = int(detections["count"])
        frame_index += 1
    cap.release()

    metrics = compute_accuracy(predicted, {f: _total(labels[f]) for f in labels})
    print("Detector accuracy vs ground truth")
    print(f"  labelled frames : {metrics['frames']}")
    print(f"  MAE             : {metrics['mae']} vehicles/frame")
    print(f"  RMSE            : {metrics['rmse']}")
    print(f"  within +/-{metrics['tolerance_veh']} veh: {metrics['within_tolerance']}")
    print(f"  bias            : {metrics['bias']}")

    out_path = (Path(__file__).resolve().parent.parent / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"video": str(video_path), "metrics": metrics}, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
