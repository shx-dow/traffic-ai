
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vision.detector import VehicleDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ambulance detection on one image")
    parser.add_argument("--image", type=Path, required=True, help="BGR image path (jpg/png)")
    parser.add_argument(
        "--ambulance-mode",
        choices=("none", "yolo_world", "aux_weights"),
        default=None,
        help="Override config AMBULANCE_DETECTION_MODE",
    )
    parser.add_argument("--json", action="store_true", help="Print full detection JSON")
    parser.add_argument(
        "--ambulance-conf",
        type=float,
        default=None,
        help="Override AMBULANCE_CONFIDENCE (try 0.15–0.2 if YOLOWorld misses)",
    )
    args = parser.parse_args()

    path = args.image.expanduser()
    if not path.is_file():
        print(f"Error: not a file: {path}", file=sys.stderr)
        return 1

    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Error: could not read image: {path}", file=sys.stderr)
        return 1

    kwargs = {}
    if args.ambulance_mode is not None:
        kwargs["ambulance_mode"] = args.ambulance_mode
    if args.ambulance_conf is not None:
        kwargs["ambulance_confidence"] = args.ambulance_conf

    detector = VehicleDetector(**kwargs)
    result = detector.detect(frame, video_source_hint=str(path.resolve()))

    amb = [v for v in result["vehicles"] if v["class"] == "ambulance"]
    print(f"emergency={result['emergency']}")
    print(f"total_vehicles={result['count']}  ambulance_boxes={len(amb)}")
    print("Per-box (COCO + optional ambulance head):")
    for v in result["vehicles"]:
        print(f"  {v['class']}: confidence={v['confidence']:.3f}")
    if amb:
        print("ambulance detections:")
        for a in amb:
            print(f"  conf={a['confidence']:.3f} bbox={a['bbox']}")
    else:
        print(
            "No ambulance from YOLOWorld/aux above threshold.\n"
            "  • COCO often labels ambulances as truck/van/car — see line(s) above.\n"
            "  • Retry: python scripts/validate_ambulance.py --image \"...\" --ambulance-conf 0.15\n"
            "  • Or edit AMBULANCE_CONFIDENCE in config.py, or use a dedicated ambulance.pt (aux_weights)."
        )

    if args.json:
        out = {k: v for k, v in result.items() if k != "raw_result"}
        print(json.dumps(out, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
