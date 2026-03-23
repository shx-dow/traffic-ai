
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.findvehicle import aggregate_conll, aggregate_jsonl, iter_jsonl, map_entity_hints_to_yolo_classes, spans_from_jsonl_record


def main() -> None:
    parser = argparse.ArgumentParser(description="FindVehicle dataset traffic analytics")
    parser.add_argument("--jsonl", type=Path, help="Path to FindVehicle_*.jsonl")
    parser.add_argument("--conll", type=Path, help="Path to FindVehicle_*.txt (CoNLL)")
    parser.add_argument("--limit", type=int, default=None, help="Max records (jsonl) or sentences (conll)")
    parser.add_argument(
        "--demo-yolo-hints",
        action="store_true",
        help="On first jsonl record, print heuristic YOLO-class hints from vehicle_type spans",
    )
    args = parser.parse_args()

    if not args.jsonl and not args.conll:
        default = ROOT / "assets" / "findvehicle" / "sample.jsonl"
        if default.is_file():
            args.jsonl = default
            print(f"Using bundled sample: {default}\n", file=sys.stderr)
        else:
            parser.error("Pass --jsonl or --conll (or add assets/findvehicle/sample.jsonl).")

    if args.jsonl:
        path = args.jsonl.expanduser()
        if not path.is_file():
            print(
                f"Error: file not found: {path}\n"
                "  FindVehicle_train.jsonl is not in this repo — download it from the authors:\n"
                "  https://github.com/GuanRunwei/FindVehicle (Baidu / Google Drive links in README).\n"
                "  Then: python scripts/analyze_findvehicle.py --jsonl /path/to/FindVehicle_train.jsonl\n"
                "  For a quick test without downloading, use: --jsonl assets/findvehicle/sample.jsonl",
                file=sys.stderr,
            )
            raise SystemExit(1)
        summary = aggregate_jsonl(path, limit=args.limit)
        print(json.dumps(summary, indent=2))
        if args.demo_yolo_hints:
            first = next(iter_jsonl(path))
            spans = spans_from_jsonl_record(first)
            hints = map_entity_hints_to_yolo_classes(spans)
            print("\nDemo (first record) YOLO hints from text:", hints, file=sys.stderr)
            print("Text:", first.get("data", "")[:200], file=sys.stderr)

    if args.conll:
        path = args.conll.expanduser()
        if not path.is_file():
            print(
                f"Error: file not found: {path}\n"
                "  Download FindVehicle_*.txt (CoNLL) from https://github.com/GuanRunwei/FindVehicle",
                file=sys.stderr,
            )
            raise SystemExit(1)
        summary = aggregate_conll(path, limit=args.limit)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
