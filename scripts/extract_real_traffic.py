
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import REAL_TIME_TRAFFIC_ASSET_DIR, REAL_TRAFFIC_ZIP_NAME
from utils.video_sources import first_video_under


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract real_traffic.zip (local file only)")
    parser.add_argument(
        "--dir",
        type=Path,
        default=ROOT / REAL_TIME_TRAFFIC_ASSET_DIR,
        help="Folder containing real_traffic.zip",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Extract even if video files are already present under --dir",
    )
    args = parser.parse_args()

    args.dir = args.dir.resolve()
    zpath = args.dir / REAL_TRAFFIC_ZIP_NAME
    if not zpath.is_file():
        print(f"Error: missing {zpath}", file=sys.stderr)
        print(f"  Place {REAL_TRAFFIC_ZIP_NAME} in that folder, then re-run.", file=sys.stderr)
        return 1
    if not zipfile.is_zipfile(zpath):
        print(f"Error: not a valid zip: {zpath}", file=sys.stderr)
        return 1
    if not args.force and first_video_under(args.dir) is not None:
        print(
            f"Videos already found under {args.dir}; skipping extract. Use --force to extract anyway.",
            file=sys.stderr,
        )
        return 0

    print(f"Extracting {zpath} → {args.dir} …", file=sys.stderr, flush=True)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(args.dir)
    print("Done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
