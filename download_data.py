from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path


def ensure_dirs() -> None:
    Path("assets/models").mkdir(parents=True, exist_ok=True)
    Path("assets/real_time_traffic").mkdir(parents=True, exist_ok=True)


def download_file(file_id: str, output_path: str) -> bool:
    import gdown

    if not file_id or file_id.startswith("<"):
        print(f"Skipping {output_path}: missing file id")
        return False
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    return Path(output_path).is_file()


def extract_zip(zip_path: Path, output_dir: Path) -> bool:
    if not zip_path.is_file():
        print(f"Skip extract: zip not found at {zip_path}")
        return False
    if not zipfile.is_zipfile(zip_path):
        print(f"Skip extract: invalid zip file {zip_path}")
        return False
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Extracted {zip_path} to {output_dir}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download models and traffic dataset from Google Drive")
    parser.add_argument("--yolov8n-id", default=os.getenv("YOLOV8N_FILE_ID", ""))
    parser.add_argument("--yolov8s-world-id", default=os.getenv("YOLOV8S_WORLD_FILE_ID", ""))
    parser.add_argument("--traffic-dataset-id", default=os.getenv("TRAFFIC_DATASET_ID", ""))
    parser.add_argument("--extract", action="store_true", help="Extract traffic dataset zip after download")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_dirs()

    print("Starting downloads...")
    download_file(args.yolov8n_id, "assets/models/yolov8n.pt")
    download_file(args.yolov8s_world_id, "assets/models/yolov8s-worldv2.pt")
    dataset_downloaded = download_file(args.traffic_dataset_id, "assets/real_time_traffic/real_traffic.zip")

    if args.extract and dataset_downloaded:
        extract_zip(Path("assets/real_time_traffic/real_traffic.zip"), Path("assets/real_time_traffic"))

    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
