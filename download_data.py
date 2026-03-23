import gdown
import os
import zipfile

# Create folders
os.makedirs("assets/models", exist_ok=True)
os.makedirs("assets/real_time_traffic", exist_ok=True)

# Download YOLO weights
gdown.download(
    "https://drive.google.com/uc?id=<YOLOV8N_FILE_ID>",
    "assets/models/yolov8n.pt",
    quiet=False
)
gdown.download(
    "https://drive.google.com/uc?id=<YOLOV8S_WORLD_FILE_ID>",
    "assets/models/yolov8s-worldv2.pt",
    quiet=False
)

# Download traffic dataset zip
gdown.download(
    "https://drive.google.com/uc?id=<TRAFFIC_DATASET_ID>",
    "assets/real_time_traffic.zip",
    quiet=False
)

# Extract traffic dataset
zip_path = "assets/real_time_traffic.zip"
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("assets/real_time_traffic")
    print("Traffic dataset extracted successfully.")
else:
    print("Traffic dataset zip not found.")
