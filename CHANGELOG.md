# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Implemented lane counting and signal control logic with headless runner and tests.
- Added GPS-based ambulance tracking and emergency status integration (`shx-02`).
- Added GPS server, utility scripts, and mobile tracker application files.
- Integrated `VehicleDetector` utilizing YOLOv8n and corresponding `test_detector` script.
- Added ambulance dataset YAML, scripts, and configuration updates.
- Tracked YOLO models (`yolov8n.pt` and `yolov8s-worldv2.pt`) with Git LFS.
- Initialized core scaffold and prototype structure with essential modules.

### Changed
- Refactored codebase to organize imports and enhance overall code readability across multiple files.
- Streamlined configuration, GPS server, and general codebase cleanup.
- Clarified dataset download instructions and updated links in the `README`.
- Generalized script parameters and updated configuration defaults for better stability.

### Fixed
- Fixed `.gitignore` to properly exclude environment, cache, and large unnecessary files.

### Documentation
- Added comprehensive GPS documentation and analysis scripts.
- Updated `README.md` with detailed information regarding models and dataset tracking.
