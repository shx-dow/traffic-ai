
class LaneCounter:
    """
    Divides the video frame into 4 directional lane zones and counts
    how many detected vehicles fall into each zone per frame.

    Input:  vehicles list from VehicleDetector.detect(frame)['vehicles']
    Output: {'north': int, 'south': int, 'east': int, 'west': int}
    """

    def __init__(
        self,
        frame_width,
        frame_height,
        *,
        mode="top_down",
        camera_lane="north",
        approach_roi=None,
        queue_roi=None,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.mid_x = frame_width / 2.0
        self.mid_y = frame_height / 2.0
        self.lanes = ("north", "south", "east", "west")
        self.mode = str(mode).lower()
        lane = str(camera_lane).lower()
        self.camera_lane = lane if lane in self.lanes else "north"
        self.approach_roi = self._clamp_roi(approach_roi)
        self.queue_roi = self._clamp_roi(queue_roi)
        self.last_queue_length = 0
        self.last_approach_count = 0

    def assign_lane(self, bbox):
        """
        Given a vehicle's [x1, y1, x2, y2] bounding box,
        returns which lane zone its center point falls in.

        Returns: 'north' | 'south' | 'east' | 'west' | 'unknown'
        """
        if self.mode == "per_camera":
            return self.camera_lane

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        dx = cx - self.mid_x
        dy = cy - self.mid_y

        if abs(dx) >= abs(dy):
            return 'east' if dx >= 0 else 'west'
        return 'south' if dy >= 0 else 'north'

    def count_per_lane(self, vehicles):
        """
        Counts vehicles per lane zone.

        Input:  vehicles list from detector.detect(frame)['vehicles']
        Output: {'north': int, 'south': int, 'east': int, 'west': int}
        """
        counts = {'north': 0, 'south': 0, 'east': 0, 'west': 0}

        if self.mode == "per_camera":
            approach_vehicles = vehicles
            if self.approach_roi is not None:
                approach_vehicles = [v for v in vehicles if self._vehicle_in_roi(v, self.approach_roi)]

            queue_count = 0
            if self.queue_roi is not None:
                queue_count = sum(1 for v in approach_vehicles if self._vehicle_in_roi(v, self.queue_roi))
            else:
                queue_count = len(approach_vehicles)

            self.last_approach_count = len(approach_vehicles)
            self.last_queue_length = queue_count
            counts[self.camera_lane] = self.last_approach_count
            return counts

        for vehicle in vehicles:
            lane = self.assign_lane(vehicle['bbox'])
            if lane in counts:
                counts[lane] += 1

        self.last_approach_count = sum(counts.values())
        self.last_queue_length = max(counts.values()) if counts else 0
        return counts

    def _vehicle_in_roi(self, vehicle, roi):
        bbox = vehicle.get('bbox') if isinstance(vehicle, dict) else None
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        cx = (float(bbox[0]) + float(bbox[2])) / 2.0
        cy = (float(bbox[1]) + float(bbox[3])) / 2.0
        x1, y1, x2, y2 = roi
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _clamp_roi(self, roi):
        if roi is None:
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in roi]
        except Exception:
            return None
        x1 = max(0, min(self.frame_width - 1, x1))
        y1 = max(0, min(self.frame_height - 1, y1))
        x2 = max(x1 + 1, min(self.frame_width, x2))
        y2 = max(y1 + 1, min(self.frame_height, y2))
        return (x1, y1, x2, y2)

