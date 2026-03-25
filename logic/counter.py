
class LaneCounter:
    """
    Divides the video frame into 4 directional lane zones and counts
    how many detected vehicles fall into each zone per frame.

    Input:  vehicles list from VehicleDetector.detect(frame)['vehicles']
    Output: {'north': int, 'south': int, 'east': int, 'west': int}
    """

    def __init__(self, frame_width, frame_height, *, mode="top_down", camera_lane="north"):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.mid_x = frame_width / 2.0
        self.mid_y = frame_height / 2.0
        self.lanes = ("north", "south", "east", "west")
        self.mode = str(mode).lower()
        lane = str(camera_lane).lower()
        self.camera_lane = lane if lane in self.lanes else "north"

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
            counts[self.camera_lane] = len(vehicles)
            return counts

        for vehicle in vehicles:
            lane = self.assign_lane(vehicle['bbox'])
            if lane in counts:
                counts[lane] += 1

        return counts

