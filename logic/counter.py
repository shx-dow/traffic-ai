
class LaneCounter:
    """
    Divides the video frame into 4 directional lane zones and counts
    how many detected vehicles fall into each zone per frame.

    Input:  vehicles list from VehicleDetector.detect(frame)['vehicles']
    Output: {'north': int, 'south': int, 'east': int, 'west': int}
    """

    def __init__(self, frame_width, frame_height):
        mid_x = frame_width // 2
        mid_y = frame_height // 2

        # Each zone: (x1, y1, x2, y2) — top-left to bottom-right in pixels
        self.lanes = {
            'north': (0, 0, frame_width, mid_y),
            'south': (0, mid_y, frame_width, frame_height),
            'east': (mid_x, 0, frame_width, frame_height),
            'west': (0, 0, mid_x, frame_height),
        }

    def assign_lane(self, bbox):
        """
        Given a vehicle's [x1, y1, x2, y2] bounding box,
        returns which lane zone its center point falls in.

        Returns: 'north' | 'south' | 'east' | 'west' | 'unknown'
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        for lane, (zx1, zy1, zx2, zy2) in self.lanes.items():
            if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                return lane

        return 'unknown'

    def count_per_lane(self, vehicles):
        """
        Counts vehicles per lane zone.

        Input:  vehicles list from detector.detect(frame)['vehicles']
        Output: {'north': int, 'south': int, 'east': int, 'west': int}
        """
        counts = {'north': 0, 'south': 0, 'east': 0, 'west': 0}

        for vehicle in vehicles:
            lane = self.assign_lane(vehicle['bbox'])
            if lane in counts:
                counts[lane] += 1

        return counts

