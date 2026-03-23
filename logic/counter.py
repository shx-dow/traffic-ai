
class LaneCounter:
    """
    Divides the video frame into 4 directional lane zones and counts
    how many detected vehicles fall into each zone per frame.

    Input:  vehicles list from VehicleDetector.detect(frame)['vehicles']
    Output: {'north': int, 'south': int, 'east': int, 'west': int}
    """

    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.mid_x = frame_width / 2.0
        self.mid_y = frame_height / 2.0

    def assign_lane(self, bbox):
        """
        Given a vehicle's [x1, y1, x2, y2] bounding box,
        returns which lane zone its center point falls in.

        Returns: 'north' | 'south' | 'east' | 'west' | 'unknown'
        """
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

        for vehicle in vehicles:
            lane = self.assign_lane(vehicle['bbox'])
            if lane in counts:
                counts[lane] += 1

        return counts

