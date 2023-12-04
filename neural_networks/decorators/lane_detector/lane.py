import numpy as np


class Lane:
    """
    A class representing a lane, in the form of a mask.
    This mask is a binary mask, where 1 represents the lane, and 0 represents the background.
    The points are in the pixel domain, and not in the real world domain.
    """
    def __init__(self, mask: np.ndarray):
        self._mask = mask

    @property
    def mask(self):
        return self._mask

    @property
    def get_max_point(self) -> np.ndarray:
        points = self.indices

        # print(points[0].shape)
        # print(points[1].shape)
        max_y = np.min(points[0])
        mapped_x = points[1][np.where(points[0] == max_y)]
        mean_x = np.mean(mapped_x)

        max_point = np.array([max_y, mean_x])

        return max_point

    @property
    def indices(self):
        return np.where(self._mask != 0)

    def __repr__(self):
        return f"Lane(mask={self.mask})"

    def __str__(self):
        return self.__repr__()


class Lanes:
    """
    A class representing the left and right lanes.
    These lanes constrain the ego vehicle's navigation.
    """
    def __init__(self, left_lane: np.ndarray, right_lane: np.ndarray):
        self._left_lane = Lane(left_lane)
        self._right_lane = Lane(right_lane)

    @property
    def left_lane(self):
        return self._left_lane

    @property
    def right_lane(self):
        return self._right_lane

    @property
    def get_lane_indices(self) -> (np.ndarray, np.ndarray):
        return self.left_lane.indices, self.right_lane.indices

    @property
    def get_center_point(self) -> np.ndarray:
        left_point = self.left_lane.get_max_point
        right_point = self.right_lane.get_max_point

        center_point = (left_point + right_point) // 2

        return center_point

    def __repr__(self):
        return f"Lanes(left_lane={self.left_lane}, right_lane={self.right_lane})"

    def __str__(self):
        return self.__repr__()
