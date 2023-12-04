def set_working_directory():
    import os
    import sys
    import re

    script_dir = os.path.abspath(__file__)
    script_dir = re.sub(pattern="lane_detection.*", repl="lane_detection/", string=script_dir)
    script_dir = os.path.abspath(script_dir)
    os.chdir(script_dir)
    sys.path.append(os.path.join(script_dir))


set_working_directory()

from neural_networks.decorators.lane_detector.lane_detector import LaneDetector
from neural_networks.models.lanenet.enet import ENet
from utils.logger import Logger
import numpy as np
from typing import Optional
import torch
import cv2
from camera.calibrator import Calibrator
from camera.camera import Camera
from data.loaders.loader import Loader


class API:
    """
    Facade for the lane detection module.
    """
    Camera = Camera
    Loader = Loader

    def __init__(self, camera: Camera):
        self._detector = LaneDetector(ENet(2, 4))
        self._calibrator = Calibrator(camera)

    def get_center_of_lane(self, frame) -> Optional[np.ndarray]:
        """
        Returns the center of the lane as a numpy array,
        representing the point in the real world where the center of the lane is.
        :param frame:
        :return: The center of the lane as a numpy array. Can be None if no center of lane was detected.
        """
        instances_map = self._detector(frame)
        if instances_map is None:
            return None

        center = self._detector.get_center_of_lane(instances_map)
        if center is None:
            return None

        return self._calibrator.to_real_world(center)

    def apply_center_segmentation(self, frame) -> Optional[cv2.typing.MatLike]:
        """
        Returns the frame with the center of the lane segmentation applied.
        :param frame:
        :return: The frame with the center of the lane segmentation applied. Can be None if no center of lane was detected.
        """
        instances_map = self._detector(frame)
        if instances_map is None:
            return None

        frame = cv2.resize(frame, LaneDetector.DEFAULT_IMAGE_SIZE)
        center = self._detector.get_center_of_lane(instances_map, get_map=True)
        if center is None:
            return None

        mask = self._make_mask(center)
        frame = self._apply_mask(frame, mask)

        frame = self._calibrator.bird_eye_view(frame)

        return frame

    def apply_lanes_segmentation(self, frame) -> Optional[cv2.typing.MatLike]:
        """
        Returns the frame with the lanes segmentation applied.
        :param frame:
        :return: The frame with the lanes segmentation applied. Can be None if no lanes were detected.
        """
        instances_map = self._detector(frame)
        if instances_map is None:
            return None

        instances_map = instances_map.cpu().numpy()

        frame = cv2.resize(frame, LaneDetector.DEFAULT_IMAGE_SIZE)
        mask = self._make_mask(instances_map)
        frame = self._apply_mask(frame, mask)

        frame = self._calibrator.bird_eye_view(frame)

        return frame

    def _extract_lanes(self, frame) -> Optional[torch.Tensor]:
        instances_map = self._detector(frame)

        return instances_map

    def _make_mask(self, instances_map):
        mask = instances_map.astype(np.uint8)
        mask = np.expand_dims(mask, axis=2)  # Add a new axis
        mask = np.repeat(mask, 3, axis=2)
        non_zero_indices = np.where(np.any(mask != [0, 0, 0], axis=-1))

        return non_zero_indices

    def _apply_mask(self, frame, mask, color=[0, 0, 255]):
        for i in range(len(mask[0])):
            x, y = mask[0][i], mask[1][i]
            frame[x, y] = color

        return frame
