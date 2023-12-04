from cv2.cuda import GpuMat
from neural_networks.decorators.lane_detector.lane import Lanes
import numpy as np
from multipledispatch import dispatch
from typing import Optional
from utils.config import DecoratorConfig
from utils.logger import Logger
from torch import Tensor


class CachedInstancesMap:
    def __init__(self, frames_to_remember: int):
        self._instances_map = None
        self._frames_to_remember = frames_to_remember
        self._frames_count = 0

    def update(self, instances_map: GpuMat):
        self._instances_map = instances_map
        self._frames_count = self._frames_to_remember

    def get(self) -> Optional[Lanes]:
        if not self.is_empty:
            if self._frames_count > 0:
                self._frames_count -= 1

                return self._instances_map
        self._instances_map = None
        Logger.info("Instances map is empty", show=True)

        return None

    @property
    def is_empty(self):
        return self._instances_map is None


class CachedLanes:
    def __init__(self, frames_to_remember: int):
        self._lanes = None
        self._frames_to_remember = frames_to_remember
        self._frames_count = 0

    def update(self, lanes: Lanes):
        self._lanes = lanes
        self._frames_count = self._frames_to_remember

    def get(self) -> Optional[Lanes]:
        if not self.is_empty:
            if self._frames_count > 0:
                self._frames_count -= 1

                return self._lanes
        self._lanes = None
        Logger.info("Lanes are empty", show=True)

        return None

    @property
    def is_empty(self):
        return self._lanes is None


class CachedCenterPoint:
    def __init__(self, frames_to_remember: int):
        self._center_point = None
        self._frames_to_remember = frames_to_remember
        self._frames_count = 0

    def update(self, center_point: np.ndarray):
        self._center_point = center_point
        self._frames_count = self._frames_to_remember

    def get(self) -> Optional[np.ndarray]:
        if not self.is_empty:
            if self._frames_count > 0:
                self._frames_count -= 1

                return self._center_point
        self._center_point = None
        Logger.info("Center point is empty", show=True)

        return None

    @property
    def is_empty(self):
        return self._center_point is None


class Memory:
    FRAMES_TO_REMEMBER = DecoratorConfig().frames_to_remember

    def __init__(self):
        self._instances_map = CachedInstancesMap(Memory.FRAMES_TO_REMEMBER)
        self._lanes = CachedLanes(Memory.FRAMES_TO_REMEMBER)
        self._center = CachedCenterPoint(Memory.FRAMES_TO_REMEMBER)

    @dispatch(Lanes)
    def update(self, lanes: Lanes):
        Logger.info("Updating lanes", show=True)
        self._lanes.update(lanes)

    @dispatch(np.ndarray)
    def update(self, center_point: np.ndarray):
        Logger.info("Updating center point", show=True)
        self._center.update(center_point)

    @dispatch(GpuMat)
    def update(self, instances_map: GpuMat):
        Logger.info("Updating instances map", show=True)
        self._instances_map.update(instances_map)

    @dispatch(Tensor)
    def update(self, instances_map: GpuMat):
        Logger.info("Updating instances map", show=True)
        self._instances_map.update(instances_map)

    def get_lanes(self) -> Optional[Lanes]:
        Logger.info("Getting lanes", show=True)
        return self._lanes.get()

    def get_center(self) -> Optional[np.ndarray]:
        Logger.info("Getting center point", show=True)
        return self._center.get()

    def get_instances_map(self) -> Optional[GpuMat]:
        Logger.info("Getting instances map", show=True)
        return self._instances_map.get()
