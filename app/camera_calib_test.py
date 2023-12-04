from essentials.add_module import set_working_directory
set_working_directory()

from camera.dummy_camera import DummyCamera
from camera.calibrator import Calibrator
from utils.logger import Logger
import numpy as np


if __name__ == "__main__":
    camera = DummyCamera()
    Logger.trace(f"Camera: {camera}", show=True)
    pixel_point = np.array([100, 150])
    calibrator = Calibrator(camera)
    Logger.trace(f"Pixel point: {pixel_point}", show=True)
    real_world_point = calibrator.to_real_world(pixel_point)
    Logger.trace(f"Real world point: {real_world_point}", show=True)
    pixel_point = calibrator.to_pixel(real_world_point)
    Logger.trace(f"Pixel point: {pixel_point}", show=True)