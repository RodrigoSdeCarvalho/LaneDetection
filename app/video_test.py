from essentials.add_module import set_working_directory
set_working_directory()

import numpy as np
from neural_networks.models.lanenet.enet import ENet
from data.loaders.video_loader import VideoLoader
from neural_networks.decorators.lane_detector import LaneDetector
import cv2
from utils.path import Path
from camera.calibrator import Calibrator
from utils.logger import Logger


def extract_lanes(frame, detector):
    Logger.trace("Processing frame", show=True)
    instances_map = detector(frame)

    return instances_map


def make_mask(instances_map):
    mask = instances_map.astype(np.uint8)
    mask = np.expand_dims(mask, axis=2)  # Add a new axis
    mask = np.repeat(mask, 3, axis=2)
    non_zero_indices = np.where(np.any(mask != [0, 0, 0], axis=-1))

    return non_zero_indices


def apply_mask(frame, mask, color=[0, 0, 255]):
    Logger.trace("Applying mask", show=True)
    for i in range(len(mask[0])):
        x, y = mask[0][i], mask[1][i]
        frame[x, y] = color

    return frame


def get_frames(video_path):
    detector = LaneDetector(ENet(2, 4))
    loader = VideoLoader(video_path)

    frames = []
    for frame in loader:
        instances_map = extract_lanes(frame, detector)
        if instances_map is None:
            continue
        instances_map = instances_map.cpu().numpy()

        frame = cv2.resize(frame, LaneDetector.DEFAULT_IMAGE_SIZE)
        mask = make_mask(instances_map)
        frame = apply_mask(frame, mask)

        frame = Calibrator(None).bird_eye_view(frame)
        cv2.imshow("Frame", frame)
        cv2.waitKey(5)

        frames.append(frame)

    return frames


def main():
    output_path = Path().get_output("lane_det_test.avi")

    codec = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30.0
    frame_height, frame_width = LaneDetector.DEFAULT_IMAGE_SIZE

    output = cv2.VideoWriter(output_path, codec, fps, (frame_width, frame_height))

    frames = get_frames("lanedet.mp4")
    for frame in frames:
        Logger.trace("Writing frame", show=True)
        output.write(frame)
    Logger.info("Done", show=True)

    output.release()


if __name__ == '__main__':
    main()
