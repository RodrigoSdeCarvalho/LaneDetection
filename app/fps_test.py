from essentials.add_module import set_working_directory
set_working_directory()

import numpy as np
from neural_networks.models.lanenet.enet import ENet
from data.loaders.video_loader import VideoLoader
from neural_networks.decorators.lane_detector import LaneDetector
import cv2
from utils.path import Path
import time
from utils.logger import Logger


def extract_lanes(frame, detector):
    Logger.trace("Processing frame", show=True)
    instances_map = detector(frame)

    return instances_map


def process_frames(video_path):
    detector = LaneDetector(ENet(2, 4))
    loader = VideoLoader(video_path)

    time_per_frame_list = []

    start = time.time()
    for frame in loader:
        frame_start = time.time()
        instances_map = extract_lanes(frame, detector)
        if instances_map is None:
            continue

        frame_process_time = time.time() - frame_start
        time_per_frame_list.append(frame_process_time)
    end_time = time.time() - start


    Logger.info(f"Average time per frame: {np.mean(time_per_frame_list)}", show=True)
    Logger.info(f"Max time per frame: {np.max(time_per_frame_list)}", show=True)
    Logger.info(f"Min time per frame: {np.min(time_per_frame_list)}", show=True)
    Logger.info(f"Total frames: {len(time_per_frame_list)}", show=True)
    Logger.info(f"FPS: {len(time_per_frame_list) / end_time}", show=True)
    Logger.info(f"Total time: {end_time}", show=True)


def main():
    process_frames("lanedet.mp4")
    Logger.info("Done", show=True)


if __name__ == '__main__':
    main()
