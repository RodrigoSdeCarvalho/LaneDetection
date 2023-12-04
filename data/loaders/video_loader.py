from utils.path import Path
from data.loaders.loader import Loader
import cv2
from typing import Callable, Optional


class VideoLoader(Loader):
    def __init__(self, video_name: str, pre_process: Callable = None):
        super().__init__(config=None)
        self._video_name = video_name
        self._video = cv2.VideoCapture(self.src_path)

        if pre_process:
            self._pre_process_ptr = pre_process
        else:
            self._pre_process_ptr = None

        if not self._video.isOpened():
            raise Exception("Video not found at assets/data/video/<video_name>")

    @property
    def src_path(self):
        return Path().get_video(self._video_name)

    def get(self) -> Optional[cv2.typing.MatLike]:
        ret, frame = self._video.read()

        if not ret:
            return None

        if self._pre_process_ptr:
            frame = self._pre_process_ptr(frame)

        return frame

    def __iter__(self):
        return self

    def __next__(self):
        frame = self.get()

        if frame is None:
            raise StopIteration

        return frame

    def __del__(self):
        self._video.release()
