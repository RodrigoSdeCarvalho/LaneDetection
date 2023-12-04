import numpy as np
import cv2
from utils.config import ImageConfig
from camera.camera import Camera
from utils.logger import Logger


class Calibrator:
    def __init__(self, camera: Camera, width=ImageConfig().width, height=ImageConfig().height):
        self.width = width
        self.height = height
        self.camera = camera

    def to_real_world(self, pixel: np.ndarray) -> np.ndarray:
        """
        :param camera:
        :param pixel: pixel coordinate in the image
        :return: real world coordinate
        """

        # S -> scale_factor
        # I -> 3x3 intrinsic matrix
        # E -> 3x4 extrinsic matrix
        # P -> 3x1 image coordinates
        # T -> 3x1 vector
        # RWC -> 4x1 real world coordinates
        S = self.camera.scale_factor
        I = self.camera.intrinsics
        E = self.camera.extrinsics
        P = np.array([pixel[1], pixel[0], 1]) # in this context, [1] is x-axis and [0] is y-axis
        T = self.camera.t

        inv_I = np.linalg.inv(I)
        inv_E = np.linalg.inv(E)

        SP = S * P
        SP_inv_I = SP @ inv_I
        SP_inv_I_T = SP_inv_I - T
        RWC = SP_inv_I_T @ inv_E

        return RWC

    def to_pixel(self, real_world: np.ndarray) -> np.ndarray:
        """
        :param camera:
        :param real_world: real world coordinate
        :return: pixel coordinate in the image
        """

        # S -> scale_factor
        # I -> 3x3 intrinsic matrix
        # E -> 3x4 extrinsic matrix
        # P -> 3x1 image coordinates
        # T -> 3x1 vector
        # RWC -> 4x1 real world coordinates
        S = self.camera.scale_factor
        I = self.camera.intrinsics
        E = self.camera.extrinsics
        T = self.camera.t

        rwc_E = real_world @ E
        rwc_E_T = rwc_E + T
        rwc_E_T_I = rwc_E_T @ I
        P = (1 / S) * rwc_E_T_I

        fixed_P_to_index_for_cv2 = np.array([P[1], P[0]])

        return fixed_P_to_index_for_cv2

    def bird_eye_view(self, frame,
                      max_y=ImageConfig().bev['max_y'],
                      max_x=ImageConfig().bev['max_x'],
                      min_y=ImageConfig().bev['min_y'],
                      min_x=ImageConfig().bev['min_x']):
        tl = [min_x, min_y]
        tr = [max_x, min_y]
        br = [max_x, max_y]
        bl = [min_x, max_y]

        corner_points_array = np.float32([tl, tr, br, bl])

        # original image dimensions
        width = self.width
        height = self.height

        # Create an array with the parameters (the dimensions) required to build the matrix
        imgTl = [0, 0]
        imgTr = [width, 0]
        imgBr = [width, height]
        imgBl = [0, height]
        img_params = np.float32([imgTl, imgTr, imgBr, imgBl])

        # Compute and return the transformation matrix
        matrix = cv2.getPerspectiveTransform(corner_points_array,img_params)
        img_transformed = cv2.warpPerspective(frame, matrix, (width, height))

        return img_transformed
