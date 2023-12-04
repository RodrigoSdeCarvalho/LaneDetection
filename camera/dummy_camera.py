from camera.camera import Camera, Intrinsics, Extrinsics, ExtrinsicPoint, T


class DummyCamera(Camera):
    def __init__(self):
        scale_factor = 1
        intrinsics = Intrinsics(1, 2, 3, 6)
        extrinsics = Extrinsics(ExtrinsicPoint(1, 3, 1),
                                ExtrinsicPoint(6, 1, 1),
                                ExtrinsicPoint(6, 1, 5))
        t = T(1, 1, 1)
        super().__init__(scale_factor, intrinsics, extrinsics, t)
