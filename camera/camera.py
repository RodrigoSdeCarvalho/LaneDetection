import numpy as np


class Intrinsics:
    def __init__(self, fx, fy,
                       cx, cy):
        self.matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])

    def __repr__(self):
        return f"Intrinsics(fx={self.matrix[0][0]}, fy={self.matrix[1][1]}, cx={self.matrix[0][2]}, cy={self.matrix[1][2]})"

    def __str__(self):
        return self.__repr__()


class ExtrinsicPoint:
    def __init__(self, r0, r1, r2):
        self.matrix = np.array([r0, r1, r2])

    def __repr__(self):
        return f"ExtrinsicPoint(r0={self.matrix[0]}, r1={self.matrix[1]}, r2={self.matrix[2]}, t={self.matrix[3]})"

    def __str__(self):
        return self.__repr__()


class Extrinsics:
    def __init__(self, *points: ExtrinsicPoint):
        points = list(points)

        if len(points) != 3:
            raise ValueError("Extrinsics must have 3 extrinsic points")

        self.matrix = np.array([points[0].matrix,
                                points[1].matrix,
                                points[2].matrix])

    def __repr__(self):
        return f"Extrinsics(p0={self.matrix[0]}, p1={self.matrix[1]}, p2={self.matrix[2]})"


class T:
    def __init__(self, t0, t1, t2):
        self.matrix = np.array([t0, t1, t2])

    def __repr__(self):
        return f"T(x={self.matrix[0]}, y={self.matrix[1]}, z={self.matrix[2]})"

    def __str__(self):
        return self.__repr__()


class Camera:
    def __init__(self,
                 scale_factor: float,
                 intrinsics: Intrinsics,
                 extrinsics: Extrinsics,
                 t: T):
        if scale_factor == 0:
            raise ValueError("Scale factor must be different than 0")

        self.scale_factor = scale_factor
        self._intrinsics = intrinsics
        self._extrinsics = extrinsics
        self._t = t

    @property
    def extrinsics(self) -> Extrinsics:
        return self._extrinsics.matrix

    @property
    def intrinsics(self) -> Intrinsics:
        return self._intrinsics.matrix

    @property
    def t(self) -> T:
        return self._t.matrix

    def __repr__(self):
        return f"Camera(s={self.scale_factor}, in={self.intrinsics}, ex={self.extrinsics}), t={self.t})"

    def __str__(self):
        return self.__repr__()
