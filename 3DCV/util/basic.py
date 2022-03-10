import numpy as np
import cv2 as cv


class Camera:
    def __init__(self, intrinsic, distCoeffs):
        self.__intrinsic = intrinsic
        self.__distCoeffs = distCoeffs

    @property
    def parameter(self):
        """
        Return fx, fy, cx, cy
        fx, fy: focal length of the camera in x and y direction in pixels
        cx, cy: principal point (the point that all rays converge) coordinates in pixels
        """
        return self.__intrinsic[0, 0], self.__intrinsic[1, 1], self.__intrinsic[0, 2], self.__intrinsic[1, 2]

    @property
    def intrinsic(self):
        return self.__intrinsic

    @property
    def distCoeffs(self):
        return self.__distCoeffs


class Pose:
    def __init__(self, R, T, scale) -> None:
        self.__R = R
        self.__T = T
        self.__scale = scale

    @property
    def R(self):
        return self.__R

    @property
    def T(self):
        return self.__T

    @T.setter
    def T(self, value):
        self.__T = value

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        self.__scale = value


class Frame(Pose):
    def __init__(self, R, T, keypoints, descriptors, scale=1, matches=None, Previous=None):
        """
        Parameters:
            R: rotation matrix in WCS
            T: translation matrix in WCS
            scale: scale value
        """
        super().__init__(R, T, scale)

        self.__keypoints = keypoints
        self.__descriptors = descriptors
        self.__matches = matches
        # store previous information
        self.Previous: Frame = Previous

    @property
    def keypoints(self):
        return self.__keypoints

    @property
    def descriptors(self):
        return self.__descriptors

    @property
    def matches(self):
        return self.__matches

    @matches.setter
    def matches(self, value):
        self.__matches = value
