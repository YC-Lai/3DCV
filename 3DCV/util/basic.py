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
        self.R = R
        self.T = T
        self.scale = scale


class Frame(Pose):
    def __init__(self, R, T, scale, keypoints, descriptors, key_inlier=None, des_inlier=None):
        """
        Parameters:
            R: rotation matrix in WCS
            T: translation matrix in WCS
            scale: scale value
            key: keypoints
            des: descriptors
            key_inlier: the inlier of previous frame to current
            des_inlier: the inlier of previous frame to current
        """
        super().__init__(R, T, scale)
        # for compute pose
        self.key = keypoints
        self.des = descriptors
        # for triangulation
        self.key_inlier = key_inlier
        self.des_inlier = des_inlier
