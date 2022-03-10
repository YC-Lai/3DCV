from typing import List
import numpy as np
import cv2 as cv
from util.basic import (Camera, Frame)
import open3d as o3d
import multiprocessing as mp


class visualOdometry:
    def __init__(self, intrinsic, distCoeffs, img_paths) -> None:

        self.camera = Camera(intrinsic, distCoeffs)

        self.feature_extractor = cv.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # image paths
        self.img_paths = img_paths

        # frame
        self.frame_til_now: Frame = None

        # initialize
        self.__initialize()

    def __initialize(self):
        img0 = cv.imread(self.img_paths[0])
        img0_kp, img0_des = self.feature_extractor.detectAndCompute(img0, None)
        frame0 = Frame(np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64), img0_kp, img0_des)

        img1 = cv.imread(self.img_paths[1])
        img1_kp, img1_des = self.feature_extractor.detectAndCompute(img1, None)
        R, T, matches = self.__get_pose_between_frames(img1_kp, img1_des, img0_kp, img0_des)
        frame1 = Frame(R, -1*T, img1_kp, img1_des, matches=matches, Previous=frame0)

        self.frame_til_now = frame1

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()

        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    # TODO:
                    # insert new camera pose here using vis.add_geometry()
                    line_set = self.get_lineSet(R, t)
                    vis.add_geometry(line_set)
            except:
                pass

            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def process_frames(self, thread_queue=None):
        for i, img_path in enumerate(self.img_paths[2:]):
            cur_img = cv.imread(img_path)
            # find the keypoints and descriptors with ORB
            cur_kp, cur_des = self.feature_extractor.detectAndCompute(cur_img, None)

            # get the relative pose
            relative_R, relative_T, matches = self.__get_pose_between_frames(
                cur_kp, cur_des, self.frame_til_now.keypoints, self.frame_til_now.descriptors)

            # R, T in WCS
            R = relative_R @ self.frame_til_now.R
            T = relative_R @ self.frame_til_now.T - relative_T
            cur_frame = Frame(R, T, cur_kp, cur_des, matches=matches, Previous=self.frame_til_now)

            # triangulation
            pre_points, cur_points, post_points = self.__find_2_points_cross_three_frames(self.frame_til_now, cur_frame)
            pre_projMat = self.camera.intrinsic @ np.hstack((np.eye(3,
                                                            dtype=np.float64), np.zeros((3, 1), dtype=np.float64)))
            cur_projMat = self.camera.intrinsic @ np.hstack((self.frame_til_now.R, self.frame_til_now.T))
            post_projMat = self.camera.intrinsic @ np.hstack((relative_R, relative_T))

            points4d_1 = cv.triangulatePoints(pre_projMat, cur_projMat, pre_points, cur_points)
            
            points3d_1 = (points4d_1[:3, :] / points4d_1[3, :].reshape(1, -1))
            points4d_2 = cv.triangulatePoints(pre_projMat, post_projMat, cur_points, post_points)
            points3d_2 = (points4d_2[:3, :]/points4d_2[3, :].reshape(1, -1))

            distance_1 = np.linalg.norm(points3d_1[:, 0] - points3d_1[:, 1])
            distance_2 = np.linalg.norm(points3d_2[:, 0] - points3d_2[:, 1])

            scale = self.frame_til_now.scale * (distance_2 / distance_1)

            print("{}/{}: scale={}".format(i+3, len(self.img_paths), scale))

            cur_frame.scale = scale
            cur_frame.T = relative_R @ self.frame_til_now.T - scale * relative_T
            self.frame_til_now = cur_frame

            if thread_queue is not None:
                thread_queue.put((R, T))

            img_show = cv.drawKeypoints(cur_img, cur_kp, None, color=(0, 255, 0))
            cv.imshow('frame', img_show)
            if cv.waitKey(30) == 27:
                break

    def __get_pose_between_frames(self, cur_kp, cur_des, last_kp, last_des):
        # find descriptors.
        matches = self.matcher.match(cur_des, last_des)

        cur_points = np.empty((0, 2))
        last_points = np.empty((0, 2))
        for matche in matches:
            cur_idx = matche.queryIdx
            last_idx = matche.trainIdx
            cur_points = np.vstack((cur_points, cur_kp[cur_idx].pt))
            last_points = np.vstack((last_points, last_kp[last_idx].pt))

        # Normalize for Esential Matrix calaculation
        cur_points = cv.undistortPoints(cur_points, self.camera.intrinsic,
                                        self.camera.distCoeffs, None, self.camera.intrinsic)
        last_points = cv.undistortPoints(last_points, self.camera.intrinsic,
                                         self.camera.distCoeffs, None, self.camera.intrinsic)

        # find essential matrix and decompose into R, t
        fx, fy, cx, cy = self.camera.parameter
        E, _ = cv.findEssentialMat(cur_points, last_points, focal=fx, pp=(
            cx, cy), method=cv.RANSAC, prob=0.999, threshold=1.0, mask=None)
        _, R, T, _ = cv.recoverPose(E, cur_points, last_points, focal=fx,
                                    pp=(cx, cy), mask=None)

        return R, T, matches

    def __find_2_points_cross_three_frames(self, last_frame: Frame, cur_frame: Frame):
        """
        Notion:
            cur_{name}: frame k
            pre_{name}: frame k-1
            post_{name}: frame k+1
            {name}1: instances between k-1 and k
            {name}2: instances between k and k+1
        """
        # get required value
        cur_kp = last_frame.keypoints
        pre_kp = last_frame.Previous.keypoints
        post_kp = cur_frame.keypoints
        matches1 = last_frame.matches
        matches2 = cur_frame.matches

        matches2 = sorted(matches2, key=lambda x: x.distance)
        count = 0
        cur_points = []
        pre_points = []
        post_points = []
        for match2 in matches2:
            for match1 in matches1:
                if count == 2:
                    break
                if match2.trainIdx == match1.queryIdx:
                    cur_point = cur_kp[match2.trainIdx].pt
                    pre_point = pre_kp[match1.trainIdx].pt
                    post_point = post_kp[match2.queryIdx].pt

                    cur_points.append(cur_point)
                    pre_points.append(pre_point)
                    post_points.append(post_point)
                    count += 1
        cur_points = np.array(cur_points)
        pre_points = np.array(pre_points)
        post_points = np.array(post_points)

        return pre_points.T, cur_points.T, post_points.T

    def get_lineSet(self, R, t):
        # get four corner points and camera center
        points = np.array(
            [[0, 0, 1], [360, 0, 1], [360, 640, 1], [0, 640, 1], [320, 180, 0]])
        points = np.linalg.pinv(self.camera.intrinsic) @ points.T
        inv_R = np.linalg.pinv(R)
        points = inv_R @ (points - t)
        points = points.T

        # generate color
        color = [1, 0, 0]
        colors = np.tile(color, (8, 1))

        # set up line
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(
                [[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]])
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set
