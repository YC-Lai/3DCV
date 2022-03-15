from typing import List
import numpy as np
import cv2 as cv
from util.basic import (Camera, Frame, Pose)
import open3d as o3d
import multiprocessing as mp
import statistics


class visualOdometry:
    def __init__(self, intrinsic, distCoeffs, img_paths) -> None:

        self.camera = Camera(intrinsic, distCoeffs)

        self.feature_extractor = cv.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # image paths
        self.img_paths = img_paths

        # frame
        self.pre_frame: Frame = None
        self.cur_frame: Frame = None

        # pose in WCS
        self.pose_til_now: Pose = None

        # initialize
        self.__initialize()

    def __initialize(self):
        img0 = cv.imread(self.img_paths[0])
        img0_kp, img0_des = self.feature_extractor.detectAndCompute(img0, None)
        frame0 = Frame(np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64), 1, img0_kp, img0_des)

        img1 = cv.imread(self.img_paths[1])
        img1_kp, img1_des = self.feature_extractor.detectAndCompute(img1, None)
        R, T, matches, post_matches = self.__get_pose_btw_frames(img1_kp, img1_des, img0_kp, img0_des)
        frame1 = Frame(R, T, 1, img1_kp, img1_des, pre_matches=matches)
        frame0.post_matches = post_matches

        self.pre_frame = frame0
        self.cur_frame = frame1
        self.pose_til_now = Pose(R, T, 1)

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
            post_img = cv.imread(img_path)
            # find the keypoints and descriptors with ORB
            post_kp, post_des = self.feature_extractor.detectAndCompute(post_img, None)

            # get the relative pose
            relative_R, relative_T, matches, post_matches = self.__get_pose_btw_frames(
                post_kp, post_des, self.cur_frame.keypoints, self.cur_frame.descriptors)
            post_frame = Frame(relative_R, relative_T, 1, post_kp, post_des, pre_matches=matches)
            self.cur_frame.post_matches = post_matches

            # triangulation
            scale = self.__compute_relative_scale(100, self.pre_frame, self.cur_frame, post_frame)
            if np.isnan(scale):
                scale = 1
            post_frame.scale = self.cur_frame.scale * scale
            if post_frame.scale < 0.1 or post_frame.scale > 5:
                post_frame.scale = 1
            print("{}/{}: scale={}".format(i+3, len(self.img_paths), post_frame.scale))

            # R, T in WCS
            R = relative_R @ self.pose_til_now.R
            # T = self.pose_til_now.T + post_frame.scale * self.pose_til_now.R @ relative_T
            T = relative_R @ self.pose_til_now.T - post_frame.scale * relative_T

            # update
            self.pose_til_now.R = R
            self.pose_til_now.T = T
            self.pre_frame = self.cur_frame
            self.cur_frame = post_frame

            if thread_queue is not None:
                thread_queue.put((R, T))

            # show image
            img_show = cv.drawKeypoints(post_img, post_kp, None, color=(0, 255, 0))
            cv.imshow('frame', img_show)
            if cv.waitKey(30) == 27:
                break

    def __get_pose_btw_frames(self, post_kp, post_des, cur_kp, cur_des):
        # find descriptors.
        matches = self.matcher.match(post_des, cur_des)
        post_matches = self.matcher.match(cur_des, post_des)

        cur_points = np.empty((0, 2))
        last_points = np.empty((0, 2))
        for matche in matches:
            cur_idx = matche.queryIdx
            last_idx = matche.trainIdx
            cur_points = np.vstack((cur_points, post_kp[cur_idx].pt))
            last_points = np.vstack((last_points, cur_kp[last_idx].pt))

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

        return R, T, matches, post_matches

    def __triple_matches(self, pre_frame: Frame, cur_frame: Frame, post_frame: Frame):
        """
        Parameters:
            n_pairs: number of pairs
            pre_frame: frame k-1
            cur_frame: frame k
            post_frame: frame k+1
        """
        pre_matches = cur_frame.pre_matches
        post_matches = cur_frame.post_matches

        pre_matches = sorted(pre_matches, key=lambda x: x.queryIdx)
        post_matches = sorted(post_matches, key=lambda x: x.queryIdx)

        cur_points = np.empty((0, 2))
        pre_points = np.empty((0, 2))
        post_points = np.empty((0, 2))
        for pre_m, post_m in zip(pre_matches, post_matches):
            cur_points = np.vstack((cur_points, cur_frame.keypoints[pre_m.queryIdx].pt))
            pre_points = np.vstack((pre_points, pre_frame.keypoints[pre_m.trainIdx].pt))
            post_points = np.vstack((post_points, post_frame.keypoints[post_m.trainIdx].pt))

        return pre_points.T, cur_points.T, post_points.T

    def __compute_relative_scale(self, n_pairs, pre_frame: Frame, cur_frame: Frame, post_frame: Frame):
        """
        Triangulation.
        Parameters:
            n_pairs: number of pairs
            pre_frame: frame k-1
            cur_frame: frame k
            post_frame: frame k+1
        """
        T1 = self.camera.intrinsic @ np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0],
                                               [0, 0, 1, 0]], dtype=np.float32)
        T2 = self.camera.intrinsic @ np.hstack((cur_frame.R, cur_frame.T))
        T3 = self.camera.intrinsic @ np.hstack(((post_frame.R @ cur_frame.R), (cur_frame.T + cur_frame.R @ post_frame.T)))

        # shape: [2, N]
        pre_points, cur_points, post_points = self.__triple_matches(pre_frame, cur_frame, post_frame)

        N = cur_points.shape[1]
        scales = []
        for _ in range(n_pairs):
            idx = np.random.randint(N, size=2)
            pre_p = pre_points[:, idx]
            cur_p = cur_points[:, idx]
            post_p = post_points[:, idx]

            points4d_1 = cv.triangulatePoints(T1, T2, pre_p, cur_p)
            points4d_2 = cv.triangulatePoints(T1, T3, pre_p, post_p)

            points3d_1 = (points4d_1[:3, :] / points4d_1[3, :].reshape(1, -1))
            points3d_2 = (points4d_2[:3, :]/points4d_2[3, :].reshape(1, -1))

            distance_1 = np.linalg.norm(points3d_1[:, 0] - points3d_1[:, 1])
            distance_2 = np.linalg.norm(points3d_2[:, 0] - points3d_2[:, 1])
            scales.append((distance_2 / distance_1))
        return statistics.median(scales)

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
