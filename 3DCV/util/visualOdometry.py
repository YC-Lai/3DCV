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
        R, T, key_inlier, des_inlier = self.__get_pose_btw_frames(img0_kp, img0_des, img1_kp, img1_des)
        T = -1*T
        frame1 = Frame(R, T, 1, img1_kp, img1_des, key_inlier, des_inlier)

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
                    ctr = vis.get_view_control()
                    ctr.change_field_of_view(step=-90.0)
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
            relative_R, relative_T, key_inlier, des_inlier = self.__get_pose_btw_frames(
                self.cur_frame.key, self.cur_frame.des, post_kp, post_des)
            relative_T = -1*relative_T
            post_frame = Frame(relative_R, relative_T, 1, post_kp, post_des, key_inlier, des_inlier)

            # triangulation
            scale = self.__compute_relative_scale(50, self.pre_frame, self.cur_frame, post_frame)
            if np.isnan(scale):
                scale = 1
            if scale > 2:
                scale = 2
            post_frame.scale = scale
            print("{}/{}: scale={}".format(i+3, len(self.img_paths), post_frame.scale))

            # R, T in WCS
            R = self.pose_til_now.R @ relative_R
            T = self.pose_til_now.T + post_frame.scale * self.pose_til_now.R @ relative_T

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

    def __get_pose_btw_frames(self, cur_kp, cur_des, post_kp, post_des):
        # find descriptors.
        matches = self.matcher.match(cur_des, post_des)

        cur_points = np.empty((0, 2))
        post_points = np.empty((0, 2))
        post_des_temp = np.empty((0, post_des.shape[1]))
        for matche in matches:
            cur_idx = matche.queryIdx
            post_idx = matche.trainIdx
            cur_points = np.vstack((cur_points, cur_kp[cur_idx].pt))
            post_points = np.vstack((post_points, post_kp[post_idx].pt))
            post_des_temp = np.vstack((post_des_temp, post_des[post_idx]))

        # Normalize for Esential Matrix calaculation
        cur_points = cv.undistortPoints(cur_points, self.camera.intrinsic,
                                        self.camera.distCoeffs, None, self.camera.intrinsic)
        post_points = cv.undistortPoints(post_points, self.camera.intrinsic,
                                         self.camera.distCoeffs, None, self.camera.intrinsic)

        # find essential matrix and decompose into R, t (post frame coordinate system to current)
        # fx, fy, cx, cy = self.camera.parameter
        E, _ = cv.findEssentialMat(cur_points, post_points, self.camera.intrinsic)
        retval, R, T, inliner = cv.recoverPose(E, cur_points, post_points, self.camera.intrinsic)

        inliner_idx = np.squeeze(np.argwhere(np.squeeze(inliner)))
        
        key_inlier = post_points[inliner_idx, :]
        des_inlier = post_des_temp[inliner_idx, :].astype("uint8")
        
        return R, T, key_inlier, des_inlier

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
        T3 = self.camera.intrinsic @ np.hstack(((cur_frame.R @ post_frame.R),
                                               (cur_frame.R @ post_frame.T + cur_frame.T)))

        # find match
        matches_1 = self.matcher.match(cur_frame.des_inlier, pre_frame.des)
        matches_2 = self.matcher.match(cur_frame.des_inlier, post_frame.des_inlier)
        queryIdx_1 = [m.queryIdx for m in matches_1]
        queryIdx_2 = [m.queryIdx for m in matches_2]
        cur_idx, pre_ind, post_ind = np.intersect1d(queryIdx_1, queryIdx_2, return_indices=True)
        cur_points = np.empty((0, 2))
        pre_points = np.empty((0, 2))
        post_points = np.empty((0, 2))
        for i in range(len(cur_idx)):
            cur_points = np.vstack((cur_points, cur_frame.key_inlier[cur_idx[i]]))
            pre_points = np.vstack((pre_points, pre_frame.key[matches_1[pre_ind[i]].trainIdx].pt))
            post_points = np.vstack((post_points, post_frame.key_inlier[matches_2[post_ind[i]].trainIdx]))

        cur_points = cur_points.T
        pre_points = pre_points.T
        post_points = post_points.T

        # Triangulation
        N = cur_points.shape[1]
        scales = []
        for _ in range(n_pairs):
            idx = np.random.randint(N, size=2)
            pre_p = pre_points[:, idx]
            cur_p = cur_points[:, idx]
            post_p = post_points[:, idx]

            points4d_1 = cv.triangulatePoints(T1, T2, pre_p, cur_p)
            points4d_2 = cv.triangulatePoints(T2, T3, cur_p, post_p)

            points3d_1 = (points4d_1[:3, :] / points4d_1[3, :].reshape(1, -1))
            points3d_2 = (points4d_2[:3, :] / points4d_2[3, :].reshape(1, -1))

            distance_1 = np.linalg.norm(points3d_1[:, 0] - points3d_1[:, 1])
            distance_2 = np.linalg.norm(points3d_2[:, 0] - points3d_2[:, 1])
            scales.append((distance_2 / distance_1))
        
        return statistics.median(scales)

    def get_lineSet(self, R, t):
        # get four corner points and camera center
        points = np.array(
            [[0, 0, 1], [360, 0, 1], [360, 640, 1], [0, 640, 1]])
        points = np.linalg.pinv(self.camera.intrinsic) @ points.T
        points /= points[-1, 0]
        points3D = t + R @ (points)
        points3D = points3D.T
        points3D = np.concatenate((points3D, t.T), axis=0)

        # generate color
        color = [1, 0, 0]
        colors = np.tile(color, (8, 1))

        # set up line
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points3D),
            lines=o3d.utility.Vector2iVector(
                [[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]])
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set
