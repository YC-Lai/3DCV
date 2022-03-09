import numpy as np
import pandas as pd
from util.helper import (P3P, ransac, PCS_to_CCS, trilateration, image_undistortion)
from util.display import (load_point_cloud, create_camera_position)
import os
import cv2 as cv
from scipy.spatial.transform import Rotation
import open3d as o3d


class Matching():
    def __init__(self, images_df, train_df, point3D_df, point_desc_df):
        self.images_df = images_df
        self.point_desc_df = point_desc_df

        # Process model descriptors
        desc_df = self.average_desc(train_df, point3D_df)
        self.kp_model = np.array(desc_df["XYZ"].to_list())
        self.desc_model = np.array(
            desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

        # camera property
        self.cameraMatrix = np.array(
            [[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])
        self.distCoeffs = np.array(
            [0.0847023, -0.192929, -0.000201144, -0.000725352])

    def query(self, idx):
        # Load query image
        fname = ((self.images_df.loc[self.images_df["IMAGE_ID"] == idx])[
                 "NAME"].values)[0]
        rimg = cv.imread("data/task3/frames/"+fname, cv.IMREAD_GRAYSCALE)

        # Load query keypoints and descriptors
        points = self.point_desc_df.loc[self.point_desc_df["IMAGE_ID"] == idx]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(
            points["DESCRIPTORS"].to_list()).astype(np.float32)

        return kp_query, desc_query

    def average_desc(self, train_df, point3D_df):
        train_df = train_df[["POINT_ID", "XYZ", "RGB", "DESCRIPTORS"]]
        desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
        desc = desc.apply(lambda x: list(np.mean(x, axis=0)))
        desc = desc.reset_index()
        desc = desc.join(point3D_df.set_index("POINT_ID"), on="POINT_ID")
        return desc

    def find_match(self, idx):
        kp_query, desc_query = self.query(idx)

        bf = cv.BFMatcher()
        matches = bf.knnMatch(desc_query, self.desc_model, k=2)

        gmatches = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                gmatches.append(m)

        point2D = np.empty((0, 2))
        point3D = np.empty((0, 3))

        for mat in gmatches:
            query_idx = mat.queryIdx
            model_idx = mat.trainIdx
            point2D = np.vstack((point2D, kp_query[query_idx]))
            point3D = np.vstack((point3D, self.kp_model[model_idx]))

        return point3D, point2D


def main():
    # load data
    print("Loading data......")
    images_df = pd.read_pickle("data/task3/images.pkl")
    train_df = pd.read_pickle("data/task3/train.pkl")
    points3D_df = pd.read_pickle("data/task3/points3D.pkl")
    point_desc_df = pd.read_pickle("data/task3/point_desc.pkl")
    image_id = images_df["IMAGE_ID"].to_list()
    print("data is now available!")

    # create match
    match = Matching(images_df, train_df, points3D_df, point_desc_df)

    # create solver
    pnpSolver = P3P(match.cameraMatrix, match.distCoeffs, PCS_to_CCS, trilateration)

    # store all the R and t
    R_result = []
    T_result = []
    R_gt = []
    T_gt = []
    img_size = [1920, 1080]
    
    # main loop
    if(os.path.isfile("result/task3/Rotation.npy") and
       os.path.isfile("result/task3/Translation.npy") and
       os.path.isfile("result/task3/gtRotation.npy") and
       os.path.isfile("result/task3/gtTranslation.npy")):

        print("Rotation and Translation exist, skip solving PnP.")
        R_result = np.load("result/task3/Rotation.npy")
        T_result = np.load("result/task3/Translation.npy")
        R_gt = np.load("result/task3/gtRotation.npy")
        T_gt = np.load("result/task3/gtTranslation.npy")
    else:
        for i in range(len(image_id)):
            # process image one by one
            idx = image_id[i]

            # read image
            points3D, points2D = match.find_match(idx)
            points3D = points3D.T
            points2D = points2D.T

            points2D = image_undistortion(points2D, match.distCoeffs, img_size)

            # load gt
            ground_truth = images_df.loc[images_df["IMAGE_ID"] == idx]
            rotq_gt = ground_truth[["QX", "QY", "QZ", "QW"]].values
            tvec_gt = ground_truth[["TX", "TY", "TZ"]].values

            print("\n[{}/{}] {}, select {} points".format(i, len(image_id),
                  ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0], points2D.shape[-1]))

            # solve PnP
            R, T = ransac(pnpSolver, points3D, points2D)

            R_result.append(R)
            T_result.append(T)
            R_gt.append(np.squeeze(rotq_gt))
            T_gt.append(np.squeeze(tvec_gt))

        R_result = np.array(R_result)
        T_result = np.array(T_result)
        R_gt = np.array(R_gt)
        T_gt = np.array(T_gt)

        np.save("result/task3/Rotation.npy", R_result)
        np.save("result/task3/Translation.npy", T_result)
        np.save("result/task3/gtRotation.npy", R_gt)
        np.save("result/task3/gtTranslation.npy", T_gt)

    ###############
    ### Display ###
    ###############
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    pcd = load_point_cloud(points3D_df)
    vis.add_geometry(pcd)
    
    for i in range(0, R_result.shape[0], 50):
        idx = image_id[i]
        print(((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0])
        R = Rotation.from_quat(R_result[i]).as_matrix()
        T = T_result[i].reshape(3, 1)
        line_set = create_camera_position(match.cameraMatrix, R, T, img_size)
        vis.add_geometry(line_set)

    # # set a proper initial camera view
    # view_ctl = vis.get_view_control()
    # param = o3d.io.read_pinhole_camera_trajectory("ScreenCamera_2022-03-09-16-05-00.json")
    # view_ctl.convert_to_pinhole_camera_parameters(param.intrinsic, param.extrinsic[0])
    
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
