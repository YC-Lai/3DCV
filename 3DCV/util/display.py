import open3d as o3d
import cv2 as cv
import numpy as np
import pandas as pd


def load_point_cloud(points3D_df: pd.DataFrame):
    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    return pcd


def create_cube():
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])          # X, Y, Z
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # R, G, B
    return axes


def create_camera_position(camera_intrinsic, R, T, img_size, color=[1, 0, 0]):
    assert T.shape[0] == 3 and T.shape[1] == 1

    camera_corner = np.array([[0, 0, 1],
                              [0, img_size[0], 1],
                              [img_size[1], img_size[0], 1],
                              [img_size[1], 0, 1]]).T
    v = np.linalg.pinv(camera_intrinsic) @ camera_corner
    camera_corner_3d = (np.linalg.pinv(R) @ v + T)
    # add center
    camera_corner_3d = np.concatenate((camera_corner_3d, T), axis=1)
    visualized_points = camera_corner_3d.T
    # create o3d object
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(visualized_points),
        lines=o3d.utility.Vector2iVector([[0, 1], [1, 2], [2, 3], [0, 3], [0, 4], [1, 4], [2, 4], [3, 4]])
    )
    colors = np.tile(color, (8, 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set