"""
Utilities for 3D computer vision tasks.
"""
import numpy as np
from cv2 import cv2 as cv
import open3d as o3d


def choose_point(img, save = False):

    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            param[0].append([x, y])

    WINDOW_NAME = 'window'

    points_add = []
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.setMouseCallback(WINDOW_NAME, on_mouse, [points_add])

    while True:
        img_ = img.copy()
        for i, p in enumerate(points_add):
            # draw points on img_
            cv.circle(img_, tuple(p), 10, (0, 255, 0), -1)
        cv.imshow(WINDOW_NAME, img_)

        key = cv.waitKey(20) % 0xFF
        if key == 27:
            break  # exist when pressing ESC

    cv.destroyAllWindows()
    print('{} Points added'.format(len(points_add)))

    if save:
        return np.array(points_add), img_
    return np.array(points_add)


class Display():
    def __init__(self, points3D, camera):
        points, pointColor = points3D

        all_rotation, all_position = camera

    def draw(self):
        pcd = self.load_point_cloud()
        o3d.visualization.draw_geometries([pcd])

    def load_point_cloud(self):
        xyz = np.vstack(self.points3D_df['XYZ'])
        rgb = np.vstack(self.points3D_df['RGB'])/255

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        return pcd