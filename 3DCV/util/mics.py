"""
Utilities for 3D computer vision tasks.
"""
import numpy as np
import cv2 as cv
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

