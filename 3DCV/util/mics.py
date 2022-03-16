"""
Utilities for 3D computer vision tasks.
"""
import numpy as np
import cv2 as cv
import open3d as o3d


def choose_point(img, save=False):

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


class Point_class():
    def __init__(self, position, color):
        self.position = position
        self.color = color


def generate_points(cube_vertice):

    point_list = []
    # top
    top_surface = list(cube_vertice[:4])
    top_x = (top_surface[1]-top_surface[0])/9
    top_y = (top_surface[2]-top_surface[0])/9
    for i in range(8):
        point_row_pose = top_surface[0] + (i+1)*top_x
        for j in range(10):
            point_pose = point_row_pose + j*top_y
            point_list.append(Point_class(point_pose, (255, 0, 0)))

    # front
    front_surface = list(cube_vertice[[0, 1, 4, 5]])
    front_x = (front_surface[1]-front_surface[0])/9
    front_y = (front_surface[2]-front_surface[0])/9
    for i in range(8):
        point_row_pose = front_surface[0] + (i+1)*front_x
        for j in range(8):
            point_pose = point_row_pose + (j+1)*front_y
            point_list.append(Point_class(point_pose, (0, 255, 0)))

    # back
    back_surface = list(cube_vertice[[2, 3, 6, 7]])
    back_x = (back_surface[1]-back_surface[0])/9
    back_y = (back_surface[2]-back_surface[0])/9
    for i in range(8):
        point_row_pose = back_surface[0] + (i+1)*back_x
        for j in range(8):
            point_pose = point_row_pose + (j+1)*back_y
            point_list.append(Point_class(point_pose, (255, 0, 255)))

    # botton
    botton_surface = list(cube_vertice[[4, 5, 6, 7]])
    botton_x = (botton_surface[1]-botton_surface[0])/9
    botton_y = (botton_surface[2]-botton_surface[0])/9
    for i in range(8):
        point_row_pose = botton_surface[0] + (i+1)*botton_x
        for j in range(10):
            point_pose = point_row_pose + j*botton_y
            point_list.append(Point_class(point_pose, (0, 0, 255)))

    # right
    right_surface = list(cube_vertice[[1, 3, 5, 7]])
    right_x = (right_surface[1]-right_surface[0])/9
    right_y = (right_surface[2]-right_surface[0])/9
    for i in range(10):
        point_row_pose = right_surface[0] + i*right_x
        for j in range(10):
            point_pose = point_row_pose + j*right_y
            point_list.append(Point_class(point_pose, (255, 255, 0)))

    # left
    left_surface = list(cube_vertice[[0, 2, 4, 6]])
    left_x = (left_surface[1]-left_surface[0])/9
    left_y = (left_surface[2]-left_surface[0])/9
    for i in range(10):
        point_row_pose = left_surface[0] + i*left_x
        for j in range(10):
            point_pose = point_row_pose + j*left_y
            point_list.append(Point_class(point_pose, (0, 255, 255)))

    return point_list
