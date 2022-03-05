"""
Utilities for 3D computer vision tasks.
"""
import numpy as np
from scipy.interpolate import griddata
from cv2 import cv2 as cv


def homography(pointSet1: np.array, pointSet2: np.array, k) -> np.array:
    """
    Parameters:
        pointSet1: 2D or 3D coordinate set
        pointSet2: image coordinate set
        k: number of selected point

    Return:
        H: homography matrix
    """

    def normalization(pointSet: np.array):
        means = np.mean(pointSet, axis=0)
        s = np.sqrt(np.mean((pointSet - means)**2) / 2)

        if pointSet.shape[1] == 2:
            T = np.array([
                [1/s, 0, -means[0]/s],
                [0, 1/s, -means[1]/s],
                [0, 0, 1]
            ])
        else:
            T = np.array([
                [1/s, 0, 0, -means[0]/s],
                [0, 1/s, 0, -means[1]/s],
                [0, 0, 1/s, -means[2]/s],
                [0, 0, 0, 1]
            ])

        points_homo = np.concatenate(
            [np.transpose(pointSet), np.ones((1, len(pointSet)))], axis=0)
        normalizedPoints = np.matmul(T, points_homo)

        return T, np.delete(np.transpose(normalizedPoints), -1, 1)

    assert pointSet1.shape[0] >= k and pointSet2.shape[0] >= k
    assert pointSet1.shape[1] == 2 or pointSet1.shape[1] == 3
    assert pointSet2.shape[1] == 2

    # normalization
    T1, pointSet1 = normalization(pointSet1)
    T2, pointSet2 = normalization(pointSet2)

    A = []
    if pointSet1.shape[1] == 2:
        for i in range(0, k):
            x, y = pointSet1[i][0], pointSet1[i][1]
            u, v = pointSet2[i][0], pointSet2[i][1]
            A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
    else:
        for i in range(0, k):
            x, y, z = pointSet1[i][0], pointSet1[i][1], pointSet1[i][2]
            u, v = pointSet2[i][0], pointSet2[i][1]
            A.append([0, 0, 0, 0, -x, -y, -z, -1, v*x, v*y, v*z, v])
            A.append([x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u])

    A = np.asarray(A)
    _, _, Vh = np.linalg.svd(A)

    if pointSet1.shape[1] == 2:
        H = Vh[-1, :].reshape(3, 3)
    else:
        H = Vh[-1, :].reshape(4, 4)

    return np.linalg.inv(T2) @ H @ T1


def DLT(H, points):
    '''
    Parameters:
        H: homography matrix, shape = [3, 4] or [3, 3]
        points: input points, shape = [3, N] or [2, N]

    Return:
        projection, shape = [N, 3] or [N, 2]
    '''

    assert (H.shape[1] == 4 and points.shape[0] == 3) or (H.shape[1] == 3 and points.shape[0] == 2)
    projection = H @ np.concatenate([points, np.ones((1, points.shape[-1]))], axis=0)
    scale = projection[-1, :]
    projection /= scale[np.newaxis, :]
    projection = np.delete(np.transpose(projection), -1, 1)

    return projection


def get_sift_correspondences(img1, img2):
    '''
    Parameters:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    # sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    return points1, points2


def backward_warping(img, selectedPoints):

    # initialize warped image
    (x, y) = img.shape[:2]
    warpedPoints = np.array([
        [0, 0],
        [x-1, 0],
        [x-1, y-1],
        [0, y-1]
    ])
    warped_img = np.full((x, y, 3), (0, 0, 255), dtype=np.uint8)

    # get backward homography matrix
    H = homography(selectedPoints, warpedPoints, len(selectedPoints))
    backward_H = np.linalg.inv(H)

    # get image coords
    grid = np.indices((warped_img.shape[0], warped_img.shape[1])).reshape(2, -1)

    # back transform
    backward_coords = DLT(backward_H, grid)

    # # result
    temp = bilinear_interpolate(img, backward_coords[:, 0], backward_coords[:, 1])
    warped_img = temp.reshape((x, y, 3))

    return warped_img


def bilinear_interpolate(im, x, y):
    '''     
                | 
        b + + + + + + d
          +     |   +  
        --+-----x---+--
          +     |   +
          +     |   +
          + + + + + +
        a       |     c

    Parameters:
        im: input image
        x: 1D numpy array [N,]
        y: 1D numpy array [N,]

    Return:
        interpolated image [N, 3]
    '''
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = ((x1-x) * (y1-y))[:, np.newaxis]
    wb = ((x1-x) * (y-y0))[:, np.newaxis]
    wc = ((x-x0) * (y1-y))[:, np.newaxis]
    wd = ((x-x0) * (y-y0))[:, np.newaxis]

    return wa*Ia + wb*Ib + wc*Ic + wd*Id
