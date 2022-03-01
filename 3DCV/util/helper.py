"""
Utilities for 3D computer vision tasks.
"""
import numpy as np
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


def interpolation2D(img, x):
    '''     | aw
        d + + + + + + c
          +     |   +  ah
        --+-----x---+-------
          +     |   +
       ch +     |   +
          + + + + + +
        a   cw  |     b
    '''
    ceil, floor = np.ceil(x), np.floor(x)
    rows = np.array([
        [ceil[1], ceil[1]],
        [floor[1], floor[1]]
    ], dtype=int)
    cols = np.array([
        [floor[0], ceil[0]],
        [floor[0], ceil[0]]
    ], dtype=int)

    aw = ceil[0] - x[0]
    ah = ceil[1] - x[1]
    cw = x[0] - floor[0]
    ch = x[1] - floor[1]

    weight = np.array([
        [aw*ch, cw*ch],
        [aw*ah, cw*ah]
    ])
    weight = np.reshape(weight, (2, 2, 1))
    img = img[rows, cols]
    P = np.sum(np.multiply(weight, img), axis=(0, 1))

    return np.round(P).astype(np.uint8)