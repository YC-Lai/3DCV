"""
Utilities for 3D computer vision tasks.
"""
import numpy as np
from cv2 import cv2 as cv
from scipy.spatial.distance import euclidean as distance


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
    randomPoints = np.random.randint(k, size=pointSet1.shape[0])
    if pointSet1.shape[1] == 2:
        for i in randomPoints:
            x, y = pointSet1[i][0], pointSet1[i][1]
            u, v = pointSet2[i][0], pointSet2[i][1]
            A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
    else:
        for i in randomPoints:
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


def p3p(points3D, points2D, cameraMatrix, size, distCoeffs=None):
    '''
    Parameters:
        points3D [3, 4]: scene point in WCS (world coordinate system)
        points2D [2, 4]: image point in PCS (pixel coordinate system)
        cameraMatrix [3, 3]: intrinsic matrix
        size [M, N]: image size
        distCoeffs [4,]: distortion parameters

    Return:
        R: rotation matrix
        T: translation matrix
    '''

    assert points3D.shape[0] == 3 and points3D.shape[1] == 4
    assert points2D.shape[0] == 2 and points2D.shape[1] == 4
    assert cameraMatrix.shape[0] == 3 and cameraMatrix.shape[1] == 3

    # Step 1: compute angles
    x = PCS_to_CCS(points2D, cameraMatrix, distCoeffs, size)
    ca = np.dot(x[:, 1], x[:, 2])
    cb = np.dot(x[:, 0], x[:, 2])
    cc = np.dot(x[:, 0], x[:, 1])

    # Step 2: compute distances
    a = distance(points3D[:, 1], points3D[:, 2])
    b = distance(points3D[:, 0], points3D[:, 2])
    c = distance(points3D[:, 0], points3D[:, 1])

    A4 = ((a**2-c**2)/(b**2) - 1)**2 - ((2*c/b)**2) * ca**2
    A3 = 4 * (((a**2-c**2)/(b**2)) * (1 - (a**2-c**2)/(b**2)) * cb -
              (1 - (a**2+c**2)/(b**2)) * ca * cc + (2*(c**2/b**2) * ca**2 * cb))
    A2 = 2 * ((((a**2-c**2)/(b**2))**2 - 1) + (2*((a**2-c**2)/(b**2))**2 * cb**2) +
              (2*((b**2-c**2)/(b**2))*ca**2) - (4*((a**2+c**2)/(b**2))*ca*cb*cc) + (2*((b**2-a**2)/(b**2))*cc**2))
    A1 = 4 * (-((a**2-c**2)/(b**2))*(1 + (a**2-c**2)/(b**2))*cb + 2 *
              (a**2/b**2)*cc**2*cb - (1 + (a**2+c**2)/(b**2))*ca*cc)
    A0 = (1 + (a**2-c**2)/(b**2))**2 - (2*a/b)**2*cc**2

    roots = np.roots([A4, A3, A2, A1, A0])

    lengths = []
    # for each solution of s
    for v in roots:
        if np.iscomplex(v):
            continue
        else:
            v = np.real(v)
            if v < 0:
                continue

        s1 = (b**2 / (1 + v**2 - 2 * v * cb))**0.5
        s3 = v * s1

        q = (-2*s3*ca)**2 - 4*(s3**2-a**2)
        if q > 0:
            s2_1 = (-(-2*s3*ca) + q**0.5) / 2
            s2_2 = (-(-2*s3*ca) - q**0.5) / 2
            for s2 in [s2_1, s2_2]:
                if s2 > 0:
                    lengths.append([s1, s2, s3])
    lengths = np.array(lengths)

    # Step 3: identify correct solution through 4th point
    x4 = x[:, -1]
    x = x[:, 0:3]
    X4 = points3D[:, -1]
    X = points3D[:, 3]
    solutions = []
    for length in lengths:
        length = np.array(length).reshape((1, 3))
        T1, T2 = trilateration(X[:, 0], X[:, 1], X[:, 2], length[0], length[1], length[2])
        # identify T
        for T in [T1, T2]:
            T = T.reshape((3, 1))
            R = (length * x) @ np.linalg.pinv(X - T)
            solutions.append([R, T, length])

    best_R = solutions[0][0]
    best_T = solutions[0][1]
    error = np.Inf
    for R, T, length in solutions:
        proj_x = R @ (X4.reshape((3, 1)) - T)
        if np.linalg.norm(proj_x - (length * x4)) < error:
            best_R = R
            best_T = T

    # Step 4: compute coordinate transformation
    return best_R, best_T


def PCS_to_CCS(points, cameraMatrix, distCoeffs, size):
    '''
    Transform from PCS (pixel coordinate system) to CCS (camera coordinate system)

    Parameters:
        points [2, n]: image point in pixel coordinate system
        cameraMatrix [3, 3]: intrinsic matrix
        size [M, N]: image size

    Return:
        interpolated image [3, n]
    '''

    assert points.shape[0] == 2
    assert cameraMatrix.shape[0] == 3 and cameraMatrix.shape[1] == 3
    assert len(distCoeffs) == 4

    # undistorted image (Brown-Conrady)
    if distCoeffs is not None:
        points = image_undistortion(points, distCoeffs, size)

    invCM = np.linalg.pinv(cameraMatrix)
    v = invCM @ np.concatenate((points, np.ones((1, points.shape[1]))))

    return v / np.linalg.norm(v, axis=1)


def image_undistortion(points, distCoeffs, size):
    '''
    Image undistortion, use Brown-Conrady model.

    Parameters:
        points [2, n]: image point in pixel coordinate system
        distCoeffs [4,]: distortion parameters
        size [M, N]: image size

    Return:
        undistorted image [2, n]
    '''
    assert points.shape[0] == 2
    assert len(distCoeffs) == 4

    center = np.array([size[1] // 2, size[0] // 2]).reshape((2, 1))
    r = np.linalg.norm((points - center), axis=1)

    xc, yc = center[0], center[1]
    x, y = points[0], points[1]
    k1, k2, p1, p2 = distCoeffs[0], distCoeffs[1], distCoeffs[2], distCoeffs[3]

    ux = x + (x - xc) * (k1 * (r**2) + k2 * (r**4)) + \
        (p1 * (r**2 + 2*((x - xc)**2)) + 2 * p2 * (x - xc) * (y - yc))
    uy = y + (y - yc) * (k1 * (r**2) + k2 * (r**4)) + \
        (p2 * (r**2 + 2*((y - yc)**2)) + 2 * p1 * (x - xc) * (y - yc))

    return np.vstack((ux, uy))


def trilateration(P1, P2, P3, r1, r2, r3):

    p1 = np.array([0, 0, 0])
    p2 = np.array([P2[0] - P1[0], P2[1] - P1[1], P2[2] - P1[2]])
    p3 = np.array([P3[0] - P1[0], P3[1] - P1[1], P3[2] - P1[2]])
    v1 = p2 - p1
    v2 = p3 - p1

    Xn = (v1)/np.linalg.norm(v1)

    tmp = np.cross(v1, v2)

    Zn = (tmp)/np.linalg.norm(tmp)

    Yn = np.cross(Xn, Zn)

    i = np.dot(Xn, v2)
    d = np.dot(Xn, v1)
    j = np.dot(Yn, v2)

    X = ((r1**2)-(r2**2)+(d**2))/(2*d)
    Y = (((r1**2)-(r3**2)+(i**2)+(j**2))/(2*j))-((i/j)*(X))
    Z1 = np.sqrt(max(0, r1**2-X**2-Y**2))
    Z2 = -Z2

    K1 = P1 + X * Xn + Y * Yn + Z1 * Zn
    K2 = P1 + X * Xn + Y * Yn + Z2 * Zn
    return K1, K2


