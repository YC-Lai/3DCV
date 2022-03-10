"""
Utilities for 3D computer vision tasks.
"""
import numpy as np
from numpy.linalg import norm
import cv2 as cv
from scipy.spatial.distance import euclidean as distance
from scipy.spatial.transform import Rotation
from typing import Callable


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
        pointa: numpy array [N, 2], N is the number of correspondences
        pointb: numpy array [N, 2], N is the number of correspondences
    '''
    # sift = cv.xfeaturebd.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()             # opencv-python==4.5.1.48
    kp1, dea = sift.detectAndCompute(img1, None)
    kp2, deb = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(dea, deb, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    pointa = np.array([kp1[m.queryIdx].pt for m in good_matches])
    pointb = np.array([kp2[m.trainIdx].pt for m in good_matches])

    return pointa, pointb


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


class P3P():
    def __init__(self, cameraMatrix, distCoeffs, PCS_to_CCS: Callable, trilateration: Callable) -> None:
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

        self.PCS_to_CCS = PCS_to_CCS
        self.trilateration = trilateration

    def __call__(self, point3D, point2D):
        '''
        Parameters:
            point3D [3, 4]: scene point in WCS (world coordinate system)
            point2D [2, 4]: image point in PCS (pixel coordinate system)

        Return:
            R: rotation matrix
            T: translation matrix
        '''
        assert point3D.shape[0] == 3 and point3D.shape[1] == 4
        assert point2D.shape[0] == 2 and point2D.shape[1] == 4
        
        # Step 1: compute angles and distances betwen 3D points
        v = self.PCS_to_CCS(point2D, self.cameraMatrix, self.distCoeffs)
        
        C_ab = np.dot(v[:, 0], v[:, 1])
        C_ac = np.dot(v[:, 0], v[:, 2])
        C_bc = np.dot(v[:, 1], v[:, 2])

        R_ab = distance(point3D[:, 0], point3D[:, 1])
        R_ac = distance(point3D[:, 0], point3D[:, 2])
        R_bc = distance(point3D[:, 1], point3D[:, 2])
        
        # Step 2: compute distances
        K_1 = (R_bc/R_ac)**2
        K_2 = (R_bc/R_ab)**2
        G_4 = (K_1*K_2 - K_1 - K_2)**2 - 4*K_1*K_2*(C_bc**2)
        G_3 = 4*(K_1*K_2 - K_1 - K_2)*K_2*(1 - K_1)*C_ab \
            + 4*K_1*C_bc*((K_1*K_2 - K_1 + K_2) * C_ac + 2*K_2*C_ab*C_bc)
        G_2 = (2*K_2*(1-K_1)*C_ab)**2 \
            + 2*(K_1*K_2 - K_1 - K_2)*(K_1*K_2 + K_1 - K_2) \
            + 4*K_1*((K_1 - K_2)*(C_bc**2) + K_1*(1-K_2)*(C_ac**2) - 2*(1+K_1)*K_2*C_ab*C_ac*C_bc)
        G_1 = 4*(K_1*K_2 + K_1 - K_2)*K_2*(1-K_1)*C_ab \
            + 4*K_1*((K_1*K_2 - K_1 + K_2)*C_ac*C_bc + 2*K_1*K_2*C_ab*(C_ac**2))
        G_0 = (K_1*K_2 + K_1 - K_2)**2 \
            - 4*(K_1**2)*K_2*(C_ac**2)

        roots = np.roots([G_4, G_3, G_2, G_1, G_0])
        x = np.array([np.real(r) for r in roots if np.isreal(r)])
        # solve y
        m, p, q = (1 - K_1), 2*(K_1 * C_ac - x*C_bc), (x**2 - K_1)
        m_, p_, q_ = 1, 2*(-x*C_bc), (x**2)*(1-K_2) + 2*x*K_2*C_ab - K_2
        y = -1*(m_*q - m*q_)/(p*m_ - p_*m)
        
        a = np.sqrt((R_ab**2)/(1+(x**2)-2*x*C_ab))
        b = x * a
        c = y * a
        
        camera_center = []
        for i in range(len(a)):
            T1, T2 = self.trilateration(point3D[:, 0], point3D[:, 1], point3D[:, 2], a[i], b[i], c[i])
            camera_center.append(T1)
            camera_center.append(T2)

        # Step 3: compute lambda and R
        solutions = []
        for T in camera_center:
            T = T.reshape((3, 1))
            for sign in [1, -1]:
                lamda = sign * norm((point3D[:, :3] - T), axis=0)
                R = (lamda * v[:, :3]) @ np.linalg.pinv(point3D[:, :3] - T)
                # print(R.T@R)
                solutions.append([R, T, lamda])

        # Step 4: identify correct solution through 4th point
        best_R = solutions[0][0]
        best_T = solutions[0][1]
        min_error = np.Inf
        for R, T, lamda in solutions:
            proj_x = self.cameraMatrix @ R @ (point3D[:, 3].reshape(3, 1) - T)
            proj_x /= proj_x[-1]
            error = norm(proj_x[:2, :] - point2D[:, 3].reshape(2, 1))
            if error < min_error:
                best_R = R
                best_T = T
                min_error = error

        return best_R, best_T


def PCS_to_CCS(points, cameraMatrix, distCoeffs):
    '''
    Transform from PCS (pixel coordinate system) to CCS (camera coordinate system)

    Parameters:
        points [2, n]: image point in PCS
        cameraMatrix [3, 3]: intrinsic matrix
        size [M, N]: image size

    Return:
        points [3, n]: image point in CCS
    '''
    assert points.shape[0] == 2
    assert cameraMatrix.shape[0] == 3 and cameraMatrix.shape[1] == 3
    assert len(distCoeffs) == 4

    invCM = np.linalg.pinv(cameraMatrix)
    v = invCM @ np.concatenate((points, np.ones((1, points.shape[1]))))    
    return v / np.linalg.norm(v, axis=0)


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

    # normalize
    points /= np.array([size[1], size[0]]).reshape((2, 1))

    center = np.array([0.5, 0.5]).reshape((2, 1))
    r = norm((points - center), axis=0)

    xc, yc = center[0], center[1]
    x, y = points[0], points[1]
    k1, k2, p1, p2 = distCoeffs[0], distCoeffs[1], distCoeffs[2], distCoeffs[3]

    ux = x + (x - xc) * (k1 * (r**2) + k2 * (r**4)) + \
        (p1 * (r**2 + 2*((x - xc)**2)) + 2 * p2 * (x - xc) * (y - yc))
    uy = y + (y - yc) * (k1 * (r**2) + k2 * (r**4)) + \
        (p2 * (r**2 + 2*((y - yc)**2)) + 2 * p1 * (x - xc) * (y - yc))

    undistorted_points = np.vstack((ux, uy)) * np.array([size[1], size[0]]).reshape((2, 1))
    return undistorted_points


def trilateration(P1, P2, P3, r1, r2, r3):
    '''
    Compute the two common points of three shperes centered in x with r

    Parameters:
        P1, P2, P3 [3,]: 3D coordinate
        r1, r2, r3 [1,]: radius

    Return:
        T1, T2: two common points
    '''
    v1 = P2 - P1
    v2 = P3 - P1
    i_v1 = v1 / norm(v1)
    i_v2 = v2 / norm(v2)

    # unit vector
    i_x = v1 / norm(v1)
    i_z = (np.cross(i_v1, i_v2)) / norm(np.cross(i_v1, i_v2))
    i_y = np.cross(i_x, i_z)

    c1 = np.array([0, 0, 0])
    c2 = np.array([np.dot(i_x, v1), 0, 0])
    c3 = np.array([np.dot(i_x, v2), np.dot(i_y, v2), 0])

    proj_x = ((r1**2) - (r2**2) + (c2[0]**2)) / (2*c2[0])
    temp = (c3[0]**2) + (c3[1]**2)
    proj_y = ((r1**2) - (r3**2) + temp - (2*c3[0]*proj_x)) / (2*c3[1])
    proj_z = np.sqrt(r1**2 - proj_x**2 - proj_y**2)

    direction_1 = proj_x * i_x + proj_y * i_y + proj_z * i_z
    direction_2 = proj_x * i_x + proj_y * i_y - proj_z * i_z

    T1 = P1 + direction_1
    T2 = P1 + direction_2

    return T1, T2


def ransac(pnpSolver, point3D, point2D, s=3, e=0.5, p=0.99, d=10):
    """
    RANSAC algorithm

    Parameters:
        pnpSolver: any pnp algorithm to get R and T.
        point3D [3, n]: scene point in WCS (world coordinate system)
        point2D [2, n]: image point in PCS (pixel coordinate system)
        s: number of points in a sample
        e: probabiliaty that a point is an outlier
        p: desired probability that we get a good sample
        d: distance threshold ( np.sqrt(5.99 * (self.s**2)) )
    """    
    assert point3D.shape[0] == 3 and point2D.shape[0] == 2
    
    # Ransac parameter
    N = np.log((1 - p)) / np.log(1 - np.power((1 - e), s))  # number of samples

    best_R = None
    best_T = None
    min_n_outliers = np.Inf
    for i in range(round(N)):
        # sample
        idx = np.random.randint(point2D.shape[1], size=4)
        # idx = [0,1000,2000,3000]
        sample3D = point3D[:, idx]
        sample2D = point2D[:, idx]
        try:
            # compute
            R, T = pnpSolver(sample3D, sample2D)
            # score
            projection = pnpSolver.cameraMatrix @ (R @ (point3D - T.reshape(3, 1)))
            projection /= projection[-1, :].reshape((1, -1))
            errors = norm(projection[:2, :] - point2D, axis=0)

            n_outliers = len(errors[np.where(errors > d)])
            if n_outliers < min_n_outliers:
                print("#{} update R and T, number of outlier down to {}".format(i, n_outliers))
                best_R = R
                best_T = T
                min_n_outliers = n_outliers
        except:
            print("#{} round can not be solved".format(i))
            
    best_R = Rotation.from_matrix(best_R).as_quat()
    best_T = best_T.reshape(-1)
    print("\nnumber of outlierr: {}".format(min_n_outliers))
    print("Rotation:")
    print(best_R)
    print("Camera Position:")
    print(best_T)

    return best_R, best_T



