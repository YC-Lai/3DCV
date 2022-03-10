import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import euclidean as distance
from scipy.spatial.transform import Rotation
from typing import (Callable, Any, Tuple)


class P3PRansac():
    def __init__(self, img_size, cameraMatrix, distCoeffs=None, s=4, e=0.5, p=0.99, d=10) -> None:
        """
        P3P with RANSAC algorithm

        RANSAC Parameters:
            s: number of points in a sample
            e: probabiliaty that a point is an outlier
            p: desired probability that we get a good sample
            d: distance threshold ( np.sqrt(5.99 * (self.s**2)) )
        """
        assert cameraMatrix.shape[0] == 3 and cameraMatrix.shape[1] == 3
        assert len(distCoeffs) == 4

        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        self.img_size = img_size

        # RANSAC parameters
        self.N = np.log((1 - p)) / np.log(1 - np.power((1 - e), s))  # number of samples
        self.d = d

    def __call__(self, point3D, point2D) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters:
            point3D [3, n]: scene point in WCS (world coordinate system)
            point2D [2, n]: image point in PCS (pixel coordinate system)
        """
        assert point3D.shape[0] == 3 and point2D.shape[0] == 2

        if self.distCoeffs is not None:
            point2D = self.image_undistortion(point2D)

        best_R = None
        best_T = None
        min_n_outliers = np.Inf
        for i in range(self.N):
            # sample
            idx = np.random.randint(point2D.shape[1], size=4)
            # idx = [0,1000,2000,3000]
            sample3D = point3D[:, idx]
            sample2D = point2D[:, idx]
            try:
                # compute
                R, T = self.solve_P3P(sample3D, sample2D)
                # score
                projection = self.cameraMatrix @ (R @ (point3D - T.reshape(3, 1)))
                projection /= projection[-1, :].reshape((1, -1))
                errors = norm(projection[:2, :] - point2D, axis=0)

                n_outliers = len(errors[np.where(errors > self.d)])
                if n_outliers < min_n_outliers:
                    # print("#{} update R and T, number of outlier down to {}".format(i, n_outliers))
                    best_R = R
                    best_T = T
                    min_n_outliers = n_outliers
            except:
                print("#{} round can not be solved".format(i))

        print('=== Summary ===')
        print("number of inlierr: {}".format(point3D.shape[1] - min_n_outliers))
        print("number of outlierr: {}".format(min_n_outliers))

        best_R = Rotation.from_matrix(best_R).as_quat()
        best_T = best_T.reshape(-1)

        return best_R, best_T

    def solve_P3P(self, point3D, point2D):
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

    def PCS_to_CCS(self, points) -> np.ndarray:
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

        invCM = np.linalg.pinv(self.cameraMatrix)
        v = invCM @ np.concatenate((points, np.ones((1, points.shape[1]))))
        return v / np.linalg.norm(v, axis=0)

    def image_undistortion(self, points) -> np.ndarray:
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

        # normalize
        points /= np.array([self.img_size[1], self.img_size[0]]).reshape((2, 1))

        center = np.array([0.5, 0.5]).reshape((2, 1))
        r = norm((points - center), axis=0)

        xc, yc = center[0], center[1]
        x, y = points[0], points[1]
        k1, k2, p1, p2 = self.distCoeffs[0], self.distCoeffs[1], self.distCoeffs[2], self.distCoeffs[3]

        ux = x + (x - xc) * (k1 * (r**2) + k2 * (r**4)) + \
            (p1 * (r**2 + 2*((x - xc)**2)) + 2 * p2 * (x - xc) * (y - yc))
        uy = y + (y - yc) * (k1 * (r**2) + k2 * (r**4)) + \
            (p2 * (r**2 + 2*((y - yc)**2)) + 2 * p1 * (x - xc) * (y - yc))

        undistorted_points = np.vstack((ux, uy)) * np.array([self.img_size[1], self.img_size[0]]).reshape((2, 1))
        return undistorted_points

    def trilateration(P1, P2, P3, r1, r2, r3) -> Tuple[np.ndarray, np.ndarray]:
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
