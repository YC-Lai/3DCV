import glob
import os
from util.visualOdometry import visualOdometry
import numpy as np

if __name__ == '__main__':
    img_paths = sorted(list(glob.glob(os.path.join("data/task5/frames/", '*.png'))))
    camera_params = np.load("data/task5/camera_parameters.npy", allow_pickle=True)[()]
    intrinsic = camera_params['K']
    distCoeffs = camera_params['dist']
    
    vo = visualOdometry(intrinsic, distCoeffs, img_paths)
    vo.run()
    