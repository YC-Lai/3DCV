import sys
try:
    sys.path.remove('/home/aicenteruav/catkin_ws/devel/lib/python2.7/dist-packages')
except:
    print("no ros")

from cv2 import cv2 as cv
import numpy as np
from util.helper import (homography, DLT, get_sift_correspondences)

if __name__ == '__main__':
    img1 = cv.imread("images/1-0.png")
    img2 = cv.imread("images/1-{}.png".format(sys.argv[1]))
    gt = np.load(
        "gt/task1/correspondence_0{}.npy".format(sys.argv[1]))
    
    points1, points2 = get_sift_correspondences(img1, img2)
    
    H = homography(points1, points2, k=20)
    
    # transform
    projection = DLT(H, np.transpose(gt[0]))
    
    # calculate loss
    loss =  np.linalg.norm(projection - gt[1]) / len(projection)
    
    print(loss)