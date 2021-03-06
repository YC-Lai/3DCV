import cv2 as cv
import numpy as np
from util.helper import (homography, DLT, get_sift_correspondences)

if __name__ == '__main__':
    img0 = cv.imread("data/task1/1-0.png")
    img1 = cv.imread("data/task1/1-1.png")
    img2 = cv.imread("data/task1/1-2.png")
    gt1 = np.load("gt/task1/correspondence_01.npy")
    gt2 = np.load("gt/task1/correspondence_02.npy")

    imgSet = [img0, img1, img2]
    gtSet = [gt1, gt2]

    for i in range(2):
        points1, points2 = get_sift_correspondences(img0, imgSet[i+1])
        print("img 1-{}".format(i+1))
        for k in [4, 8, 20, 80]:
            H = homography(points1, points2, k)
            # transform
            projection = DLT(H, np.transpose(gtSet[i][0]))
            # calculate loss
            loss = np.linalg.norm(projection - gtSet[i][1]) / len(projection)
            print("k = {}: loss = {}".format(k, loss))


