from cv2 import cv2 as cv
import numpy as np
from util.helper import (backward_warping)
from util.mics import choose_point

if __name__ == '__main__':
    raw_img = cv.imread("data/task2/1-3.jpg")
    img = cv.resize(raw_img, (500, int(raw_img.shape[0]*(500/raw_img.shape[1]))))
    selectedPoints, img_w_point = choose_point(img, save=True)
    warped_img = backward_warping(img, selectedPoints)
    result = np.concatenate((img_w_point, warped_img), axis=1)
    # plot
    cv.imwrite('result/1-3.jpg', result)
    cv.namedWindow('book', cv.WINDOW_NORMAL)
    cv.imshow('book', warped_img)
    
    while cv.getWindowProperty('just_a_window', cv.WND_PROP_VISIBLE) >= 1:
        keyCode = cv.waitKey(1000)
        if (keyCode & 0xFF) == ord("q"):
            cv.destroyAllWindows()
            break