import sys

from matplotlib import projections
try:
    sys.path.remove('/home/aicenteruav/catkin_ws/devel/lib/python2.7/dist-packages')
except:
    print("no ros")

from cv2 import cv2 as cv
import numpy as np
from util.helper import (homography, get_sift_correspondences)

