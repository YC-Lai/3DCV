import numpy as np
import pandas as pd
from util.helper import (P3P, ransac, Matching)
from util.mics import (Display)
import os


def main():
    # load data
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")
    image_id = images_df["IMAGE_ID"].to_list()

    # create match
    match = Matching(images_df, train_df, points3D_df, point_desc_df)

    # store all the R and t
    store_R = []
    store_t = []
    gt_R = []
    gt_t = []
    
if __name__ == '__main__':
    main()