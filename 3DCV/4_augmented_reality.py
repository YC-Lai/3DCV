import pandas as pd
import numpy as np
import cv2
from scipy.spatial.transform import Rotation


def main():
    images_df = pd.read_pickle("data/task3/images.pkl")
    image_name = images_df["IMAGE_ID"].to_list()
    valid_img = []
    idx_list = []
    for i in range(len(image_name)):
        idx = image_name[i]
        if("valid" in ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]):
            print("processing {}/{} image {}".format(i, len(image_name),
                                                     ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]))
        else:
            continue

        fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
        rimg = cv2.imread("data/task3/frames/"+fname, cv2.IMREAD_COLOR)

        valid_img.append(rimg)
        idx_list.append(idx)

    # create 3D cube
    cube3D = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                         [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]])

    # render the cube in each image
    R_result = np.load("result/task3/Rotation.npy")
    T_result = np.load("result/task3/Translation.npy")
    cameraMatrix = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])
    for i, img in enumerate(valid_img):
        idx = idx_list[i]
        R = Rotation.from_quat(R_result[idx-1]).as_matrix()
        T = T_result[idx-1].reshape(3, 1)
        cube2D = cameraMatrix @ R @ (cube3D.T - T)
        cube2D /= cube2D[-1, :].reshape(1, -1)
        cube2D = np.round(cube2D[:2, :].T).astype(int)

        # draw ground floor in green
        img = cv2.drawContours(img, [cube2D[4:]], -1, (0, 255, 0), -3)

        # draw pillars in blue color
        for i, j in zip(range(4), range(4, 8)):
            img = cv2.line(img, tuple(cube2D[i]), tuple(cube2D[j]), (255), 3)

        # draw top layer in red color
        img = cv2.drawContours(img, [cube2D[:4]], -1, (0, 0, 255), 3)

        while(True):
            image = cv2.resize(img, (540, 960), interpolation=cv2.INTER_AREA)
            cv2.imshow('frame', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter("ARVideo.avi", fourcc, 1,
    #                       (valid_img[0].shape[0], valid_img[0].shape[1]), isColor=True)
    # for i in range(len(valid_img)):
    #     out.write(valid_img[i])

    # out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
