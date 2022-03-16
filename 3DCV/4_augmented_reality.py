import pandas as pd
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import open3d as o3d
from util.mics import generate_points


def draw_cube(img, rotation, translation, cube_vertice):
    cameraMatrix = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])
    rotation = Rotation.from_quat(rotation).as_matrix()
    points = generate_points(cube_vertice)

    points.sort(key=lambda point: np.linalg.norm((point.position-translation)), reverse=True)

    for i in range(len(points)):
        pixel = (cameraMatrix @ (rotation @ (points[i].position - translation).T)).T
        pixel = (pixel/pixel[2])
        if((pixel < 0).any()):
            continue
        img = cv2.circle(img, (int(pixel[0]), int(pixel[1])), radius=5, color=points[i].color, thickness=-1)

    return img


def main():
    images_df = pd.read_pickle("data/task3/images.pkl")
    image_name = images_df["IMAGE_ID"].to_list()
    valid_img_name = []

    for i in range(len(image_name)):
        idx = image_name[i]
        if("valid" in ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]):
            fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
            valid_img_name.append(fname)
    valid_img_name = sorted(valid_img_name, key=lambda name: int(name[name.find('g')+1:name.find('.')]))

    valid_img = []
    store_R = []
    store_t = []
    R_result = np.load("result/task3/Rotation.npy")
    t_result = np.load("result/task3/Translation.npy")
    for i in range(len(valid_img_name)):
        print("processing {}/{} image {}".format(i, len(valid_img_name), valid_img_name[i]))
        idx = ((images_df.loc[images_df["NAME"] == valid_img_name[i]])["IMAGE_ID"].values)[0]
        fname = valid_img_name[i]
        rimg = cv2.imread("data/task3/frames/"+fname, cv2.IMREAD_COLOR)
        valid_img.append(rimg)
        store_R.append(R_result[idx-1])
        store_t.append(t_result[idx-1])
    shape = (int(valid_img[0].shape[1]), int(valid_img[0].shape[0]))
    
    # create 3D cube
    cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    cube_vertice = np.asarray(cube.vertices).copy()
    cube_vertice += np.array([3, -2, 1.5])
    cube_vertice *= 0.5

    for i in range(len(valid_img)):
        valid_img[i] = draw_cube(valid_img[i], store_R[i], store_t[i], cube_vertice)

    out = cv2.VideoWriter("ARVideo.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, shape)

    for i in range(len(valid_img)):
        out.write(valid_img[i])

    cv2.destroyAllWindows()
    out.release()


if __name__ == '__main__':
    main()
