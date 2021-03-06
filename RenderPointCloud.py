"""
This file is for generating training data.
It generates re-projected 3D point image, which can be normal map or rgb image.

NOTE:
In blender, we move a camera on surface of a sphere coordinate system with radius R,
angle theta and phi, all camera look at center of object.

In blender world, x right, y inside, z up.
In this project, a 3D point cloud is in "screen", x right, y bottom, z inside.

If we put the point cloud in a screen, then we see the bottom of the model.
So we use two rotations and one translation to simulate Blender.
The first rotation rotate the point cloud around x axis,
then the second rotation rotate the point cloud around its local z axis.
finally translate the point cloud along z axis with value R.

First rotation: rotate object around its (also global) x axis,
"determine the height of camera in blender"

Second rotation: rotate object around its (this time local) z axis,
"determine the angle of camera in blender"

Translation: translate object, "determine the distance from
camera center to object center in blender"

"""

import numpy as np
import glob
import os
import pandas as pd
import transforms3d
from skimage import io
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d as a3
# import matplotlib.colors as colors
import multiprocessing
import warnings
warnings.filterwarnings("ignore")


def main(fn_read):

    csv_data = pd.read_csv(fn_read)

    # TODO: from *.obj 3D space to blender 3D space
    data = np.array(csv_data)
    xyz = data[:, 1:4]

    # DEBUG
    # xyz = xyz[xyz[:, 0]>-1]
    # xyz = xyz[xyz[:, 2]>0]

    # I don't know:
    # xyz[:, 0] *= -1  # ???
    # xyz[:, [1, 2]] = xyz[:, [2, 1]]

    # TODO: intrinsic parameters
    f_mm = 35
    f_fov = 49.1343 * np.pi / 180
    image_size = 1024

    f = f_pix = image_size / (2 * np.tan(f_fov/2))
    cx = cy = image_size / 2

    intrinsic_mtx = np.array([[f, 0, cx],
                              [0, f, cy],
                             [0, 0, 1]],
                             dtype=float)

    # DEBUG: add radial distortion parameters?

    # TODO: extrinsic parameters
    # R: to move a point cloud, "how far is camera center to object center".
    # Initial camera looks at vector (0, 0, 1), i.e., z axis, since we look through the screen
    R = 8
    translation_vector = np.array([0, 0, R])

    theta = 75  # 75d from "up"
    for phi in np.arange(0, 360, 30):

        # rotate
        rotation_vector1 = (1, 0, 0)
        rotation_angle1 = (180-theta) * np.pi / 180
        rotation_vector2 = (0, -np.sin(theta * np.pi / 180), -np.cos(theta * np.pi / 180))
        rotation_angle2 = (-phi-90) * np.pi / 180  # this is because: rotation between camera and object is relative
        rotation_mtx1 = transforms3d.axangles.axangle2mat(rotation_vector1, rotation_angle1)
        rotation_mtx2 = transforms3d.axangles.axangle2mat(rotation_vector2, rotation_angle2)
        rotation_mtx = rotation_mtx2.dot(rotation_mtx1)

        # extrinsic matrix
        extrinsic_mtx = np.concatenate((rotation_mtx, translation_vector[np.newaxis, :].transpose()), axis=1)

        # TODO: xyz to xy
        xyz_homo = np.concatenate((xyz.transpose(), np.ones((1, xyz.shape[0]))), axis=0)  # homogeneous
        xy_homo = intrinsic_mtx.dot(extrinsic_mtx).dot(xyz_homo)  # homogeneous
        xy = (xy_homo[0:2, :] / xy_homo[2, :])[0:2, :].transpose()

        # TODO: save reprojected image (skimage)
        # filter
        xy = xy[np.array(xy[:, 0] < 1024) * np.array(xy[:, 1] < 1024)]
        xy = xy[np.array(xy[:, 0] >= 0) * np.array(xy[:, 1] >= 0)]

        # if xy.shape[0] != 0:
        #     print(xy.shape)

        # hard mask
        image_hard = np.zeros((image_size, image_size))
        image_hard[np.array(xy[:, 1], dtype=int), np.array(xy[:, 0], dtype=int)] = 1

        # soft mask
        image_soft = np.zeros((image_size, image_size))
        for i in range(xy.shape[0]):
            x, y = xy[i, 0:2]
            x, y = int(x), int(y)
            image_soft[y, x] += 1
        image_soft = image_soft / np.max(image_soft)
        image_soft = np.power(image_soft, 0.3)

        # to 0-255
        image_soft = np.array((image_soft - np.min(image_soft)) / (np.max(image_soft) - np.min(image_soft)) * 255,
                              dtype=int)

        obj_name = os.path.basename(fn_read).split('.')[0]
        # fn_write = file_loc_root + 'image/' + '%s/' % obj_name + '%s_%03d%03d_seg.png' % (obj_name, theta, phi)
        fn_write = fn_read.split('\\')[0].replace('pts', 'image') + '/%s_%03d%03d_seg.png' % (obj_name, theta, phi)
        io.imsave(fn_write, image_soft)


def multi_processing(iter):
    fn_read = fn_read_list[iter]
    main(fn_read)


fn_read_list = glob.glob('C:/PythonProject/MyProject/utils/mesh-generation/pts/*.csv')

if __name__ == '__main__':

    # multiprocessing
    pool = multiprocessing.Pool(processes=6)
    pool.map(multi_processing, range(len(fn_read_list)))
    pool.close()
    pool.join()


# # TODO: save reprojected image (plt)
# # mask soft
# image_mask_soft = np.zeros((image_size,image_size))
# for i in range(xy_nxnynzrgb.shape[0]):
#     x, y = xy_nxnynzrgb[i, 0:2]
#     x, y = int(x), int(y)
#     nx, ny, nz, r, g, b = xy_nxnynzrgb[i, 2:8]
#     image_mask_soft[y-1:y+1, x-1:x+1] += nz
#     # image_mask_soft[y, x] += 1
# image_mask_soft = image_mask_soft / np.max(image_mask_soft)
# image_mask_soft = np.power(image_mask_soft, 0.5)
#
# # save mask soft
# fn_write2 = fn.replace('.csv', '_%03d%03d_soft.png' % (theta, phi))
# plt.imshow(image_mask_soft)
# fig = plt.gcf()
# fig.set_size_inches(7.0/3,7.0/3)   # dpi = 300, output = 700*700 pixels
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# plt.margins(0, 0)
# fig.savefig(fname=fn_write2,
#             format='png', transparent=True, dpi=300, pad_inches=0)
# plt.show()


# # TODO: plot 3d points to check
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=0.3)
# ax.set_xlabel('X')
# ax.set_xlim(-10, 10)
# ax.set_ylabel('Y')
# ax.set_ylim(-10, 10)
# ax.set_zlabel('Z')
# ax.set_zlim(-10, 10)
# plt.show()



