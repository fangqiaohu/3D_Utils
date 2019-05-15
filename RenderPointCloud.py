"""
This file is for generating training data.
It generates re-projected 3D point image, which can be normal map or rgb image.
"""


"""NOTE:
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
import warnings
warnings.filterwarnings("ignore")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors

file_loc_root = '<ROOT>'
fn = '<FILEPATH>'

csv_data = pd.read_csv(fn)

# TODO: from *.obj 3D space to blender 3D space
data = np.array(csv_data)
xyz = data[:, 1:4]

# debug!!!
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
phi = 30

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
xy_nxnynzrgb = np.concatenate((xy, data[:, 4:10]), axis=1)


# TODO: save reprojected image (skimage)
image = np.zeros((image_size, image_size))

# filter
xy_nxnynzrgb = xy_nxnynzrgb[np.array(xy_nxnynzrgb[:, 0]<1024) * np.array(xy_nxnynzrgb[:, 1]<1024)]
xy_nxnynzrgb = xy_nxnynzrgb[np.array(xy_nxnynzrgb[:, 0]>=0) * np.array(xy_nxnynzrgb[:, 1]>=0)]
# xy_nxnynzrgb = np.array(xy_nxnynzrgb, dtype=int)

if xy_nxnynzrgb.shape[0] != 0:
    print(xy_nxnynzrgb.shape)

image[np.array(xy_nxnynzrgb[:, 1], dtype=int), np.array(xy_nxnynzrgb[:, 0], dtype=int)] = 1

# io.imsave(fn.replace('.csv', '_%03d_%03d_%03d.png' % tuple(rotation_euler * 180 / np.pi)), image)
# io.imsave(fn.replace('.csv', '.png'), image)
fn_write = fn.replace('.csv', '_%03d%03d.png' % (theta, phi))
io.imsave(fn_write, image)


# TODO: save reprojected image (plt)
# mask soft
image_mask_soft = np.zeros((image_size,image_size))
for i in range(xy_nxnynzrgb.shape[0]):
    x, y = xy_nxnynzrgb[i, 0:2]
    x, y = int(x), int(y)
    nx, ny, nz, r, g, b = xy_nxnynzrgb[i, 2:8]
    image_mask_soft[y-1:y+1, x-1:x+1] += nz
    # image_mask_soft[y, x] += 1
image_mask_soft = image_mask_soft / np.max(image_mask_soft)
image_mask_soft = np.power(image_mask_soft, 0.5)

# save mask soft
fn_write2 = fn.replace('.csv', '_%03d%03d_soft.png' % (theta, phi))
plt.imshow(image_mask_soft)
fig = plt.gcf()
fig.set_size_inches(7.0/3,7.0/3)   # dpi = 300, output = 700*700 pixels
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
fig.savefig(fname=fn_write2,
            format='png', transparent=True, dpi=300, pad_inches=0)
plt.show()


# # TODO: save reprojected image (skimage)
# image = np.zeros((image_size, image_size))
#
# xy = xy[np.array(xy[:, 0]<1024) * np.array(xy[:, 1]<1024)]
# xy = xy[np.array(xy[:, 0]>=0) * np.array(xy[:, 1]>=0)]
# xy = np.array(xy, dtype=int)
#
# if xy.shape[0] != 0:
#     print(xy.shape)
#
# image[xy[:, 1], xy[:, 0]] = 1
#
# # io.imsave(fn.replace('.csv', '_%03d_%03d_%03d.png' % tuple(rotation_euler * 180 / np.pi)), image)
# # io.imsave(fn.replace('.csv', '.png'), image)
# fn_write = fn.replace('.csv', '_%03d%03d.png' % (theta, phi))
# io.imsave(fn_write, image)
#
#
# # TODO: save reprojected image (plt)
# # mask soft
# image_mask_soft = np.zeros((image_size,image_size))
# for i in range(xy.shape[0]):
#     x, y = xy[i]
#     image_mask_soft[y-1:y+1, x-1:x+1] += 1
#     # image_mask_soft[y, x] += 1
# image_mask_soft = image_mask_soft / np.max(image_mask_soft)
# image_mask_soft = np.power(image_mask_soft, 0.3)
#
# # save mask soft
# fn_write2 = fn.replace('.csv', '_%03d%03d_soft.png' % (theta, phi))
# plt.imshow(image_mask_soft)
# fig = plt.gcf()
# # fig.set_size_inches(7.0/3,7.0/3)   # dpi = 300, output = 700*700 pixels
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



