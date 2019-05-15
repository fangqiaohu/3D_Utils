"""
This code read segmented and NOT ROTATED (NOT ALIGNED) 3D points and camera poses as input,
they are in *.npy files, and contains 'fused-filtered-3.npy' (dense 3D points),
'sparse/cameras_intrinsic.npy' and 'sparse/cameras_extrinsic.npy' (camera parameters).
The out put is re-projected point cloud image.
The images are used as segmented images.
"""


import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from skimage import io, transform
import os
import warnings
warnings.filterwarnings("ignore")

file_locate_root = 'pcd2'
# file = 'points3D.npy'
file = 'fused-filtered-2.npy'
# file = 'pcd2_bbox.npy'
show_which = 120
point_size = 3

fn = os.path.join(file_locate_root, 'dense/'+file)
points3D_xyzrgb = np.load(fn)

points3D = np.array(points3D_xyzrgb[:, 0:3], dtype=float)
rgb = np.array(points3D_xyzrgb[:, 3:6], dtype=float)

cameras_intrinsic = np.load(os.path.join(file_locate_root, 'sparse/cameras_intrinsic.npy'))
cameras_extrinsic = np.load(os.path.join(file_locate_root, 'sparse/cameras_extrinsic.npy'))

if cameras_intrinsic.shape[0]!=1:
    cameras_intrinsic = cameras_intrinsic[show_which]
elif cameras_intrinsic.shape[0]==1:
    cameras_intrinsic = cameras_intrinsic[0]
CAMERA_ID, MODEL, WIDTH, HEIGHT = cameras_intrinsic[0:4]
PARAMS = cameras_intrinsic[4:]

IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME = cameras_extrinsic[show_which]

resolution = np.array([HEIGHT, WIDTH], dtype=int)

# intrinsic
f, cx, cy, k = np.array(PARAMS, dtype=float)
intrinsic_mtx = np.array([[f, 0, cx],
                          [0, f, cy],
                         [0, 0, 1]],
                         dtype=float)

# extrinsic
# quaternion for rotation
quaternion = np.array([QW, QX, QY, QZ], dtype=float)
quaternion = Quaternion(quaternion)
rotation_mtx = quaternion.rotation_matrix
# translation vector
translation_vec = np.array([[TX, TY, TZ]], dtype=float).transpose()
extrinsic_mtx = np.concatenate((rotation_mtx, translation_vec), axis=1)


# get camera center
camera_center = -np.dot(rotation_mtx.transpose(), translation_vec)


# get points2D
XYZ = np.concatenate((points3D.transpose(), np.ones((1,points3D.shape[0]))), axis=0)
xy = intrinsic_mtx.dot(extrinsic_mtx).dot(XYZ)

xy[0, :] = xy[0, :] / xy[2, :]
xy[1, :] = xy[1, :] / xy[2, :]

points2D = xy[0:2, :]
points2D = points2D.transpose()

NAME = str(NAME)
print(IMAGE_ID, NAME)


# # TODO: radial distortion matrix using k
# print('radial distortion factor:', k)
# rr = ((points2D[:, 0]-cx) / resolution[1]) ** 2 + ((points2D[:, 1]-cy) / resolution[1]) ** 2
# rr = np.tile(rr, reps=(2, 1)).transpose()
# points2D = (1 + k * rr) * points2D


# TODO: save image
image = io.imread(os.path.join(file_locate_root, 'img/' + NAME))
if image.shape[0] == 2:
    image = image[0]
image = transform.resize(image, resolution)
image = image / 2

points2D = np.array(np.around(points2D), dtype=int)

# filter out points that not in image
points2D = points2D[np.array(points2D[:, 0]<resolution[1]) * np.array(points2D[:, 1]<resolution[0])]
points2D = points2D[np.array(points2D[:, 0]>=0) * np.array(points2D[:, 1]>=0)]
points2D = np.array(points2D, dtype=int)

print(np.max(points2D, axis=0), np.min(points2D, axis=0))
print(resolution[1], resolution[0])

# save overlay image
image[points2D[:, 1], points2D[:, 0], 0] = 0.75
io.imsave(fname=os.path.join(file_locate_root, 'dense/result_%s' % NAME), arr=image)

# mask hard
image_mask_hard = np.zeros((resolution[0], resolution[1]))
image_mask_hard[points2D[:, 1], points2D[:, 0]] = 1
print('foreground / all:', np.sum(image_mask_hard), resolution[0]*resolution[1])

# mask soft
image_mask_soft = np.zeros((resolution[0], resolution[1]))
for i in range(points2D.shape[0]):
    x, y = points2D[i]
    rgb_current = rgb[i]
    image_mask_soft[y-2:y+2, x-2:x+2] += 1
    # image_mask_soft[y, x] += 1
image_mask_soft = image_mask_soft / np.max(image_mask_soft)
image_mask_soft = np.power(image_mask_soft, 0.3)

# # debug: save raw image
# image_mask_soft = io.imread(os.path.join(file_locate_root, 'img/' + NAME))
# if image_mask_soft.shape[0] == 2:
#     image_mask_soft = image_mask_soft[0]

# save mask soft (skimage)
io.imsave(fname=os.path.join(file_locate_root, 'dense/result_%s' % NAME.split('.')[0]+'.png'), arr=image_mask_soft)

# # save mask soft
# plt.imshow(image_mask_soft, cmap='gray')
# fig = plt.gcf()
# # fig.set_size_inches(7.0/3,7.0/3)   # dpi = 300, output = 700*700 pixels
# # plt.gca().xaxis.set_major_locator(plt.NullLocator())
# # plt.gca().yaxis.set_major_locator(plt.NullLocator())
# # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
# # plt.margins(0,0)
# fig.savefig(fname=os.path.join(file_locate_root, 'dense/result_%s' % NAME.split('.')[0]+'.png'),
#             format='png', transparent=False, dpi=300, pad_inches=0)
# plt.show()




