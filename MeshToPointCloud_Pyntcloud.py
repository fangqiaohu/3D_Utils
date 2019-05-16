"""
This code can generate point cloud from mesh in *.obj format, without label.
"""

import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import glob
import multiprocessing


def random(a, b):
    c = a + (b-a)*np.random.random()
    return c


def main(fn_read):

    fn_write = fn_read.replace('obj_bridge', 'pts')

    # load obj
    obj = PyntCloud.from_file(fn_read)

    # sample points
    points = obj.get_sample('mesh_random', as_PyntCloud=True, n=100000, rgb=False, normals=False)

    # TODO: add noise
    # add noise (self)
    points_after_noise = points.xyz + random(0.01, 0.025) * np.random.randn(points.xyz.shape[0], points.xyz.shape[1])
    # add noise (surroundings)
    points_after_noise = np.concatenate((points_after_noise, random(3, 5)*np.random.randn(points.xyz.shape[0], points.xyz.shape[1])))

    # filter
    criterion = np.array(points_after_noise[:, 0] <= (np.max(points.xyz, axis=0)[0])) \
                * np.array(points_after_noise[:, 1] <= (np.max(points.xyz, axis=0)[1])*random(1.1, 1.3)) \
                * np.array(points_after_noise[:, 2] <= (np.max(points.xyz, axis=0)[2])*random(1.3, 1.5)) \
                * np.array(points_after_noise[:, 0] >= (np.min(points.xyz, axis=0)[0])) \
                * np.array(points_after_noise[:, 1] >= (np.min(points.xyz, axis=0)[1])*random(1.1, 1.3)) \
                * np.array(points_after_noise[:, 2] >= (np.min(points.xyz, axis=0)[2])*random(1.3, 1.5))
    points_after_noise = points_after_noise[criterion]

    # TODO: uneven distribution

    # exchange yz
    points_after_noise[:, [1, 2]] = points_after_noise[:, [2, 1]]

    # apply points
    points = PyntCloud(points=pd.DataFrame(points_after_noise, columns=['x', 'y', 'z']))

    # save *.obj
    points.to_file(fn_write)

    # save *.csv
    points.to_file(fn_write.replace('.obj', '.csv'))

    # # dataframe to numpy array
    # points = obj.get_sample('mesh_random', as_PyntCloud=False, n=10000, rgb=False, normals=False)
    # points = points.as_matrix(columns=None)

    # # dataframe to csv
    # points = obj.get_sample('mesh_random', as_PyntCloud=False, n=10000, rgb=False, normals=False)
    # points.to_file(fn_write.replace('.obj', '.csv'))


def multi_processing(iter):
    fn_read = fn_read_list[iter]
    main(fn_read)


fn_read_list = glob.glob('C:/PythonProject/MyProject/utils/mesh-generation/obj_bridge/*.obj')

if __name__ == '__main__':

    # multiprocessing
    pool = multiprocessing.Pool(processes=6)
    pool.map(multi_processing, range(len(fn_read_list)))
    pool.close()
    pool.join()

    # without multiprocessing
    # for i in fn_read_list:
    #     main(fn_read=i)


