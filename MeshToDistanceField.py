"""This code is for generating unsigned distance field grid from mesh (e.g., *.obj file).
"""
import numpy as np
import time
import glob
import trimesh
from skimage import measure
import multiprocessing
import warnings
warnings.filterwarnings("ignore")


# TODO: change parameters here
# a list of file name
fn_read_list = glob.glob('<PATH-TO-YOUR-FILE>')
# resolution of distance field grid (resolution * resolution * resolution)
resolution = 64
# number of multi-processing works
num_works = 2


def save_obj(verts, faces, fn_write):
    f = open(fn_write, 'w')
    for vert in verts:
        f.write('v {} {} {}\n'.format(vert[0], vert[1], vert[2]))
    for face in faces:
        f.write('f {} {} {}\n'.format(
            face[0] + 1, face[1] + 1, face[2] + 1))
    f.close()


def save_vox_to_obj(volume, level, spacing, fn_write):
    verts, faces, normals, values = measure.marching_cubes_lewiner(volume=volume,
                                                                   level=level,
                                                                   spacing=spacing)
    save_obj(verts, faces, fn_write)


def mesh_to_distance_field(fn_read, resolution):

    try:
        # read file
        mesh = trimesh.load_mesh(fn_read)

        # get verts
        verts = mesh.vertices

        # move verts to 0.5 center, and scale to 1*1*1 cube (max_dim=0.9, i.e. with padding)
        center = (np.max(verts, axis=0) + np.min(verts, axis=0)) / 2
        dim = np.max(verts, axis=0) - np.min(verts, axis=0)
        max_dim = max(dim)
        verts = verts - center
        verts = 0.9 * verts / max_dim
        verts = verts + 0.5
        mesh.vertices = verts

        # create a N*N*N*3 grid centers
        voxel_length = 1 / resolution
        basis = np.linspace(0 + voxel_length / 2, 1 - voxel_length / 2, resolution)
        basis_x = np.tile(basis[:, np.newaxis, np.newaxis], reps=(1, resolution, resolution))[:, :, :, np.newaxis]
        basis_y = np.tile(basis[np.newaxis, :, np.newaxis], reps=(resolution, 1, resolution))[:, :, :, np.newaxis]
        basis_z = np.tile(basis[np.newaxis, np.newaxis, :], reps=(resolution, resolution, 1))[:, :, :, np.newaxis]
        voxel_centers = np.concatenate((basis_x, basis_y, basis_z), axis=3)  # N*N*N centers, each center has 3 coordinates
        voxel_centers = np.reshape(voxel_centers, (resolution ** 3, 3))

        # find distances from grid centers to mesh
        '''NOTE: time complexity for this step is O(N*log(M)) N: resolution^3, M: number of faces'''
        start_time = time.time()
        closest, distance, triangle_id = trimesh.proximity.closest_point(mesh, voxel_centers)  # distance
        # distance = trimesh.proximity.signed_distance(mesh, voxel_centers)  # signed distance
        elapsed_time = time.time() - start_time

        # from N**3 distances vector to N*N*N distance field
        distance_field = np.reshape(distance, (resolution, resolution, resolution))

        # # we can easily obtain a N*N*N binary voxel grid from distance field
        # voxels = distance_field <= 0.707*voxel_length

        # write distance_field to *.obj file for visualization
        fn_write = fn_read.replace(fn_read.split('.')[0], fn_read.split('.')[0] + '_df')
        spacing = np.array([voxel_length, voxel_length, voxel_length])
        save_vox_to_obj(volume=distance_field,
                        level=voxel_length,
                        spacing=spacing,
                        fn_write=fn_write)

        # write distance_field array to *.npy file
        fn_write = fn_write.replace('.obj', '.npy')
        np.save(fn_write, distance_field)

        print('Successfully processed [%s], distance calculation takes [%.3f] seconds.' %(fn_read.split('\\')[-1], elapsed_time))

    except Exception as e:
        print('Failed to process [%s]. Error: [%s].' % (fn_read.split('\\')[-1], e))
        return 0


def multi_processing(iter):
    fn_read = fn_read_list[iter]
    mesh_to_distance_field(fn_read, resolution)


# multiprocessing
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=num_works)
    pool.map(multi_processing, range(len(fn_read_list)))
    pool.close()
    pool.join()

