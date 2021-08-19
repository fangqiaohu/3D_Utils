"""This code is for generating unsigned distance field grid from mesh (an *.obj file).
"""
import numpy as np
import time
import trimesh
from skimage import measure
import os
import warnings
warnings.filterwarnings("ignore")


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


def pre_processing(mesh):
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    return mesh


def transform_mesh(mesh):

    mesh = pre_processing(mesh)

    # get R_z and apply
    # sample verts
    verts = trimesh.sample.sample_surface(mesh, 1000000)[0]
    # move verts to 0 center, and scale to a 1*1*1 cube
    center = (np.max(verts, axis=0) + np.min(verts, axis=0)) / 2
    dim = np.max(verts, axis=0) - np.min(verts, axis=0)
    verts = (verts - center) / max(dim)
    # NOTE: from screen 3d to object 3d
    verts[:, [1, 2]] = verts[:, [2, 1]]
    # # select upper part
    # verts = verts[verts[:,2]>0]
    # drop dim z, from 3d verts to 2d verts
    verts = verts[:, 0:2]
    # get z-rotation-only min volume bbox
    transform, rectangle = trimesh.bounds.oriented_bounds_2D(verts)
    rz = np.arccos(transform[0, 0])
    # rz = oriented_bounds_2D(verts)
    R = trimesh.transformations.rotation_matrix(rz+np.pi/2, (0,1,0))
    mesh.apply_transform(R)

    # move verts to 0.5 center, and scale to a 1*1*1 cube (max_dim=0.9, i.e. with padding)
    verts = mesh.vertices
    center = (np.max(verts, axis=0) + np.min(verts, axis=0)) / 2
    dim = np.max(verts, axis=0) - np.min(verts, axis=0)
    verts = 0.9 * (verts - center) / max(dim) + 0.5
    mesh.vertices = verts.copy()

    return mesh


def reduce_faces(mesh, n_faces=10000):
    area_faces = mesh.area_faces
    faces = mesh.faces
    if faces.shape[0] > n_faces:
        threshold = np.flip(np.sort(area_faces))[n_faces]
        faces = faces[area_faces>threshold]
    mesh.faces = faces
    mesh = pre_processing(mesh)
    return mesh


def slice_mesh(mesh):
    mesh = trimesh.intersections.slice_mesh_plane(mesh, (-1, 0, 0), (0.5, 0.5, 0.5))
    mesh = trimesh.intersections.slice_mesh_plane(mesh, (0, 0, -1), (0.5, 0.5, 0.5))
    return mesh


def process(fn_input, fn_output, resolution):

    start_time = time.time()

    fn_output_obj = fn_output
    fn_output_npy = fn_output.replace('.obj', '.npy')

    if os.path.exists(fn_output_obj) and os.path.exists(fn_output_npy):
        print('Failed to process [%s], error: [%s]' % (fn_input, 'Exist'))
        return 0

    # read file
    mesh = trimesh.load_mesh(fn_input)
    # mesh = trimesh.boolean.union(mesh, engine=None)
    # mesh = mesh[0]
    # # print('merged')

    mesh = transform_mesh(mesh)
    n_verts, n_faces = mesh.vertices.shape[0], mesh.faces.shape[0]

    mesh = reduce_faces(mesh, n_faces=20000)

    # create a N*N*N*3 grid centers
    voxel_length = 1 / resolution
    basis = np.linspace(0 + voxel_length / 2, 1 - voxel_length / 2, resolution)
    basis_x = np.tile(basis[:, np.newaxis, np.newaxis], reps=(1, resolution, resolution))[:, :, :, np.newaxis]
    basis_y = np.tile(basis[np.newaxis, :, np.newaxis], reps=(resolution, 1, resolution))[:, :, :, np.newaxis]
    basis_z = np.tile(basis[np.newaxis, np.newaxis, :], reps=(resolution, resolution, 1))[:, :, :, np.newaxis]
    voxel_centers = np.concatenate((basis_x, basis_y, basis_z), axis=3)  # N*N*N centers, each center has 3 coordinates

    # if divide voxel to save RAM?
    IS_DIVIDE = True
    if IS_DIVIDE:
        n_divide = 256
        distance_field = []
        voxel_centers_divided = np.reshape(voxel_centers, (n_divide, -1, 3))
        for voxel_centers_block in voxel_centers_divided:
            # find distances from grid centers to mesh
            '''NOTE: time complexity for this step is O(N*log(M)) N: resolution^3, M: number of faces'''
            closest, distance, triangle_id = trimesh.proximity.closest_point(mesh, voxel_centers_block)  # distance
            # distance = trimesh.proximity.signed_distance(mesh, voxel_centers)  # signed distance
            distance_field.append(distance)
        distance_field = np.array(distance_field)
        distance_field = np.reshape(distance_field, (resolution, resolution, resolution))

    # if symmetry?
    IS_SYM = False
    if IS_SYM:
        mesh = slice_mesh(mesh)
        voxel_centers = voxel_centers[0:resolution//2, :, 0:resolution//2, :]
        voxel_centers = np.reshape(voxel_centers, (resolution ** 3 // 4, 3))

        # find distances from grid centers to mesh
        closest, distance, triangle_id = trimesh.proximity.closest_point(mesh, voxel_centers)  # distance
        # distance = trimesh.proximity.signed_distance(mesh, voxel_centers)  # signed distance

        # from N**3 distances vector to N*N*N distance field
        distance = np.reshape(distance, (resolution//2, resolution, resolution//2))
        distance_field = np.zeros((resolution, resolution, resolution))
        distance_field[0:resolution//2, :, 0:resolution//2] = distance
        distance_field[resolution//2:, :, 0:resolution//2] = np.flip(distance, axis=0)
        distance_field[0:resolution//2, :, resolution//2:] = np.flip(distance, axis=2)
        distance_field[resolution//2:, :, resolution//2:] = np.flip(distance, axis=(0,2))

    if not (IS_DIVIDE or IS_SYM):
        voxel_centers = np.reshape(voxel_centers, (resolution ** 3, 3))
        # find distances from grid centers to mesh
        closest, distance, triangle_id = trimesh.proximity.closest_point(mesh, voxel_centers)  # distance
        # distance = trimesh.proximity.signed_distance(mesh, voxel_centers)  # signed distance
        distance_field = np.reshape(distance, (resolution, resolution, resolution))

    # write distance_field to *.obj file for visualization
    spacing = np.array([voxel_length, voxel_length, voxel_length])
    save_vox_to_obj(volume=distance_field,
                    # level=voxel_length/2,
                    level=voxel_length,
                    spacing=spacing,
                    fn_write=fn_output_obj)

    # write distance_field array to *.npy file
    np.save(fn_output_npy, distance_field)

    elapsed_time = time.time() - start_time
    print('Successfully processed [%s, verts=%s, faces=%s] in [%.3f] seconds.' %(fn_input.split('\\')[-1], n_verts, n_faces,elapsed_time))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='input file name, *.obj')
    parser.add_argument('-o', '--output', type=str, required=True, help='output file name, *.obj')
    parser.add_argument('-r', '--resolution', type=int, required=True, help='resolution of voxel grid')
    args = parser.parse_args()
    fn_input = args.input
    fn_output = args.output
    resolution = args.resolution
    process(fn_input, fn_output, resolution)
