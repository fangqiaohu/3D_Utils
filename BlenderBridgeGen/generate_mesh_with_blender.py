import bpy
import numpy as np
import os
import glob


def reset_blend():
    """reset blender"""

    # bpy.ops.wm.read_factory_settings()

    for scene in bpy.data.scenes:
        for obj in scene.objects:
            scene.objects.unlink(obj)

    # only worry about data in the startup scene
    for bpy_data_iter in (
            bpy.data.objects,
            bpy.data.meshes,
            bpy.data.lamps,
            bpy.data.cameras,
    ):
        for id_data in bpy_data_iter:
            bpy_data_iter.remove(id_data)


def draw_cube(p1, p2, width, height):
    """
    Draw a cube using given end point 1, end point 2, w, h. 8 params in total.
    :param p1, p2: end points of a line
    :param width, height: width and height of section
    """
    dx, dy, dz = p2 - p1
    length = np.sqrt(dx**2 + dy**2 + dz**2)

    scale = (height, width, length)
    location = (p1 + p2) / 2

    bpy.ops.mesh.primitive_cube_add(radius=0.5, location=location)
    # sets the value directly in the properties panel, absolute method
    bpy.context.scene.objects.active.scale = scale
    # # scale object the same way as pressing S in the 3D view, relative method
    # bpy.ops.transform.resize(value=scale)

    phi = np.arctan2(dy, dx)
    theta = np.arccos(dz/length)

    bpy.context.object.rotation_euler[1] = theta
    bpy.context.object.rotation_euler[2] = phi


def draw_bbox(center, dim):
    """
    Draw a cube using center, dimension. 6 params in total.
    :param center, dim: center and dim
    """
    bpy.ops.mesh.primitive_cube_add(radius=0.5, location=center)
    bpy.context.scene.objects.active.scale = dim


def draw_cylinder(p1, p2, radius):
    """
    Draw a cylinder using given end point 1, end point 2, radius. 7 params in total.
    :param p1, p2: end points of a line
    :param radius: radius of cylinder
    """
    dx, dy, dz = p2 - p1
    length = np.sqrt(dx**2 + dy**2 + dz**2)
    location = (p1 + p2) / 2

    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length, location=location)

    phi = np.arctan2(dy, dx)
    theta = np.arccos(dz/length)

    bpy.context.object.rotation_euler[1] = theta
    bpy.context.object.rotation_euler[2] = phi


def draw_cube_2(p1, p2, radius):
    """
    Draw a cube using given end point 1, end point 2, radius. 7 params in total.
    :param p1, p2: end points of a line
    :param radius: radius for square section
    """
    # radius: radius of section
    dx, dy, dz = p2 - p1
    length = np.sqrt(dx**2 + dy**2 + dz**2)

    scale = (radius, radius, length)
    location = (p1 + p2) / 2

    bpy.ops.mesh.primitive_cube_add(radius=0.5, location=location)
    # sets the value directly in the properties panel, absolute method
    bpy.context.scene.objects.active.scale = scale
    # # scale object the same way as pressing S in the 3D view, relative method
    # bpy.ops.transform.resize(value=scale)

    phi = np.arctan2(dy, dx)
    theta = np.arccos(dz/length)

    bpy.context.object.rotation_euler[1] = theta
    bpy.context.object.rotation_euler[2] = phi


def random(a, b):
    c = a + (b-a)*np.random.random()
    return c


def draw_tower(fn_tower, num_tower, center=None, color=None):
    """
    :param fn_tower: string, for importing the tower object
    :param num_tower: int in {1, 2, 3}
    :param center: triple, for num_tower=2 or 3, center of the bbox
    :param color: triple, (R, G, B)
    :return:
    """
    if num_tower not in [1, 2, 3]:
        raise ValueError

    # import obj file
    # imported_object = bpy.ops.import_scene.obj(filepath=fn_tower)
    bpy.ops.import_scene.obj(filepath=fn_tower)
    bpy.context.scene.objects.active = bpy.context.selected_objects[0]
    mat = bpy.data.materials.new(name="tower")  # set new material to variable
    bpy.context.active_object.data.materials.append(mat)  # add the material to the object
    bpy.context.active_object.active_material.diffuse_color = color

    # # move all objects to center (BUG: need to merge these objs before move to center)
    # bpy.ops.object.select_all(action='SELECT')
    # bpy.ops.object.origin_set(type="GEOMETRY_ORIGIN")

    if num_tower == 1:
        label = 'tower_side=center'
        bpy.context.active_object.name = label

    elif num_tower == 2:

        bpy.context.active_object.location.xyz = center
        label = 'tower_side=left'
        bpy.context.active_object.name = label

        # duplicate and mirror to get another tower
        bpy.ops.object.duplicate()
        bpy.context.active_object.location.xyz = np.array([-center[0], center[1], center[2]])
        label = 'tower_side=right'
        bpy.context.active_object.name = label

    elif num_tower == 3:

        label = 'tower_side=center'
        bpy.context.active_object.name = label

        # duplicate and mirror to get another tower
        bpy.ops.object.duplicate()
        bpy.context.active_object.location.xyz = center
        label = 'tower_side=left'
        bpy.context.active_object.name = label

        # duplicate and mirror to get another tower
        bpy.ops.object.duplicate()
        bpy.context.active_object.location.xyz = np.array([-center[0], center[1], center[2]])
        label = 'tower_side=right'
        bpy.context.active_object.name = label


def draw_road(center_girder, dim_girder, color=None):
    """
    :param center_girder: triple, center of the bbox
    :param dim_girder: triple, size of the bbox
    :param color: triple, (R, G, B)
    """
    draw_bbox(center=center_girder, dim=dim_girder)
    # TODO: add label


def get_keypoints(center, dim, num_block):
    """
    :param center: triple, center of the bbox of girder
    :param dim: triple, size of the bbox of girder
    :param num_block: int, number of blocks
    """
    # 4 key-points in one repeat, all x=0
    vert = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1],
                     [0, 1, 1]])
    # TODO: translations are not necessary equal from one to another, add random or?
    trans = np.arange(num_block + 1).repeat(4)
    verts = np.tile(vert, reps=(num_block + 1, 1))
    # all key-points
    verts[:, 0] += trans
    # fit key-points to bbox
    def fit_bbox(verts, center, dim):
        center2 = (np.max(verts, axis=0) + np.min(verts, axis=0)) / 2
        dim2 = np.max(verts, axis=0) - np.min(verts, axis=0)
        verts = verts - center2
        verts = verts / dim2 * dim
        verts = verts + center
        return verts
    verts = fit_bbox(verts, center, dim)
    return verts


def draw_truss(center, dim, num_block, case=1, sym_plane=None, color=None):
    """
    :param center: triple, center of the bbox of girder
    :param dim: triple, size of the bbox of girder
    :param num_block: int, number of blocks
    :param case: {0, 1, 2, 3}, cases for slant lines
    :param sym_plane: float, for case==2, determine the sym_plane_1 on "x axis"
    :param color: triple, (R, G, B)
    """
    if case not in [0, 1, 2, 3]:
        raise ValueError

    # get keypoints
    verts = get_keypoints(center, dim, num_block*2)  # each block has 3*4=12 keypoints

    def rep_lines(line, num_reps, interval=8):
        """Since 4 key-points in one repeat, given a group of lines, it returns its repeats."""
        _reps = np.arange(num_reps).repeat(line.shape[0]) * interval
        _reps = np.tile(_reps, reps=(2, 1)).T
        lines = np.tile(line, reps=(num_reps, 1)) + _reps
        return lines

    # horizontal lines that need n repeats
    line = np.array([[0, 8],
                     [1, 9],
                     [2, 10],
                     [3, 11]])
    lines = rep_lines(line, num_block)

    # horizontal lines that need n+1 repeats
    line = np.array([[0, 1],
                     [2, 3]])
    lines = np.concatenate((lines, rep_lines(line, num_block+1)))

    # vertical lines that need n+1 repeats, optional
    if 1:
        line_vertical = np.array([[0, 2],
                                  [1, 3]])
        lines = np.concatenate((lines, rep_lines(line_vertical, num_block+1)))

    # slant lines that need n repeats
    line_slant_1 = np.array([[0, 10],
                             [1, 11]])
    line_slant_2 = np.array([[2, 8],
                             [3, 9]])

    # slant lines that need 2n repeats
    line_slant_3 = np.array([[0, 6],
                             [1, 7]])
    line_slant_4 = np.array([[2, 4],
                             [3, 5]])

    # if exchange two slant lines
    if 1:
        line_slant_1, line_slant_2 = line_slant_2, line_slant_1
        line_slant_3, line_slant_4 = line_slant_4, line_slant_3

    # case0: two adjacent lines are symmetrical; each block has one slant line
    if case == 0:
        line_slant = np.concatenate((line_slant_1, line_slant_2 + 8))
        lines_slant = rep_lines(line_slant, num_reps=num_block//2, interval=16)

    # case1: two adjacent lines are symmetrical; each block has two slant lines
    elif case == 1:
        line_slant = np.concatenate((line_slant_3, line_slant_4 + 4))
        lines_slant = rep_lines(line_slant, num_reps=num_block, interval=8)

    # case2: 1 symmetrical plane: original point
    elif case == 2:
        _sym_plane = num_block // 2
        lines_slant_1 = rep_lines(line_slant_1, num_block)
        lines_slant_2 = rep_lines(line_slant_2, num_block)
        lines_slant = np.zeros((2 * num_block, 2))
        lines_slant[0:2*_sym_plane] = lines_slant_1[0:2*_sym_plane]
        lines_slant[2*_sym_plane:2*num_block] = lines_slant_2[2*_sym_plane:2*num_block]

    # case3: 3 symmetrical plane: original point and tower
    else:
        sym_plane_1 = int(round(abs(sym_plane - (center[0] - dim[0] / 2)) / (dim[0] / num_block)))
        # sym_plane_1 = num_block // 4
        sym_plane_2 = num_block // 2
        sym_plane_3 = num_block - sym_plane_1
        lines_slant_1 = rep_lines(line_slant_1, num_block)
        lines_slant_2 = rep_lines(line_slant_2, num_block)
        lines_slant = np.zeros((2 * num_block, 2))
        lines_slant[0:2*sym_plane_1] = lines_slant_1[0:2*sym_plane_1]
        lines_slant[2*sym_plane_1:2*sym_plane_2] = lines_slant_2[2*sym_plane_1:2*sym_plane_2]
        lines_slant[2*sym_plane_2:2*sym_plane_3] = lines_slant_1[2*sym_plane_2:2*sym_plane_3]
        lines_slant[2*sym_plane_3:2*num_block] = lines_slant_2[2*sym_plane_3:2*num_block]

    # append slant lines to all lines
    lines = np.concatenate((lines, lines_slant))
    for line in lines:
        draw_cube(p1=verts[line[0]], p2=verts[line[1]], width=0.01, height=0.01)

    # draw road
    draw_bbox(center=(center[0], center[1], center[2]+0.6*dim[2]), dim=(dim[0], dim[1], 0.05*dim[2]))


def get_lines_from_kps(kps, n):
    """
    :param kps: (x*4) * 3 keypoints of cables, one side needs 4 * 3 keypoints, x sides in total
    :param n: number of cables for one side
    :return: np.array, shape=(x*n)*6
    """
    x = kps.shape[0] // 4
    lines = np.zeros((x * n, 6))
    for i in range(x):
        start1, stop1 = kps[4 * i + 0], kps[4 * i + 1]
        start2, stop2 = kps[4 * i + 2], kps[4 * i + 3]
        inter1 = (stop1 - start1) / (n - 1)
        inter2 = (stop2 - start2) / (n - 1)
        for j in range(n):
            lines[n * i + j, 0:3] = start1 + j * inter1
            lines[n * i + j, 3:6] = start2 + j * inter2
    return lines


def draw_cable_truss(num_tower, center_girder, dim_girder, num_block, blanks=(1, 1, 2), num_side=2,
                     shift_from_tower_center=None, center_tower=None, radius=0.005, color=None):
    """
    :param num_tower: int in {1, 2, 3}
    :param center_girder: triple, center of the bbox of girder
    :param dim_girder: triple, size of the bbox of girder
    :param num_block: int, number of blocks
    :param blanks: int triple, number of blanks, control the keypoints of cables
            (endpoint to cable, cable to tower, cable to cable)
    :param num_side: int in {1, 2, 3}, number of cable sides
    :param shift_from_tower_center: 2*3 float, label for the connection of tower and cable
    :param center_tower: float, for the center of the first tower (if 2 or 3)
    :param radius: radius of cable
    :param color: triple, (R, G, B)
    """
    blank_1, blank_2, blank_3 = blanks  # (endpoint to cable, cable to tower, cable to cable)
    # get keypoints
    verts = get_keypoints(center_girder, dim_girder, num_block)
    verts_front = verts[2::4]  # keypoints on truss in the front side

    if num_tower not in [1, 2, 3] or num_side not in [1, 2, 3]:
        raise ValueError

    if num_tower==1:
        center_tower = np.array([0, 0, 0])  # center_tower at origin by default
        # control the number of cables
        num_cable = num_block//2 - blank_1 - blank_2 + 1

        # find 4 keypoints for one side of cables: (kps_x)
        # left and front side
        line_start = np.array([center_tower + shift_from_tower_center[0],
                               center_tower + shift_from_tower_center[1]])
        line_end = np.array([verts_front[blank_1], verts_front[num_cable + blank_1 - 1]])
        kps_1 = np.concatenate((line_start, line_end), axis=0)

        # lines according to keypoints
        # left and front side
        lines_1 = get_lines_from_kps(kps_1, n=num_cable)
        # right and front side
        lines_2 = lines_1.copy()
        lines_2[:, [0, 3]] *= -1
        # front side
        lines_front = np.concatenate((lines_1, lines_2), axis=0)

    elif num_tower==2:
        which_block = int(round(abs(center_tower[0] - (center_girder[0] - dim_girder[0] / 2)) / (dim_girder[0] / num_block)))
        num_cable_1 = which_block - blank_1 - blank_2 + 1
        num_cable_2 = num_block//2 - which_block - blank_3//2 - blank_2 + 1

        # find 4 keypoints for one side of cables: (kps_x)
        # left and front side (left part only)
        line_start_1 = np.array([center_tower + shift_from_tower_center[0],
                               center_tower + shift_from_tower_center[1]])
        line_end_1 = np.array([verts_front[blank_1], verts_front[num_cable_1 + blank_1 - 1]])
        kps_1 = np.concatenate((line_start_1, line_end_1), axis=0)
        # right and front side (left part only)
        _shift_from_tower_center = shift_from_tower_center.copy()
        _shift_from_tower_center[:, 0] *= -1
        line_start_2 = np.array([center_tower + _shift_from_tower_center[0],
                               center_tower + _shift_from_tower_center[1]])
        line_end_2 = np.array([verts_front[which_block+blank_2+num_cable_2-1], verts_front[which_block+blank_2]])
        kps_2 = np.concatenate((line_start_2, line_end_2), axis=0)

        # lines according to keypoints
        # left and front side (left part only)
        lines_1 = get_lines_from_kps(kps_1, n=num_cable_1)
        # right and front side (left part only)
        lines_2 = get_lines_from_kps(kps_2, n=num_cable_2)
        # front side (left part only)
        lines_3 = np.concatenate((lines_1, lines_2), axis=0)
        # front side (right part only)
        lines_4 = lines_3.copy()
        lines_4[:, [0, 3]] *= -1
        # front side
        lines_front = np.concatenate((lines_3, lines_4), axis=0)

    else:
        # which block is the tower (the first symmetry plane)
        which_block_1 = int(round(abs(center_tower[0] - (center_girder[0] - dim_girder[0] / 2)) / (dim_girder[0] / num_block)))
        # which block is the second symmetry plane
        which_block_2 = int(round(abs(center_tower[0] / 2 - (center_girder[0] - dim_girder[0] / 2)) / (dim_girder[0] / num_block)))
        num_cable_1 = which_block_1 - blank_1 - blank_2 + 1
        num_cable_2 = num_block//2 - which_block_2 - blank_3 - blank_2 + 1

        # find 4 keypoints for one side of cables: (kps_x)
        # left and front side (left part only)
        line_start_1 = np.array([center_tower + shift_from_tower_center[0],
                               center_tower + shift_from_tower_center[1]])
        line_end_1 = np.array([verts_front[blank_1], verts_front[num_cable_1 + blank_1 - 1]])
        kps_1 = np.concatenate((line_start_1, line_end_1), axis=0)
        # right and front side (left part only)
        _shift_from_tower_center = shift_from_tower_center.copy()
        _shift_from_tower_center[:, 0] *= -1
        line_start_2 = np.array([center_tower + _shift_from_tower_center[0],
                               center_tower + _shift_from_tower_center[1]])
        line_end_2 = np.array([verts_front[which_block_1+blank_2+num_cable_2-1], verts_front[which_block_1+blank_2]])
        kps_2 = np.concatenate((line_start_2, line_end_2), axis=0)
        # left and front side (center part only)
        line_start_3 = np.array([shift_from_tower_center[0], shift_from_tower_center[1]])
        line_end_3 = np.array([verts_front[which_block_2+blank_3-1], verts_front[which_block_2+blank_3+num_cable_2-1]])
        kps_3 = np.concatenate((line_start_3, line_end_3), axis=0)

        # lines according to keypoints
        # left and front side (left part only)
        lines_1 = get_lines_from_kps(kps_1, n=num_cable_1)
        # right and front side (left part only)
        lines_2 = get_lines_from_kps(kps_2, n=num_cable_2)
        # left and front side (center part only)
        lines_3 = get_lines_from_kps(kps_3, n=num_cable_2+1)
        # front side (left + 1/2 center part only)
        lines_4 = np.concatenate((lines_1, lines_2, lines_3), axis=0)
        # front side (right + 1/2 center part only)
        lines_5 = lines_4.copy()
        lines_5[:, [0, 3]] *= -1
        # front side
        lines_front = np.concatenate((lines_4, lines_5), axis=0)

    # back side (use front side)
    lines_back = lines_front.copy()
    lines_back[:, [1, 4]] *= -1
    # center side
    lines_center = (lines_front + lines_back) / 2

    # side number
    if num_side == 3:
        lines = np.concatenate((lines_front, lines_back, lines_center), axis=0)
    elif num_side == 2:
        lines = np.concatenate((lines_front, lines_back), axis=0)
    else:
        lines = lines_center

    # draw lines
    for line in lines:
        p1, p2 = line[0:3], line[3:6]
        draw_cube_2(p1, p2, radius=radius)


def draw_cable_road(num_tower, center_girder, dim_girder, blanks=(0.2, 0.2, 0.2), interval=0.1, num_side=2,
                     shift_from_tower_center=None, center_tower=None, radius=0.005, color=None):
    """
    :param num_tower: int in {1, 2, 3}
    :param center_girder: triple, center of the bbox of girder
    :param dim_girder: triple, size of the bbox of girder
    :param blanks: float triple, distance of blanks, control the keypoints of cables
            (endpoint to cable, cable to tower, cable to cable)
    :param interval: float, an **approximate** interval for two adjacent cables
    :param num_side: int in {1, 2, 3}, number of cable sides
    :param shift_from_tower_center: 2*3 float, label for the connection of tower and cable
    :param center_tower: float, for the center of the first tower (if 2 or 3)
    :param radius: radius of cable
    :param color: triple, (R, G, B)
    """
    # (endpoint to cable, cable to tower, cable to cable)
    blank_1, blank_2, blank_3 = blanks
    blank_3 = blank_3 / 2  # for easier calculation
    # front left up keypoint on girder
    vert_flu = np.array([-dim_girder[0]/2, -dim_girder[1]/2, dim_girder[2]/2])

    if num_tower not in [1, 2, 3] or num_side not in [1, 2, 3]:
        raise ValueError

    if num_tower==1:
        center_tower = np.array([0, 0, 0])  # center_tower at origin by default
        # control the number of cables
        num_cable = int(round((dim_girder[0] / 2 - blank_1 - blank_2) / interval))

        # find 4 keypoints for one side of cables: (kps_x)
        # left and front side
        line_start = np.array([center_tower + shift_from_tower_center[0],
                               center_tower + shift_from_tower_center[1]])
        line_end = np.array([[-dim_girder[0]/2+blank_1, -dim_girder[1]/2, dim_girder[2]/2] + center_girder,
                             [-blank_2, -dim_girder[1]/2, dim_girder[2]/2] + center_girder])
        kps_1 = np.concatenate((line_start, line_end), axis=0)

        # lines according to keypoints
        # left and front side
        lines_1 = get_lines_from_kps(kps_1, n=num_cable)
        # right and front side
        lines_2 = lines_1.copy()
        lines_2[:, [0, 3]] *= -1
        # front side
        lines_front = np.concatenate((lines_1, lines_2), axis=0)

    elif num_tower==2:
        # control the number of cables
        num_cable_1 = int(round((dim_girder[0] / 2 - abs(center_tower[0]) - blank_1 - blank_2) / interval))
        num_cable_2 = int(round((abs(center_tower[0]) - blank_2 - blank_3 / 2) / interval))

        # find 4 keypoints for one side of cables: (kps_x)
        # left and front side
        line_start_1 = np.array([center_tower + shift_from_tower_center[0],
                               center_tower + shift_from_tower_center[1]])
        line_end_1 = np.array([[-dim_girder[0]/2+blank_1, -dim_girder[1]/2, dim_girder[2]/2] + center_girder,
                             [center_tower[0]-blank_2, -dim_girder[1]/2, dim_girder[2]/2] + center_girder])
        kps_1 = np.concatenate((line_start_1, line_end_1), axis=0)
        # right and front side (left part only)
        _shift_from_tower_center = shift_from_tower_center.copy()
        _shift_from_tower_center[:, 0] *= -1
        line_start_2 = np.array([center_tower + _shift_from_tower_center[0],
                               center_tower + _shift_from_tower_center[1]])
        line_end_2 = np.array([[-blank_3, -dim_girder[1]/2, dim_girder[2]/2] + center_girder,
                             [center_tower[0]+blank_2, -dim_girder[1]/2, dim_girder[2]/2] + center_girder])
        kps_2 = np.concatenate((line_start_2, line_end_2), axis=0)

        # lines according to keypoints
        # left and front side (left part only)
        lines_1 = get_lines_from_kps(kps_1, n=num_cable_1)
        # right and front side (left part only)
        lines_2 = get_lines_from_kps(kps_2, n=num_cable_2)
        # front side (left part only)
        lines_3 = np.concatenate((lines_1, lines_2), axis=0)
        # front side (right part only)
        lines_4 = lines_3.copy()
        lines_4[:, [0, 3]] *= -1
        # front side
        lines_front = np.concatenate((lines_3, lines_4), axis=0)

    else:
        # control the number of cables
        num_cable_1 = int(round((dim_girder[0] / 2 - abs(center_tower[0]) - blank_1 - blank_2) / interval))
        num_cable_2 = int(round((abs(center_tower[0]/2) - blank_2 - blank_3 / 2) / interval))

        # find 4 keypoints for one side of cables: (kps_x)
        # left and front side
        line_start_1 = np.array([center_tower + shift_from_tower_center[0],
                               center_tower + shift_from_tower_center[1]])
        line_end_1 = np.array([[-dim_girder[0]/2+blank_1, -dim_girder[1]/2, dim_girder[2]/2] + center_girder,
                             [center_tower[0]-blank_2, -dim_girder[1]/2, dim_girder[2]/2] + center_girder])
        kps_1 = np.concatenate((line_start_1, line_end_1), axis=0)
        # right and front side (left part only)
        _shift_from_tower_center = shift_from_tower_center.copy()
        _shift_from_tower_center[:, 0] *= -1
        line_start_2 = np.array([center_tower + _shift_from_tower_center[0],
                               center_tower + _shift_from_tower_center[1]])
        line_end_2 = np.array([[center_tower[0]/2-blank_3, -dim_girder[1]/2, dim_girder[2]/2] + center_girder,
                             [center_tower[0]+blank_2, -dim_girder[1]/2, dim_girder[2]/2] + center_girder])
        kps_2 = np.concatenate((line_start_2, line_end_2), axis=0)
        # left and front side (center part only)
        line_start_3 = np.array([shift_from_tower_center[0], shift_from_tower_center[1]])
        line_end_3 = np.array([[center_tower[0]/2+blank_3, -dim_girder[1]/2, dim_girder[2]/2] + center_girder,
                             [-blank_2, -dim_girder[1]/2, dim_girder[2]/2] + center_girder])
        kps_3 = np.concatenate((line_start_3, line_end_3), axis=0)

        # lines according to keypoints
        # left and front side (left part only)
        lines_1 = get_lines_from_kps(kps_1, n=num_cable_1)
        # right and front side (left part only)
        lines_2 = get_lines_from_kps(kps_2, n=num_cable_2)
        # left and front side (center part only)
        lines_3 = get_lines_from_kps(kps_3, n=num_cable_2+1)
        # front side (left + 1/2 center part only)
        lines_4 = np.concatenate((lines_1, lines_2, lines_3), axis=0)
        # front side (right + 1/2 center part only)
        lines_5 = lines_4.copy()
        lines_5[:, [0, 3]] *= -1
        # front side
        lines_front = np.concatenate((lines_4, lines_5), axis=0)

    # back side (use front side)
    lines_back = lines_front.copy()
    lines_back[:, [1, 4]] *= -1
    # center side
    lines_center = (lines_front + lines_back) / 2

    # side number
    if num_side == 3:
        lines = np.concatenate((lines_front, lines_back, lines_center), axis=0)
    elif num_side == 2:
        lines = np.concatenate((lines_front, lines_back), axis=0)
    else:
        lines = lines_center

    # draw lines
    for line in lines:
        p1, p2 = line[0:3], line[3:6]
        draw_cube_2(p1, p2, radius=radius)


def draw_sus_cable_truss(num_tower, center_girder, dim_girder, num_block, blank=1,
                         shift_from_tower_center=None, center_tower=None, radius=0.005, color=None):
    """
    Draw suspension cables. Only suitable for num_tower={2, 3} and num_side=2.
    :param num_tower: int in {2, 3}
    :param center_girder: triple, center of the bbox of girder
    :param dim_girder: triple, size of the bbox of girder
    :param num_block: int, number of blocks
    :param blank: int, number of blanks, control the endpoint to cable
    :param shift_from_tower_center: 2*3 float, label for the connection of tower and cable
    :param center_tower: float, for the center of the first tower (if 2 or 3)
    :param radius: radius of cable
    :param color: triple, (R, G, B)
    """
    if num_tower not in [2, 3]:
        raise ValueError

    # get keypoints
    verts = get_keypoints(center_girder, dim_girder, num_block)
    verts_front = verts[2::4]  # keypoints on truss in the front side

    def solve_parabola(x1, x2, x3, y1, y2, y3):
        X = np.array([[x1 ** 2, x1, 1],
                      [x2 ** 2, x2, 1],
                      [x3 ** 2, x3, 1]])
        Y = np.array([[y1], [y2], [y3]])
        a, b, c = np.linalg.inv(X).dot(Y)
        return a, b, c

    def _draw(p1, p2, p3, kps_1, radius):
        # get interpolating points, the parabola line is divided to 16 segments
        kps_2_x = np.linspace(p1[0], p2[0], num=16)
        a1, b1, c1 = solve_parabola(p1[0], p2[0], p3[0], p1[1], p2[1], p3[1])
        kps_2_y = a1 * kps_2_x ** 2 + b1 * kps_2_x + c1
        a2, b2, c2 = solve_parabola(p1[0], p2[0], p3[0], p1[2], p2[2], p3[2])
        kps_2_z = a2 * kps_2_x ** 2 + b2 * kps_2_x + c2
        kps_2 = np.concatenate((kps_2_x, kps_2_y, kps_2_z)).reshape((3, -1)).T

        # find keypoints on main cable wrt truss
        kps_3_x = kps_1[:, 0]
        kps_3_y = a1 * kps_3_x ** 2 + b1 * kps_3_x + c1
        kps_3_z = a2 * kps_3_x ** 2 + b2 * kps_3_x + c2
        kps_3 = np.concatenate((kps_3_x, kps_3_y, kps_3_z)).reshape((3, -1)).T

        # draw main cable
        for i in range(kps_2.shape[0] - 1):
            p1, p2 = kps_2[i], kps_2[i + 1]
            draw_cube_2(p1, p2, radius=2*radius)
        # draw cable
        for i in range(kps_1.shape[0]):
            p1, p2 = kps_3[i], kps_1[i]
            draw_cube_2(p1, p2, radius=radius)

    if num_tower == 2:

        '''center part'''
        # find keypoints on truss (kps_1)
        which_block = int(abs(center_tower[0] - (center_girder[0] - dim_girder[0] / 2)) / (dim_girder[0] / num_block)) + 1
        kps_1 = verts_front[which_block:num_block-which_block+1]
        # if every 2 block
        if 1:
            kps_1 = kps_1[::2]

        # find keypoints on main cable (kps_2)
        # three points on the suspension parabola line
        p1 = np.array([center_tower[0] - shift_from_tower_center[0, 0],
                       center_tower[1] + shift_from_tower_center[0, 1],
                       center_tower[2] + shift_from_tower_center[0, 2]])
        p2 = p1.copy()
        p2[0] *= -1
        p3 = np.array([center_girder[0], center_girder[1]-dim_girder[1]/2, center_girder[2]+3*dim_girder[2]/2])

        # draw
        _draw(p1, p2, p3, kps_1, radius)
        p1[1], p2[1], p3[1], kps_1[:, 1] = -p1[1], -p2[1], -p3[1], -kps_1[:, 1]
        _draw(p1, p2, p3, kps_1, radius)

        '''left part'''
        # find keypoints on truss (kps_1)
        which_block = int(abs(center_tower[0] - (center_girder[0] - dim_girder[0] / 2)) / (dim_girder[0] / num_block))
        kps_1 = verts_front[0:which_block+1]
        # if every 2 block
        if 1:
            kps_1 = kps_1[::2]

        # find keypoints on main cable (kps_2)
        # three points on the suspension parabola line
        p1 = np.array([center_tower[0] + shift_from_tower_center[0, 0],
                       center_tower[1] + shift_from_tower_center[0, 1],
                       center_tower[2] + shift_from_tower_center[0, 2]])
        p2 = kps_1[0]
        # TODO: adjust p3
        p3 = np.array([0.5*p1[0] + 0.5*p2[0], 0.5*p1[1] + 0.5*p2[1], 0.4*p1[2] + 0.6*p2[2]])

        # draw
        _draw(p1, p2, p3, kps_1, radius)
        p1[1], p2[1], p3[1], kps_1[:, 1] = -p1[1], -p2[1], -p3[1], -kps_1[:, 1]
        _draw(p1, p2, p3, kps_1, radius)
        p1[0], p2[0], p3[0], kps_1[:, 0] = -p1[0], -p2[0], -p3[0], -kps_1[:, 0]
        _draw(p1, p2, p3, kps_1, radius)
        p1[1], p2[1], p3[1], kps_1[:, 1] = -p1[1], -p2[1], -p3[1], -kps_1[:, 1]
        _draw(p1, p2, p3, kps_1, radius)

    else:
        # TODO: num_tower=3
        pass


if __name__ == '__main__':

    '''parameters'''

    # clear console window
    os.system('cls')

    # reset
    reset_blend()

    # where is the *.obj file
    file_loc_root = 'C:/PythonProject/MyProject/utils/mesh-generation/obj_tower'
    # file_loc_root = '/Users/hufangqiao/Nutstore Files/my_nutstore/my_code/mesh-generation/obj_tower'

    '''tower'''
    num_tower = 2
    files_tower = glob.glob(os.path.join(file_loc_root, '*.obj'))
    files_tower.sort()

    fn_tower = np.random.choice(files_tower)

    center_tower = np.array([-2.5, 0, 0])
    color_tower = np.array([0.65, 0.65, 0.55])

    '''label on tower'''
    # read label dict from txt file
    f = open(os.path.join(file_loc_root,'tower_label_dict.txt'), 'r')
    string = f.read()
    label_tower = eval(string)
    f.close()
    key = fn_tower.split('/')[-1].split('\\')[-1].split('.')[0]
    label_tower_current = np.array(label_tower[key])
    # where to draw cable, shift from center_tower
    shift_from_tower_center = label_tower_current[2:4]

    '''girder'''
    num_block = 72
    center_girder = label_tower_current[0]
    dim_girder = label_tower_current[1]
    dim_girder[0] = random(7, 8)
    color_girder = np.array([[0.8, 0.15, 0.15],
                             [0.65, 0.65, 0.65]])[np.random.randint(0, 2)] * random(0.85, 1)

    '''draw'''
    # draw tower
    draw_tower(fn_tower=fn_tower, num_tower=num_tower, center=center_tower, color=color_tower)

    # draw truss
    draw_truss(center=center_girder, dim=dim_girder, num_block=num_block, case=0, sym_plane=-2.5, color=color_girder)

    # # draw truss cable
    # draw_cable_truss(num_tower=num_tower, center_girder=center_girder, dim_girder=dim_girder,
    #                  num_block=num_block, num_side=3, center_tower=center_tower,
    #                  shift_from_tower_center=shift_from_tower_center, radius=0.005, color=None)

    # draw truss suspension cable
    draw_sus_cable_truss(num_tower=2, center_girder=center_girder, dim_girder=dim_girder, num_block=num_block,
                         blank=1, shift_from_tower_center=shift_from_tower_center,
                         center_tower=center_tower, radius=0.005, color=None)

    # # draw road
    # # move along z axis
    # center_girder[2] += dim_girder[2] * random(-0.2, 0.4)
    # dim_girder[2] *= random(0.1, 0.3)
    # draw_road(center_girder=center_girder, dim_girder=dim_girder, color=color_girder)
    #
    # # draw road cable
    # draw_cable_road(num_tower=num_tower, center_girder=center_girder, dim_girder=dim_girder,
    #                 blanks=(0.2, 0.2, 0.2), interval=0.1, num_side=2,
    #                 shift_from_tower_center=shift_from_tower_center, center_tower=center_tower,
    #                 radius=0.005, color=None)

