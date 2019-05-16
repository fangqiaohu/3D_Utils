import bpy
import numpy as np
import time
import glob
import os

fn_read_list = glob.glob(<'PATH'>)
fn_read = fn_read_list[0]

# delete current model
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# renew
def reset_blend():
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


reset_blend()

# import obj file
imported_object = bpy.ops.import_scene.obj(filepath=fn_read)
obj_object = bpy.context.object
# print('Imported name: ', obj_object.name)

# import a camera
bpy.ops.object.camera_add()
# scene = bpy.context.scene
# scene.camera = bpy.context.object

# import a point light
bpy.ops.object.lamp_add(type='POINT')
lamp = bpy.data.lamps['Point']
lamp.energy = 3
lamp.distance = 100


# render: camera looks at object center, the distance from object center to camera center is fixed,
# theta and phi are under Spherical Coordinate System (wiki)
# for theta in np.arange(np.pi/3, np.pi/2, np.pi/12):

theta = 75
# from d to rad
theta = theta * np.pi / 180

for phi in np.arange(0, 360, 30):

    # from d to rad
    phi = phi * np.pi / 180

    # R: how far is camera center to object center
    R = 8
    R_PointLight = 1.2 * R
    central_point = cp = [0, 0, 0]

    x = R * np.sin(theta) * np.cos(phi) + cp[0]
    y = R * np.sin(theta) * np.sin(phi) + cp[1]
    z = R * np.cos(theta) + cp[2]

    rx = 0.0
    ry = 0.0
    rz = 0.0

    rx += theta
    ry += 0.0
    rz += - (np.pi / 2.0 - phi) + np.pi

    x_PointLight = R_PointLight * np.sin(theta) * np.cos(phi) + cp[0]
    y_PointLight = R_PointLight * np.sin(theta) * np.sin(phi) + cp[1]
    z_PointLight = R_PointLight * np.cos(theta) + cp[2]

    # translate and rotate the camera
    # s.t. fit the image center (u0, v0) to world center (X0, Y0, Z0)
    bpy.data.objects['Camera'].location = (x, y, z)
    bpy.data.objects['Camera'].rotation_euler = (rx, ry, rz)

    # translate and rotate the light
    bpy.data.objects['Point'].location = (x_PointLight, y_PointLight, z_PointLight)

    # resolution of rendered image
    bpy.data.scenes['Scene'].render.resolution_x = 1024
    bpy.data.scenes['Scene'].render.resolution_y = 1024
    bpy.data.scenes['Scene'].render.resolution_percentage = 100

    # file name
    t = int(round(theta / np.pi * 180))
    p = int(round(phi / np.pi * 180))
    obj_name = os.path.basename(fn_read).split('.')[0]
    # fn_write = file_loc_root + 'image/' + '%s/' % obj_name + '%s_%03d%03d.png' % (obj_name, t, p)
    fn_write = fn_read.split('\\')[0].replace('obj_bridge', 'image') + '/%s_%03d%03d.png' % (obj_name, t, p)

    scene = bpy.data.scenes["Scene"]
    scene.render.image_settings.file_format = 'PNG'

    bpy.data.objects['Point'].select = 0
    bpy.data.objects['Camera'].select = 1
    bpy.context.scene.camera = bpy.data.objects['Camera']

    # render op1
    bpy.data.scenes['Scene'].render.filepath = fn_write
    bpy.data.scenes['Scene'].render.use_placeholder = True
    bpy.data.scenes['Scene'].render.alpha_mode = 'TRANSPARENT'
    bpy.data.scenes['Scene'].render.image_settings.color_mode = 'RGBA'
    bpy.data.scenes['Scene'].render.image_settings.compression = 50
    # DO NOT USE bpy.ops.render.render('INVOKE_DEFAULT', write_still=True)!!!
    bpy.ops.render.render(write_still=True)

    # render op2
    # bpy.ops.render.render()
    # bpy.data.images["Render Result"].save_render(save_fn)

    # time.sleep(0.01)

