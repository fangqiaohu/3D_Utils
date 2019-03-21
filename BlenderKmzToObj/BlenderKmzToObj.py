import bpy
import glob

file_loc_root = '<ROOT>'
kmz_files = glob.glob(file_loc_root + '*.kmz')
kmz_files.sort()

# check
kmz_file = kmz_files[0]


# delete current model
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# import kmz file
bpy.ops.import_scene.sketchup(filepath=kmz_file)

# export obj file
obj_file = kmz_file.replace('.kmz', '.obj')
bpy.ops.export_scene.obj(filepath=obj_file,
                         check_existing=True,
                         axis_forward='-Z',
                         axis_up='Y',
                         filter_glob="*.obj;*.mtl",
                         use_selection=False,
                         use_animation=False,
                         use_mesh_modifiers=True,
                         use_edges=False,
                         use_smooth_groups=False,
                         use_smooth_groups_bitflags=False,
                         use_normals=True,
                         use_uvs=False,
                         use_materials=False,
                         use_triangles=True,
                         use_nurbs=False,
                         use_vertex_groups=False,
                         use_blen_objects=False,
                         group_by_object=False,
                         group_by_material=False,
                         keep_vertex_order=False,
                         global_scale=1,
                         path_mode='AUTO')
