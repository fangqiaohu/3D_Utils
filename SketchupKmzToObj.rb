"# import *.kmz or *.dae file, export *.obj file and iso view *.jpg"


# kmz to obj
def kmz2obj(fn_read)

  # # test
  # fn_read = "D:/Dataset/buildings/models_02913152/1a0e3f37dfd0de0e66a38f1adabdbc8e/models/model.dae"
  # fn_read = "D:/Dataset/buildings/models_02913152/1a0e3f37dfd0de0e66a38f1adabdbc8e.kmz"

  fn_write_obj = fn_read.gsub(".kmz", ".obj").gsub("models_02913152", "obj_02913152")
  fn_write_img = fn_read.gsub(".kmz", ".jpg").gsub("models_02913152", "img_pers_02913152")

  if File::exist?(fn_write_obj) and File::exist?(fn_write_img)
    print("FILE EXISE: [%s] and [%s]\n" % [fn_write_obj, fn_write_img])
    return FALSE
  end

  # delete all model
  model = Sketchup.active_model
  entities = model.active_entities
  model.active_entities.erase_entities entities.to_a

  # import model
  options_hash = {
  :validate_kmz => TRUE,
  :merge_coplanar_faces => TRUE,
  :show_summary => FALSE}
  result = model.import(fn_read, options_hash)
  if not result
    return FALSE
  end

  # place to center
  cdef = model.definitions[-1]
  point = Geom::Point3d::new(0, 0, 0)
  transform = Geom::Transformation::new(point)
  cinst = model.active_entities.add_instance(cdef, transform)

  # export *.obj model
  options_hash = {
  :units => "model",
  :triangulated_faces => TRUE,
  :doublesided_faces => FALSE,
  :edges => TRUE,
  :texture_maps => TRUE,
  :swap_yz => FALSE,
  :selectionset_only => FALSE,
  :show_summary => FALSE}
  model.export(fn_write_obj, options_hash)

  # export *.jpg model
  options_hash = {
  :filename => fn_write_img,
  :width => 2560,
  :height => 1440,
  :antialias => false,
  :compression => 0.9,
  :transparent => true}
  model = Sketchup.active_model
  view = model.active_view
  entities = model.active_entities
  view = view.zoom entities
  view.write_image(options_hash)

  # delete all model
  model = Sketchup.active_model
  entities = model.active_entities
  model.active_entities.erase_entities entities.to_a

end


# glob a folder
fn_read_list = Dir::glob("D:/Dataset/buildings/models_02913152/*.kmz")[1,6]
fn_read_list.sort!
fn_write_list = fn_read_list.clone
fn_write_list.collect! {|e| e.gsub(".kmz", ".obj")}
# print(fn_read_list)

# iteration (simple way)
fn_read_list.collect{|fn_read| kmz2obj(fn_read)}
