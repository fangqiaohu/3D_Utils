"# import *.kmz or *.dae file, export iso view *.jpg and mesh file *.obj"

# kmz to obj
def kmz2obj(fn_read)

  fn_write_obj = fn_read.gsub(".kmz", ".obj").gsub("models_02913152", "obj_02913152")
  fn_write_img = fn_read.gsub(".kmz", ".jpg").gsub("models_02913152", "img_pers_02913152")

  # if File::exist?(fn_write_obj) and File::exist?(fn_write_img)
  if File::exist?(fn_write_img)
    return "Exist error"
  end

  # delete all model
  model = Sketchup.active_model
  entities = model.active_entities
  model.active_entities.erase_entities entities.to_a

  # clear again
  Sketchup.active_model.entities.clear!

  # clear unused for 8 times
  for i in 0..7
    layers = Sketchup.active_model.layers
    layers.purge_unused
    materials = Sketchup.active_model.materials
    materials.purge_unused
    definitions = Sketchup.active_model.definitions
    definitions.purge_unused
    styles = Sketchup.active_model.styles
    styles.purge_unused
  end

  # import model
  options_hash = {
  :show_summary => FALSE,
  :validate_kmz => TRUE,
  :merge_coplanar_faces => TRUE}
  result = model.import(fn_read, options_hash)
  if not result
    return "Import error"
  end

  # clear unused for 1 time
  layers = Sketchup.active_model.layers
  layers.purge_unused
  materials = Sketchup.active_model.materials
  materials.purge_unused
  definitions = Sketchup.active_model.definitions
  definitions.purge_unused
  styles = Sketchup.active_model.styles
  styles.purge_unused

  # place to center
  cdef = model.definitions[-1]
  point = Geom::Point3d::new(0, 0, 0)
  transform = Geom::Transformation::new(point)
  cinst = model.active_entities.add_instance(cdef, transform)

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
  view.camera.perspective = TRUE
  Sketchup.send_action("viewIso:")
  entities = model.active_entities
  view.zoom entities
  view.write_image(options_hash)

  # # export *.obj model
  # options_hash = {
  # :units => "model",
  # :triangulated_faces => TRUE,
  # :doublesided_faces => FALSE,
  # :edges => TRUE,
  # :texture_maps => TRUE,
  # :swap_yz => FALSE,
  # :selectionset_only => FALSE,
  # :show_summary => FALSE}
  # model.export(fn_write_obj, options_hash)

  return "Success"

end


# glob a folder
fn_read_list = Dir::glob("D:/Dataset/buildings/models_02913152/*.kmz")
fn_read_list.sort!
fn_read_list = fn_read_list[2020..2040]
# fn_read_list = fn_read_list[0..-1]

# iteration (tedious way)
fn_read_list.each do |fn_read|

  # run main function
  time_start = Time.now
  status = kmz2obj(fn_read)
  time_cost = Time.now - time_start

  # write log
  aFile = File.new("D:/Dataset/buildings/log.csv", "a")
  line = "%s,%s,%s\n" % [fn_read.split("/")[-1].split(".")[0], status, time_cost]
  aFile.syswrite(line)
  aFile.close

end
