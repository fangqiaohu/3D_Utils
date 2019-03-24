# send keys
WIN = Sketchup::platform == :platform_win rescue RUBY_PLATFORM !~ /darwin/i
if WIN
  require "win32ole"
end
def send_escape()
  if WIN
    WIN32OLE::new('WScript.Shell').SendKeys('{ESC}')
  else
    Sketchup::send_action('cancelOperation:')
  end
end
def send_enter()
    WIN32OLE::new('WScript.Shell').SendKeys('{ENTER}')
end


# open *.skp file
model = Sketchup::active_model
fn_read = "SKPFILE"
cdef = model.definitions.load(fn_read)
point = Geom::Point3d::new( 0, 0, 0 )
cinst = model.active_entities.add_instance(
  cdef,
  Geom::Transformation::new( point )
)


# open *.skp file from url
model = Sketchup::active_model
url = "https://3dwarehouse.sketchup.com/warehouse/v1.0/publiccontent/e68714f3-5f9e-4045-881a-27365e360569"
cdef = model.definitions.load_from_url(url)
return unless cdef # good practice
point = Geom::Point3d::new( 0, 0, 0 )
cinst = model.active_entities.add_instance(
  cdef,
  Geom::Transformation::new( point )
)


# select all model
model = Sketchup.active_model
entities = model.active_entities
model.selection.add entities.to_a


# iteration (simple way)
fn_read_list.collect{|fn_read| kmz2obj(fn_read)}

# iteration (tedious way)
fn_read_list.each do |fn_read|
  kmz2obj(fn_read)
  # # sleep 1 sec and close the tip window
  # sleep(1)
  # send_enter()
end


# glob a folder
fn_read_list = Dir::glob("D:/Dataset/buildings/models_02913152/*.kmz")
fn_read_list.sort!
# fn_read_list = fn_read_list[1,6]
fn_read_list = fn_read_list[260..-1]
fn_write_list = fn_read_list.clone
fn_write_list.collect! {|e| e.gsub(".kmz", ".obj")}
