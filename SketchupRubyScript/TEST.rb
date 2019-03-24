# delete all model
model = Sketchup.active_model
entities = model.active_entities
model.active_entities.erase_entities entities.to_a

# import model
options_hash = {
:show_summary => false,
:validate_kmz => FALSE,
:merge_coplanar_faces => FALSE}
result = model.import("D:/Dataset/buildings/models_02913152/270dac48b7b602e0ab9b7ce77d5e846e.kmz", options_hash)
