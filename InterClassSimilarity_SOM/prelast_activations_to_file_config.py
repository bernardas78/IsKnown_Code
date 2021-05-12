from InterClassSimilarity_SOM.prelast_activations_to_file import put_prelast_act_to_file
from Globals.globalvars import Glb
import os

hier_lvl = 2

#model = load_model( os.path.join( Glb.results_folder, "model_clsf_from_isVisible_20210415_gpu1.h5") ) # 83% test accuracy
#act_filename_pattern = os.path.join( Glb.results_folder, "activations_prelast_clsf_from_isVisible_20210415_gpu1_{}.h5")
model_filename = os.path.join( Glb.results_folder, "model_clsf_from_isVisible_20210511_gpu0.h5")  # Hier1-4
act_filename_pattern = os.path.join( Glb.results_folder, "activations_prelast_clsf_from_isVisible_20210511_gpu0_{}_hier{}.h5") # Hier1-4

for set_name in ["Test","Train","Val"]:
    put_prelast_act_to_file(model_filename=model_filename, act_filename_pattern=act_filename_pattern, set_name=set_name, hier_lvl=hier_lvl)