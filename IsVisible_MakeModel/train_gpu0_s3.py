import  train_multi_versions_single_gpu
import os
from datetime import date
from Globals.globalvars import Glb

gpu_id = 0
hier_lvl = 0

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)

model_filename = os.path.join(Glb.results_folder,
                               "model_clsf_from_isVisible_{}.h5".format(date.today().strftime("%Y%m%d")))
lc_filename = os.path.join(Glb.results_folder,
                           "lc_clsf_from_isVisible_{}.csv".format(date.today().strftime("%Y%m%d") ))
data_dir = Glb.images_balanced_folder

for i in range(1):
    model_clsf = train_multi_versions_single_gpu.train_on_single_gpu (gpu_id=gpu_id,
                                                                      model_filename=model_filename,
                                                                      lc_filename=lc_filename,
                                                                      data_dir=data_dir)
