import  train_multi_versions_single_gpu
import os
from datetime import date
from Globals.globalvars import Glb

gpu_id = 1
hier_lvl = 0

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)

model_filename = os.path.join(Glb.results_folder,
                               "model_clsf_from_isVisible_{}_gpu{}_hier{}.h5".format(date.today().strftime("%Y%m%d"),
                                                                                     gpu_id,
                                                                                     hier_lvl))
lc_filename = os.path.join(Glb.results_folder,
                           "lc_clsf_from_isVisible_{}_gpu{}_hier{}.csv".format(date.today().strftime("%Y%m%d"),
                                                                               gpu_id,
                                                                               hier_lvl))
data_dir = os.path.join(Glb.images_balanced_folder, "Bal_v14", "Ind-{}".format(hier_lvl) )

for i in range(1):
    model_clsf = train_multi_versions_single_gpu.train_on_single_gpu (gpu_id=gpu_id,
                                                                      model_filename=model_filename,
                                                                      lc_filename=lc_filename,
                                                                      data_dir=data_dir)
