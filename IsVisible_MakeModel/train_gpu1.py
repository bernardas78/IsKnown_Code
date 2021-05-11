import  train_multi_versions_single_gpu
import os

gpu_id = 1
hier_lvl = 4

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)

for i in range(5):
    model_clsf = train_multi_versions_single_gpu.train_on_single_gpu (gpu_id=gpu_id, hier_lvl=hier_lvl)
