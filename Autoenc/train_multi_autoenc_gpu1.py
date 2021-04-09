import  Train_autoenc
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

autoenc_params = {
    "autoenc_version": 3,
}
for i in range(1):
    model_clsf = Train_autoenc.trainModel (**autoenc_params)
