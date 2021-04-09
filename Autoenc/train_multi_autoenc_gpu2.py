import  Train_autoenc
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

autoenc_params = {
    "autoenc_version": 2,
}
for i in range(1):
    model_clsf = Train_autoenc.trainModel (**autoenc_params)
