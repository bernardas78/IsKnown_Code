import  Train_clsf_from_autoenc
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

clsf_from_autoenc_params = {
    "fc_version": 1,
}
for i in range(1):
    model_clsf = Train_clsf_from_autoenc.train_single_classifier (**clsf_from_autoenc_params)
