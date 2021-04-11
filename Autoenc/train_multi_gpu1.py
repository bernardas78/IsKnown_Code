import  Train_clsf_from_autoenc
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

clsf_from_autoenc_params = {
    "fc_version": 2,
    "autoenc_version": 3,
    "autoenc_datetrained": "20210409"
    #"autoenc_filename": "model_autoenc_20210409_v3.h5"
}
for i in range(1):
    model_clsf = Train_clsf_from_autoenc.train_single_classifier (**clsf_from_autoenc_params)
