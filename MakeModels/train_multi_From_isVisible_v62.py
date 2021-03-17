import train_From_IsVisible as t
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

for model_id in range(10):
  t.trainModel(
    epochs=100,
    isvisible_model_version=62,
    hier_lvl=0,
    aff_aug_lvl=20,
    val_acc_name='val_accuracy',   # linux: 'val_accuracy'; my: 'val_acc'
    model_id=model_id)
