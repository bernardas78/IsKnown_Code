# Using IsVisible classifiers
#   2 best F-1 models
# Make Datasets from cropped data (each model)

from Globals.globalvars import Glb
from tensorflow.keras.models import load_model
from pathlib import Path    # recursive list of files
from PIL import Image
import PIL
import numpy as np
import os
import shutil
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

def prepareImage (filename, target_size):
	# Load image ( RGB )
    img = PIL.Image.open(filename)
    # Crop bottom square frame (assume H>W)
    w, h = img.size
    img = img.crop ((0,h-w,w,h)) if h>w else img
    # Resize to target
    img = img.resize ((target_size,target_size))
    # resnet prepare
    if is_resnet:
        img = resnet_preprocess_input(np.array(img))
    else:
        img = np.array(img) / 255.
    #print ("img[0:2,:,:]: {}".format(img[:2,0,0]))dest_folder_notempty = os.path.join( Glb.images_folder, "Affine_EmptyNot", "Affine_NotEmpty")

    return np.stack([img])  # quick way to add a dimension


#data_folder = "A:\\IsKnown_Images\\Affine"
data_folder =  os.path.join( Glb.images_folder, "Affine")

#dest_folder_notempty = "A:\\IsKnown_Images\\Affine_EmptyNot\\Affine_NotEmpty\\"
#dest_folder_empty = "A:\\IsKnown_Images\\Affine_EmptyNot\\Affine_Empty\\"
dest_folder_notempty = os.path.join( Glb.images_folder, "Affine_EmptyNot", "Affine_NotEmpty")
dest_folder_empty = os.path.join( Glb.images_folder, "Affine_EmptyNot", "Affine_Empty")

#remove folders if exist
if os.path.exists(dest_folder_notempty):
    shutil.rmtree(dest_folder_notempty)
if os.path.exists(dest_folder_empty):
    shutil.rmtree(dest_folder_empty)

# model to load from
#model_path = r"A:\\IsKnown_Results\\model_emptyNot_20210222_Balanced.h5"
#model_path = r"A:\\IsKnown_Results\\model_emptyNot_20210222_NotEmptyUndersampled.h5"
model_path = os.path.join ( Glb.results_folder, "model_emptyNot_20210222_Balanced.h5")

is_overfit = "Undersampled" in model_path
is_notempty_thr = (1-0.9954367) if is_overfit else 0.5

is_resnet = False

# Load model
print ("Loading model {}".format(model_path))
model = load_model ( model_path )
if is_resnet:
    target_size = model.layers[0].input_shape[0][1] #for resnet input_shape is [(None, 224, 224, 3)];
else:
    target_size = model.layers[0].input_shape[1]# for isVisible (None, 256, 256, 3)
#print (model.summary())
print ("Done Loading model. Target size: {}".format(target_size))

# Get list of image files
img_file_lst = [str(thepath) for thepath in Path(data_folder).rglob("*.jpg")]
print ("Total files found: {}".format( len(img_file_lst) ) )

# Counter for files copied and loop
total_copied = 0
loop_cntr = 0

for img_file in img_file_lst:
    # Classify and copy if Visible
    img_prepped = prepareImage(img_file, target_size)
    pred = model.predict (img_prepped)
    #print ("pred.shape:{}".format(pred.shape))

    # classes are [0:Empty; 1:NotEmpty]
    is_notempty = pred[0, 1] > is_notempty_thr
    barcode, file_name_stripped = img_file.split( os.sep )[-2:]

    # Split to barcode folders only non-empty; place empty to 1 folder
    if is_notempty:
        # make folder and copy file
        dest_folder_barcode = os.path.join(dest_folder_notempty, barcode)
        if not os.path.exists(dest_folder_barcode):
            os.makedirs(dest_folder_barcode)
        dest_fullname = os.path.join(dest_folder_barcode,file_name_stripped)
        shutil.copy(img_file, dest_fullname)
    #else:
    #    dest_fullname = os.path.join(dest_folder_empty, file_name_stripped)defined


    total_copied += 1 if is_notempty else 0
    loop_cntr +=1

    if loop_cntr%100==0:
        print ("Files copied {}, processed {} / {}".format(total_copied,loop_cntr,len(img_file_lst)))

print ("Total copied {}/{}".format(total_copied, len(img_file_lst)))