# Using IsVisible classifiers
#   2 best F-1 models
# Make Datasets from cropped data (each model)

data_folder = "A:\\IsKnown_Images\\Affine_S3"


# filename=sigmoid value (for sorting)
#dest_folder_sigmoid = "A:\\IsKnown_Images\\EmptyNotDebug\\Sigmoid_NotEmptyUndersampled\\"
dest_folder_sigmoid = "A:\\IsKnown_Images\\EmptyNotDebug\\Sigmoid_Balanced\\"

# model to load from
#model_path = r"A:\\IsKnown_Results\\model_emptyNot_20210222_NotEmptyUndersampled.h5"
model_path = r"A:\\IsKnown_Results\\model_emptyNot_20210222_Balanced.h5"
is_resnet=False

from tensorflow.keras.models import load_model
from pathlib import Path    # recursive list of files
from PIL import Image
import numpy as np
import os
import shutil
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from matplotlib import pyplot as plt

def prepareImage (filename, target_size):
	# Load image ( RGB )
    img = Image.open(filename)
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
        #print ("img[0:2,:,:]: {}".format(img[:2,0,0]))
    return np.stack([img])  # quick way to add a dimension


# Load model
print ("Loading model {}".format(model_path))
model = load_model ( model_path )
if is_resnet:
    target_size = model.layers[0].input_shape[0][1] # for resnet input_shape is [(None, 224, 224, 3)]
else:
    target_size = model.layers[0].input_shape[1] # for isVisible              (None, 256, 256, 3)
#print (model.summary())
print ("Done Loading model. Target size: {}".format(target_size))

# Get list of image files
img_file_lst = [str(thepath) for thepath in Path(data_folder).rglob("*.jpg")]
print ("Total files found: {}".format( len(img_file_lst) ) )

# Counter for files copied and loop
loop_cntr = 0

# Assemble sigmoid values by display
dic_sigmoids = {"Empty": [], "NotEmpty": []}

for img_file in img_file_lst:
    # Get "sigmoid" value and copy
    img_prepped = prepareImage(img_file, target_size)
    sigmoidvalue = model.predict (img_prepped)[0,0] # shape is [m,2], where [:,0]=Empty_prob, [:,1]=NotEmpty_prob

    # Assemble sigmoid values
    barcode = img_file.split("\\")[-2]

    # copy file: filename=sigmoid value
    dest_dir_barcode = os.path.join(dest_folder_sigmoid, barcode)
    if not os.path.exists(dest_dir_barcode):
        os.makedirs(dest_dir_barcode)
    dest_fullname = os.path.join(dest_dir_barcode, '{:0>8}.jpg'.format( int(sigmoidvalue * 1e+7) ) )
    shutil.copy(img_file, dest_fullname)

    loop_cntr +=1
    if loop_cntr%100==0:
        print ("Files processed {} / {}".format(loop_cntr,len(img_file_lst)))

print ("Total copied {}/{}".format(loop_cntr, len(img_file_lst)))
