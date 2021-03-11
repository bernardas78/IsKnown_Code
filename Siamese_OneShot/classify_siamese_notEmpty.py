# Using Siamese classifier
# Sorts test images by similarity to reference empty images
# Outputs to 2 dirs: Avg_dist, Min_dist (similarity to reference images)

# folder to classify
data_folder = "A:\\IsKnown_Images\\Affine"

# compare against all of these empty images
anchors_folder = "A:\\IsKnown_Images\\EmptyNot\\Empty"

dest_folder_notempty = "A:\\IsKnown_Images\\Siamese_EmptyNot\\Siamese_NotEmpty\\"
dest_folder_empty = "A:\\IsKnown_Images\\Siamese_EmptyNot\\Siamese_Empty\\"

dest_folder_avg_dist = "A:\\IsKnown_Images\\Siamese_EmptyNot\\Avg_dist\\"
dest_folder_min_dist = "A:\\IsKnown_Images\\Siamese_EmptyNot\\Min_dist\\"

# model to load from
#model_path = r"A:\\IsKnown_Results\\model_siamese_emptyNot_20210212.h5"
model_path = r"A:\\IsKnown_Results\\model_siamese_emptyNot_20210215.h5"

target_size = 224

from tensorflow.keras.models import load_model
from pathlib import Path    # recursive list of files
from PIL import Image
import numpy as np
import os
import shutil
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

def prepareImage (filename, target_size):
	# Load image ( RGB )
    img = Image.open(filename)
    # Crop bottom square frame (assume H>W)
    w, h = img.size
    img = img.crop ((0,h-w,w,h)) if h>w else img
    # Resize to target
    img = img.resize ((target_size,target_size))
    # resnet prepare
    img = resnet_preprocess_input(np.array(img))
    #print ("img[0:2,:,:]: {}".format(img[:2,0,0]))
    return np.stack([img])  # quick way to add a dimension


# Load model
print ("Loading model {}".format(model_path))
model = load_model ( model_path )
target_size = model.layers[0].input_shape[0][1] #for resnet input_shape is [(None, 224, 224, 3)]; for isVisible (None, 256, 256, 3)
#print (model.summary())
print ("Done Loading model. Target size: {}".format(target_size))

# First load all anchor images (total 151)
print ("Loading anchors...")
anchors_all = []
cnt_anchors = 0
for anchor_filename in os.listdir(anchors_folder):
    full_anchor_filename = os.path.join(anchors_folder, anchor_filename)
    anchor_x = prepareImage( full_anchor_filename, target_size)
    anchors_all.append(anchor_x)
    cnt_anchors += 1
print ("Loaded {} anchors".format(cnt_anchors))

# Get list of image files to classify
img_file_lst = [str(thepath) for thepath in Path(data_folder).rglob("*.jpg")]
print ("Total files found: {}".format( len(img_file_lst) ) )


# function get a)min distance; b)avg distance  from all anchors
def get_distance_from_anchors (img):
    min_dist = 100000000.
    sum_dist = 0.
    for anchor in anchors_all:
        full_input = [anchor,img]
        pred_dist = model.predict(full_input)
        if pred_dist<min_dist:
            min_dist=pred_dist
        sum_dist += pred_dist
    return (min_dist, sum_dist/len(anchors_all))


# Counter for files copied and loop
total_copied = 0
loop_cntr = 0

for img_file in img_file_lst:
    # Classify and copy if Visible
    img_prepped = prepareImage(img_file, target_size)
    (min_dist, avg_dist) = get_distance_from_anchors(img_prepped)

    # copy to folder sorted by avg dist
    avg_dist_filename = os.path.join(dest_folder_avg_dist, '{:0>8}.jpg'.format( int(avg_dist * 1e+7) ) )
    shutil.copy(img_file, avg_dist_filename)

    # copy to folder sorted by min dist
    min_dist_filename = os.path.join(dest_folder_min_dist, '{:0>8}.jpg'.format( int(min_dist * 1e+7) ) )
    shutil.copy(img_file, min_dist_filename)


    # classes are [0:Empty; 1:NotEmpty]
    #dest_folder = dest_folder_notempty if pred[0,1] > 0.5 else dest_folder_empty
    #barcode, file_name_stripped = img_file.split("\\")[-2:]

    # make folder and copy file
    #dest_folder_barcode = os.path.join(dest_folder, barcode)
    #if not os.path.exists(dest_folder_barcode):
    #    os.makedirs(dest_folder_barcode)
    #dest_fullname = os.path.join(dest_folder_barcode,file_name_stripped)
    #print ("shutil.copy({}, {})".format(img_file, dest_fullname))

    #shutil.copy(img_file, dest_fullname)

    #total_copied += 1 if pred[0,1] > 0.5 else 0
    loop_cntr +=1

    if loop_cntr%100==0:
        print ("Files copied {}, processed {} / {}".format(total_copied,loop_cntr,len(img_file_lst)))

print ("Total copied {}/{}".format(total_copied, len(img_file_lst)))