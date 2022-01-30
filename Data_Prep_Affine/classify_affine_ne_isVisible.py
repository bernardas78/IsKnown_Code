# Using IsVisible classifiers
#   2 best F-1 models
# Make Datasets from cropped data (each model)

from tensorflow.keras.models import load_model
from pathlib import Path    # recursive list of files
from PIL import Image
import numpy as np
import os
import shutil
from Globals.globalvars import Glb


#data_folder = "A:\\IsKnown_Images\\Affine_EmptyNot\\Affine_NotEmpty"
#data_folder_root = "A:\\IsKnown_Images\\Affine_EmptyNot\\Affine_NotEmpty_{}"
data_folder_root = os.path.join( Glb.images_folder,"Affine_EmptyNot","Affine_NotEmpty")

dest_folder_visible = os.path.join( Glb.images_folder, "Cleaned_Aff_NE_AutoVisible")
if os.path.exists(dest_folder_visible):
    shutil.rmtree(dest_folder_visible)
dest_folder_visible = os.path.join(dest_folder_visible, "{}_v{}")

dest_folder_invisible = os.path.join( Glb.images_folder, "Cleaned_Aff_NE_AutoInvisible")
if os.path.exists(dest_folder_invisible):
    shutil.rmtree(dest_folder_invisible)
dest_folder_invisible = os.path.join(dest_folder_invisible, "{}_v{}")



#emptynesses = ["Bal", "Overfit", "Siam"]
emptynesses = ["Bal"]
#versions = [62,14]
versions = [14]

# model to load from
model_path_pattern = os.path.join( Glb.class_mixture_models_folder, "model_v{}.h5")

# best F-1 versions (os.environ['GDRIVE'] + "\\PhD_Data\\ClassMixture_Metrics\\eval_metrics.csv")



def prepareImage (filename, target_size):
	# Load image ( RGB )
	img = Image.open(filename)
	# Crop bottom square frame (assume H>W)
	w, h = img.size
	img = img.crop ((0,h-w,w,h)) if h>w else img
	# Resize to target
	img = img.resize ((target_size,target_size))
	# Normalize to range 0..1
	img = np.array(img) / 255.
	#print ("img[0:2,:,:]: {}".format(img[:2,0,0]))    data_folder = data_folder_root.format(emptyness)

	return np.stack([img])  # quick way to add a didest_folder_dest_folder_invisiblevisibledest_folder_dest_folder_invisiblevisiblemension


for emptyness in emptynesses:
    #data_folder = data_folder_root.format(emptyness)
    data_folder = data_folder_root

    for version in versions:
        # Load model
        model_path = model_path_pattern.format(version)
        print ("Loading model {}".format(model_path))
        model = load_model ( model_path )
        target_size = model.layers[0].input_shape[1]
        #print (model.summary())
        print ("Done Loading model. Target sdest_folderize: {}".format(target_size))

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

            # classes are [0:Invisible; 1:Visible]
            dest_folder = dest_folder_visible if pred[0,1] > 0.5 else dest_folder_invisible
            barcode, file_name_stripped = img_file.split(os.sep)[-2:]

            # make folder and copy file
            dest_folder_barcode = os.path.join(dest_folder.format(emptyness, version), barcode)
            if not os.path.exists(dest_folder_barcode):
                os.makedirs(dest_folder_barcode)
            dest_fullname = os.path.join(dest_folder_barcode,file_name_stripped)
            #print ("shutil.copy({}, {})".format(img_file, dest_fullname))

            shutil.copy(img_file, dest_fullname)
            total_copied += 1 if pred[0,1] > 0.5 else 0
            loop_cntr +=1

            if loop_cntr%100==0:
                print ("Files copied {}, processed {} / {}".format(total_copied,loop_cntr,len(img_file_lst)))

        print ("Total copied {}/{}".format(total_copied, len(img_file_lst)))