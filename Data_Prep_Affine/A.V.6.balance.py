# Create N augmented files for each category
#   Src: D:\Visible_Data\3.SplitTrainVal\[Train|Val]\[1|2|3|4|m|ma]\* (listed in ListLabelledFiles.csv)
#   Dest: D:\Visible_Data\4.Augmented\[Train|Val]\[1|2|3|4|m|ma]\<origfilename>_counter.[png|jpg]


import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import os
import numpy as np

src_folder = r"A:\IsKnown_Images\A_Hier"
#src_folder = r"A:\IsKnown_Images\A_V_Hier"

save_to_dir_template = r"A:\IsKnown_Images\A_Balanced"
#save_to_dir_template = r"A:\IsKnown_Images\A_V_Balanced"

def augment_folder (src_classcode_lvl_folder, dest_classcode_lvl_folder, img_cnt):
    # img_cnt - how many images total should be created in destination. Originals copied anyway

    datagen=ImageDataGenerator(
        rotation_range=10,
        width_shift_range=32,
        height_shift_range=32,
        zoom_range=0.1,
        horizontal_flip=True
    )

    # First, copy original files to dest
    print ("Copying original files in {}".format (src_classcode_lvl_folder) )
    shutil.copytree( src_classcode_lvl_folder, dest_classcode_lvl_folder)   # succeeds if no dest folder parent
    files_copied = len ( os.listdir(dest_classcode_lvl_folder) )
    print("Done Copying {} original files".format(files_copied) )

    # Init how many files augmented for the cur_barcode (originals included)
    files_agmented_cur_barcode = files_copied

    #print('Before flow_from_dataframe. Shape: ' + str(df_files_cur.shape))
    class_code = src_classcode_lvl_folder.split("\\")[-1]
    filepaths = [ os.path.join(src_classcode_lvl_folder,classs) for classs in os.listdir (src_classcode_lvl_folder) ]
    df_files = pd.DataFrame( {'filepath': filepaths, 'class_code': np.repeat(class_code , len(filepaths) ) } )
    augmenter=datagen.flow_from_dataframe(dataframe=df_files, x_col="filepath", y_col="class_code",
                                              class_mode=None, target_size=(256,256),
                                              save_to_dir= dest_classcode_lvl_folder , save_format="jpg", save_prefix="",
                                              batch_size=32, shuffle=False)

    while files_agmented_cur_barcode < img_cnt:
        X = augmenter.next()
        files_agmented_cur_barcode += X.shape[0]
        #print ("Class {0}, augmented {1} of {2}".format ( cur_barcode, files_agmented_cur_barcode, files_per_class[cur_set] ) )



# Loop structure v<ver>\[Hier-x]\[TrainValTest]\classcode and balance
for ver_lvl in os.listdir(src_folder):
    ver_lvl_full_folder = os.path.join(src_folder,ver_lvl)

    for hier_lvl in os.listdir(ver_lvl_full_folder): # list Hier-x
        hier_lvl_full_folder = os.path.join(ver_lvl_full_folder,hier_lvl)

        #print (hier_lvl_full_folder)
        for set_lvl in os.listdir(hier_lvl_full_folder):    # list Train, Val, Test
            set_lvl_folder = os.path.join(hier_lvl_full_folder,set_lvl)

            #if set_lvl!="Test":
            #    continue

            # Balance up to max number of images in the set level
            if set_lvl=="Test":
                max_count_imgs_set_lvl = 0  # don't augment test set, just copy originals
            else:
                max_count_imgs_set_lvl = np.max( [ len( os.listdir(os.path.join(set_lvl_folder,classs) ) ) for classs in os.listdir(set_lvl_folder) ] )
            #print (set_lvl_folder+" "+str(max_count_imgs_set_lvl))

            for classcode_lvl in os.listdir(set_lvl_folder):    # list class codes (barcode or shortened)
                classcode_lvl_folder = os.path.join(set_lvl_folder,classcode_lvl)
                dest_classcode_lvl_folder = os.path.join(save_to_dir_template,ver_lvl,hier_lvl,set_lvl,classcode_lvl)
                #print (classcode_lvl_folder)

                augment_folder (src_classcode_lvl_folder=classcode_lvl_folder, dest_classcode_lvl_folder=dest_classcode_lvl_folder, img_cnt=max_count_imgs_set_lvl)