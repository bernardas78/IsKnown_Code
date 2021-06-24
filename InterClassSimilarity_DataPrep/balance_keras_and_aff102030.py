# Create N augmented files for each category
#   Src: A:\IsKnown_Images\Mrg_A_NE_TrainValTest\Bal_v14\[conf_mat|emb_dist|som_purity_impr]_<class_cnt>\[Train|Val|Test]\clstrId\*.jpg
#   Dest: A:\IsKnown_Images\Mrg_A_NE_BalKerasAff102030\Bal_v14\...


import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
import os
import numpy as np
from Data_Prep_Affine import affSequence

# Internal str: [conf_mat|emb_dist|som_purity_impr]_<class_cnt>\[Train|Val|Test]\clstrId\*.jpg
src_folder = r"A:\IsKnown_Images\Mrg_A_NE_TrainValTest\Bal_v14"

save_to_dir_template = r"A:\IsKnown_Images\Mrg_A_NE_BalKerasAff102030\Bal_v14"


def augment_folder (src_classcode_lvl_folder, dest_classcode_lvl_folder, img_cnt, keras_or_aff, aff_variation=None, copy_orig=False):
    # img_cnt - how many images total should be created in destination. Originals copied anyway

    if keras_or_aff=="aff":
        datagen=affSequence.affSequence(variation=aff_variation)
        save_prefix="aff" + str(aff_variation)
    else:
        datagen=ImageDataGenerator(
            rotation_range=10,
            width_shift_range=32,
            height_shift_range=32,
            zoom_range=0.1,
            horizontal_flip=True,
        )
        save_prefix="keras"

    files_copied = 0
    if copy_orig:
        # First, copy original files to dest
        print("Copying original files in {}".format(src_classcode_lvl_folder))
        shutil.copytree(src_classcode_lvl_folder, dest_classcode_lvl_folder)  # succeeds if no dest folder parent
        files_copied = len(os.listdir(dest_classcode_lvl_folder))
        print("Done Copying {} original files".format(files_copied))

    # Init how many files augmented for the cur_barcode (originals included)
    files_agmented_cur_barcode = files_copied

    #print('Before flow_from_dataframe. Shape: ' + str(df_files_cur.shape))
    class_code = src_classcode_lvl_folder.split("\\")[-1]
    filepaths = [ os.path.join(src_classcode_lvl_folder,classs) for classs in os.listdir (src_classcode_lvl_folder) ]
    df_files = pd.DataFrame( {'filepath': filepaths, 'class_code': np.repeat(class_code , len(filepaths) ) } )
    augmenter=datagen.flow_from_dataframe(dataframe=df_files, x_col="filepath", y_col="class_code",
                                              class_mode=None, target_size=(256,256),
                                              save_to_dir= dest_classcode_lvl_folder , save_format="jpg", save_prefix=save_prefix,
                                              batch_size=32, shuffle=False)

    while files_agmented_cur_barcode < img_cnt:
        X = augmenter.next()
        files_agmented_cur_barcode += X.shape[0]
    print ("Class {0}, augmented {1}".format ( class_code, files_agmented_cur_barcode ) )



# Loop structure [conf_mat|emb_dist|som_purity_impr]_<class_cnt>\[Train|Val|Test]\clstrId\*.jpg and balance
for mrg_meth_class_cnt_lvl in os.listdir(src_folder):
    mrg_meth_class_cnt_folder = os.path.join(src_folder,mrg_meth_class_cnt_lvl)

    if os.path.exists( os.path.join(save_to_dir_template,mrg_meth_class_cnt_lvl) ):
        print("Skipping {}".format(mrg_meth_class_cnt_lvl))
        continue
    print ("Processing {}".format(mrg_meth_class_cnt_lvl))

    #print (mrg_meth_class_cnt_folder_full)
    for set_lvl in os.listdir(mrg_meth_class_cnt_folder):    # list Train, Val, Test
        set_lvl_folder = os.path.join(mrg_meth_class_cnt_folder,set_lvl)

        # Balance up to max number of images in the set level
        if set_lvl=="Test":
            max_count_imgs_set_lvl = 0  # don't augment test set, just copy originals
        else:
            max_count_imgs_set_lvl = np.max( [ len( os.listdir(os.path.join(set_lvl_folder,classs) ) ) for classs in os.listdir(set_lvl_folder) ] )
        #print (set_lvl_folder+" "+str(max_count_imgs_set_lvl))

        for classcode_lvl in os.listdir(set_lvl_folder):    # list class codes (barcode or shortened)
            classcode_lvl_folder = os.path.join(set_lvl_folder,classcode_lvl)
            dest_classcode_lvl_folder = os.path.join(save_to_dir_template,mrg_meth_class_cnt_lvl,set_lvl,classcode_lvl)
            #print (classcode_lvl_folder)

            augment_folder(src_classcode_lvl_folder=classcode_lvl_folder, dest_classcode_lvl_folder=dest_classcode_lvl_folder,
                           img_cnt=max_count_imgs_set_lvl, keras_or_aff="keras", copy_orig=True)
            augment_folder(src_classcode_lvl_folder=classcode_lvl_folder, dest_classcode_lvl_folder=dest_classcode_lvl_folder,
                           img_cnt=max_count_imgs_set_lvl, keras_or_aff="aff", aff_variation=10)
            augment_folder(src_classcode_lvl_folder=classcode_lvl_folder, dest_classcode_lvl_folder=dest_classcode_lvl_folder,
                           img_cnt=max_count_imgs_set_lvl, keras_or_aff="aff", aff_variation=20)
            augment_folder(src_classcode_lvl_folder=classcode_lvl_folder, dest_classcode_lvl_folder=dest_classcode_lvl_folder,
                           img_cnt=max_count_imgs_set_lvl, keras_or_aff="aff", aff_variation=30)
            print ("Finished {}".format(classcode_lvl_folder))