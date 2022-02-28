from Globals.globalvars import Glb
import os
import random
import shutil

# Source directory where barcode folders are located
images_folder = os.path.join ( Glb.images_folder, "Cleaned_Aff_NE_AutoVisible", "Bal_v14" )

# Destination directory
sets_folder = os.path.join ( Glb.images_folder, "Aff_NE_TrainValTest", "Bal_v14", "Ind-0" )
if os.path.exists(sets_folder):
    shutil.rmtree(sets_folder)

pct_train=0.64
pct_val=0.16
pct_test=0.2

# Expected structure: Cleaned_AutoVisible\Bal_v14\barcode\*.jpg
train_folder = os.path.join ( sets_folder, "Train" )
val_folder = os.path.join ( sets_folder, "Val" )
test_folder = os.path.join ( sets_folder, "Test" )

for barcode in os.listdir(images_folder):
    #print (barcode)

    barcode_dir = os.path.join (images_folder,barcode)

    barcode_filenames = os.listdir(barcode_dir)
    random.shuffle(barcode_filenames)
    if len(barcode_filenames)<3:
        continue

    #temp counts how many files already copied to train, val, test sets
    actual_train=0
    actual_val=0
    actual_test=0
    #

    for filename in barcode_filenames:
        # copy each file to a proper set's folder
        tot_copied = actual_train + actual_val + actual_test + 1e-7

        if (pct_train - actual_train/tot_copied > pct_val - actual_val/tot_copied and
            pct_train - actual_train/tot_copied > pct_test - actual_test/tot_copied):
            actual_train+=1
            dest_folder = train_folder
        elif (pct_val - actual_val/tot_copied > pct_test - actual_test/tot_copied ):
            actual_val+=1
            dest_folder = val_folder
        else:
            actual_test+=1
            dest_folder = test_folder

        dest_folder_full = os.path.join( sets_folder, dest_folder, barcode )
        if not os.path.exists(dest_folder_full):
            os.makedirs(dest_folder_full)
        full_filename = os.path.join(barcode_dir,filename)
        shutil.copy (full_filename, dest_folder_full)
