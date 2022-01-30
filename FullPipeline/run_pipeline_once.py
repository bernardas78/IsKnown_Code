from Globals.globalvars import Glb
import os

def run_pipeline_once():
    print (Glb.results_folder)
    # remove empty images
    filename = "../DataPrep_EmptyNot/classify_affine_notEmpty.py"
    #os.system( " ".join(["python",filename]) )

    # remove images with invisible products
    filename= "../Data_Prep_Affine/classify_affine_ne_isVisible.py"
    #os.system( " ".join(["python",filename]) )


    # split train/val/test
    #Data_Prep_Affine\A.V.NE.split_train_val_test.ps1

    # balance (aff+persp)
    #Data_Prep_Affine\A.V.NE.balance.py
    #Data_Prep_Affine\A.V.NE.balance_affAugment.py

    # train
    #IsVisible_MakeModel/train_gpu0.py

    # record metrics:
    #Not_implemented

run_pipeline_once()