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
    filename= "../Data_Prep_Affine/A.V.NE.split_train_val_test.py"
    #os.system( " ".join(["python",filename]) )

    # balance (aff+persp)
    filename= "../Data_Prep_Affine/A.V.NE.balance_pipe.py"
    #os.system( " ".join(["python",filename]) )
    filename= "../Data_Prep_Affine/A.V.NE.balance_affAugment_pipe.py"
    #os.system( " ".join(["python",filename]) )

    # train
    filename= "../IsVisible_MakeModel/train_gpu0.py"
    #os.system( " ".join(["python",filename]) )


run_pipeline_once()