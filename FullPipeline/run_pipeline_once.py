import os
import sys

sys.path.append( os.getcwd() )
sys.path.append( os.path.split( os.getcwd() )[0] )

from Globals.globalvars import Glb

def run_pipeline_once(cntr):
    print (Glb.results_folder)
    python_int = sys.executable
    # remove empty imagesGlobals
    filename = "../DataPrep_EmptyNot/classify_affine_notEmpty.py"
    #os.system( " ".join([python_int,filename]) )

    # remove images with invisible products
    filename= "../Data_Prep_Affine/classify_affine_ne_isVisible.py"
    #os.system( " ".join([python_int,filename]) )


    # split train/val/test
    filename= "../Data_Prep_Affine/A.V.NE.split_train_val_test.py"
    os.system( " ".join([python_int,filename]) )

    # balance (aff+persp)
    filename= "../Data_Prep_Affine/A.V.NE.balance_pipe.py"
    os.system( " ".join([python_int,filename]) )
    filename= "../Data_Prep_Affine/A.V.NE.balance_affAugment_pipe.py"
    os.system( " ".join([python_int,filename]) )

    # train
    doEffNet=True
    if doEffNet:
        filename = "../../keras-efficientnets/test.py " + Glb.images_balanced_folder + "/Bal_v14/Ind-0"
    else:
        filename = "../IsVisible_MakeModel/train_gpu0.py"
    os.system( " ".join([python_int,filename]) )


for cntr in range(1):
    run_pipeline_once(cntr)