from Common.conf_mat import Conf_Mat
from Globals.globalvars import Glb
import os

conf_mat = Conf_Mat(data_folder=r"C:\IsKnown_Images_IsVisible\Bal_v14\Ind-0\Test",
                    model_file=os.path.join(Glb.results_folder,"model_clsf_from_isVisible_20210415_gpu1.h5") )    # 83% test accuracy

conf_mat.make_conf_mat ( "conf_mat.png", None )