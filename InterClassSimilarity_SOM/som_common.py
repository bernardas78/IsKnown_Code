from Orange.data import Domain, DiscreteVariable, ContinuousVariable
import Orange
import os
import pickle
import time
import numpy as np
from Globals.globalvars import Glb

act_filename_pattern = "activations_prelast_clsf_from_isVisible_20210415_gpu1_{}.h5"

def loadActivations(set_name):
    act_filename = os.path.join(Glb.results_folder, act_filename_pattern.format(set_name) )
    # Load activation tables (~12:17-min train set)
    now = time.time()
    (act_prelast,lbls) = pickle.load(open(act_filename, 'rb'))
    print("Loaded activations in {} seconds".format(time.time() - now))

    domain = Domain(
                [ContinuousVariable.make("Feat_"+str(i)) for i in np.arange(act_prelast.shape[1])],
                DiscreteVariable.make(name="lbls", values=np.unique(lbls.astype(str) ) ) )
    now = time.time()
    orange_tab = Orange.data.Table.from_numpy( domain=domain, X=act_prelast, Y=lbls.astype(str))
    print("Made SOM clusters in {} seconds".format(time.time() - now))

    return orange_tab
