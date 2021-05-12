from Orange.data import Domain, DiscreteVariable, ContinuousVariable
import Orange
import os
import pickle
import time
import numpy as np
from Globals.globalvars import Glb

#act_filename_pattern = "activations_prelast_clsf_from_isVisible_20210415_gpu1_{}.h5"   #Hier0
act_filename_pattern = "activations_prelast_clsf_from_isVisible_20210511_gpu0_{}_hier{}.h5"   #Hier1-4

def loadActivations(set_name, hier_lvl):
    act_filename = os.path.join(Glb.results_folder, act_filename_pattern.format(set_name, hier_lvl) )
    # Load activation tables (~12:17-min train set)
    now = time.time()
    (act_prelast,lbls) = pickle.load(open(act_filename, 'rb'))
    print("Loaded activations in {} seconds".format(time.time() - now))

    domain = Domain(
                [ContinuousVariable.make("Feat_"+str(i)) for i in np.arange(act_prelast.shape[1])],
                DiscreteVariable.make(name="lbls", values=np.unique(lbls.astype(str) ) ) )
    now = time.time()
    orange_tab = Orange.data.Table.from_numpy( domain=domain, X=act_prelast, Y=lbls.astype(str))
    print("Made Orange table from np_arrays in {} seconds".format(time.time() - now))

    return orange_tab
