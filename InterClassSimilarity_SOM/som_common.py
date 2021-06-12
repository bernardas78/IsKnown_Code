from Orange.data import Domain, DiscreteVariable, ContinuousVariable
import Orange
import os
import pickle
import time
import numpy as np
from Globals.globalvars import Glb


def loadActivations(set_name, hier_lvl, incl_filenames):
    if hier_lvl==0:
        act_filename_pattern = "activations_prelast_clsf_from_isVisible_20210415_gpu1_{}_hier{}.{}.h5"   #Hier0
    else:
        act_filename_pattern = "activations_prelast_clsf_from_isVisible_20210511_gpu0_{}_hier{}.h5"  # Hier1-4

    act_filename = os.path.join(Glb.results_folder, act_filename_pattern.format(set_name, hier_lvl, "filenames" if incl_filenames else "nofilenames") )
    # Load activation tables (~12:17-min train set)
    now = time.time()
    tuple_contents = pickle.load(open(act_filename, 'rb'))
    # file may or may not contain filenames as last member of tuple
    filenames=None
    if len(tuple_contents) == 3:
        (act_prelast, lbls, filenames) = tuple_contents
    else:
        (act_prelast, lbls) = tuple_contents
    #(act_prelast,lbls) = pickle.load(open(act_filename, 'rb'))
    print("Loaded {} activations in {} seconds".format(set_name, time.time() - now))

    #return (act_prelast, lbls)
    return tuple_contents

    #domain = Domain(
    #            [ContinuousVariable.make("Feat_"+str(i)) for i in np.arange(act_prelast.shape[1])],
    #            DiscreteVariable.make(name="lbls", values=np.unique(lbls.astype(str) ) ) )
    #now = time.time()
    #orange_tab = Orange.data.Table.from_numpy( domain=domain, X=act_prelast, Y=lbls.astype(str))
    #print("Made Orange table from np_arrays in {} seconds".format(time.time() - now))


    #return orange_tab,filenames
    #return orange_tab,None

def makeOrangeTable(act_prelast,lbls):
    domain = Domain(
        [ContinuousVariable.make("Feat_" + str(i)) for i in np.arange(act_prelast.shape[1])],
        DiscreteVariable.make(name="lbls", values=np.unique(lbls.astype(str))))
    now = time.time()
    orange_tab = Orange.data.Table.from_numpy(domain=domain, X=act_prelast, Y=lbls.astype(str))
    print("Made Orange table from np_arrays in {} seconds".format(time.time() - now))
    return orange_tab