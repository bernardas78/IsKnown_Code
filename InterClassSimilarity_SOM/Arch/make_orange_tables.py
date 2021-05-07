#   Make tables for Orange API to create SOM
#   Input: activation files (made by InterClassSimilarity\adhoc_dist_matrix.py
#   Output: activation files (same info) in Orange table format

import pickle
import Orange
from Orange.data import Domain, DiscreteVariable, ContinuousVariable
import numpy as np
import os

act_filename_pattern = r"a:\IsKnown_Results\activations_prelast_clsf_from_isVisible_20210415_gpu1_{}.h5"
results_folder = r"a:\IsKnown_Results"

#set_name = "Test"
set_name = "Train"

# Load the activation in pickle format
act_filename = act_filename_pattern.format(set_name)
(act_prelast,lbls) = pickle.load(open(act_filename, 'rb'))

# convert to Orange table and save
#   Later used for:
#       a) Here - later in code - SOM clustering
domain = Domain(
            [ContinuousVariable.make("Feat_"+str(i)) for i in np.arange(act_prelast.shape[1])],
            DiscreteVariable.make(name="lbls", values=np.unique(lbls.astype(str) ) ) )
#train_class_indices = np.asarray ( [subcategories.index(train_class) for train_class in train_classes] )
orange_tab = Orange.data.Table.from_numpy( domain=domain, X=act_prelast, Y=lbls.astype(str))


# Save Orange table files
activations_orangeTable_filename = os.path.join (results_folder,"{}_activations_preLast_Orange.tab".format(set_name))
orange_tab.save (activations_orangeTable_filename)