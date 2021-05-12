from tensorflow.keras.models import load_model
from Globals.globalvars import Glb, Glb_Iterators
from tensorflow.keras.backend import function
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances, rbf_kernel
import pickle
import os

model = load_model( os.path.join( Glb.results_folder, "model_clsf_from_isVisible_20210415_gpu1.h5") ) # 83% test accuracy
act_filename_pattern = os.path.join( Glb.results_folder, "activations_prelast_clsf_from_isVisible_20210415_gpu1_{}.h5")

# Data iterator
batch_size = 350

set_name = "Test"
#set_name = "Train"
#set_name = "Val"

#hier_lvl = 0
#hier_lvl = 1
#hier_lvl = 2
#hier_lvl = 3
hier_lvl = 4

# which layer is needed?
#   model.summary()
prelast_dense_layer = model.layers[-2]  #model.layers[dense_layer_ids[-2]]
prelast_func_activation = function([model.input], [prelast_dense_layer.output])
prelast_output_shape = prelast_dense_layer.output_shape[1]


data_iterator = Glb_Iterators.get_iterator(os.path.join( Glb.images_folder, "Bal_v14", "Ind-{}".format(hier_lvl), set_name), "div255", batch_size=batch_size)
# Total number of images
cnt_imgs = len(data_iterator.classes)
cnt_classes = len(data_iterator.class_indices)

act_filename = act_filename_pattern.format(set_name)

# Allocate buffer for storing activations and labels
act_prelast = np.zeros ((cnt_imgs, prelast_output_shape), dtype=np.float32)
lbls = np.zeros ((cnt_imgs), dtype=np.int)

cntr = 0
now = datetime.now()

# Save activations
for X,y in data_iterator:
    cnt_samples_in_batch = y.shape[0]
    print ("Batch {}/{}".format(cntr, len(data_iterator)))
    act_prelast[ (cntr*batch_size):(cntr*batch_size+cnt_samples_in_batch),:] = prelast_func_activation([X])[0]
    lbls [ (cntr*batch_size):(cntr*batch_size+cnt_samples_in_batch) ] = np.argmax(y, axis=1)
    cntr += 1
    if cntr >= len(data_iterator):
        break
print ("Total seconds: {}".format((datetime.now() - now).seconds))
pickle.dump( (act_prelast,lbls), open(act_filename, 'wb') )