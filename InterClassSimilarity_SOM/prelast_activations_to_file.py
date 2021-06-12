from tensorflow.keras.models import load_model
from Globals.globalvars import Glb, Glb_Iterators
from tensorflow.keras.backend import function
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances, rbf_kernel
import pickle
import pandas as pd
import os

#model = load_model( os.path.join( Glb.results_folder, "model_clsf_from_isVisible_20210415_gpu1.h5") ) # 83% test accuracy
#act_filename_pattern = os.path.join( Glb.results_folder, "activations_prelast_clsf_from_isVisible_20210415_gpu1_{}.h5")
#model = load_model( os.path.join( Glb.results_folder, "model_clsf_from_isVisible_20210511_gpu0.h5") ) # Hier1
#act_filename_pattern = os.path.join( Glb.results_folder, "activations_prelast_clsf_from_isVisible_20210511_gpu0_{}.h5")


def put_prelast_act_to_file(model_filename, act_filename_pattern, hier_lvl, set_name, incl_filenames):
    model = load_model(model_filename)
    act_filename = act_filename_pattern.format(set_name, hier_lvl, "filenames" if incl_filenames else "nofilenames")

    # Data iterator
    batch_size = 128

    #set_name = "Test"
    #set_name = "Train"
    #set_name = "Val"

    #hier_lvl = 0
    #hier_lvl = 1
    #hier_lvl = 2
    #hier_lvl = 3
    #hier_lvl = 4

    # which layer is needed?
    #   model.summary()
    prelast_dense_layer = model.layers[-2]  #model.layers[dense_layer_ids[-2]]
    prelast_func_activation = function([model.input], [prelast_dense_layer.output])
    prelast_output_shape = prelast_dense_layer.output_shape[1]


    #data_iterator = Glb_Iterators.get_iterator(os.path.join( Glb.images_folder, "Bal_v14", "Ind-{}".format(hier_lvl), set_name), "div255", batch_size=batch_size)
    data_folder = os.path.join(Glb.images_folder, "Bal_v14", "Ind-{}".format(hier_lvl), set_name)
    print ("Datafolder:{}".format(data_folder))
    if incl_filenames:
        data_iterator = Glb_Iterators.get_iterator_incl_filenames( data_folder=data_folder, batch_size=batch_size, target_size=256)
    else:
        data_iterator = Glb_Iterators.get_iterator (data_folder=data_folder, div255_resnet="div255", batch_size=batch_size, target_size=256, shuffle=True)

    #cntr = 0
    now = datetime.now()
    all_filenames = []

    # Save activations
    #for X,y in data_iterator:
    for cntr, batch_tuple in enumerate( data_iterator ):

        if incl_filenames:
            (X, y, filenames) = batch_tuple
        else:
            (X, y) = batch_tuple

        if cntr==0:
            if incl_filenames:
                cnt_imgs = len(Glb_Iterators.all_filepaths)
            else:
                cnt_imgs = len(data_iterator.classes)
            # Allocate buffer for storing activations and labels
            act_prelast = np.zeros((cnt_imgs, prelast_output_shape), dtype=np.float32)
            lbls = np.zeros((cnt_imgs), dtype=np.int)

        cnt_samples_in_batch = y.shape[0]
        #print ("Batch {}/{}".format(cntr, len(data_iterator)))
        print("Batch {}/{}".format(cntr, Glb_Iterators.len_iterator if incl_filenames else len(data_iterator)))
        act_prelast[ (cntr*batch_size):(cntr*batch_size+cnt_samples_in_batch),:] = prelast_func_activation([X])[0]
        lbls [ (cntr*batch_size):(cntr*batch_size+cnt_samples_in_batch) ] = np.argmax(y, axis=1)
        if incl_filenames:
            all_filenames += filenames
        if not incl_filenames and (cntr+1) >= len(data_iterator):
            break

    print ("Total seconds: {}".format((datetime.now() - now).seconds))

    if incl_filenames:
        pickle.dump( (act_prelast,lbls,all_filenames), open(act_filename, 'wb') )
    else:
        pickle.dump( (act_prelast,lbls), open(act_filename, 'wb') )
