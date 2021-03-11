
from keras.models import load_model
import keras.layers.core
import numpy as np
from keras.backend import function
import cv2 as cv
import pandas as pd


#model_file = "J:\\AK Dropbox\\n20190113 A\\Models\\model_20200913_8prekes.h5"
#model_file = "J:\\AK Dropbox\\n20190113 A\\Models\\model_20201010_24prekes.h5"
model_file="A:\\RetellectModels\\model_20201128_54prekes_acc897_test707.h5"
#products_file = "J:\\AK Dropbox\\n20190113 A\\Models\\prekes_8classes.csv"
products_file = "J:\\AK Dropbox\\n20190113 A\\Models\\prekes_20201010_24classes.csv"
meansigmas_dic_file = "../API/meansigmas.h5"
distances_filename = ".\\distances.csv"

def get_model():
    print ("Loading model...")
    model = load_model(model_file)
    print ("Loaded model")
    return model

def get_meansigmas_dic_filename():
    return meansigmas_dic_file

def get_distances_filename():
    return distances_filename


# Image preparation function
def prepareImage (filename, target_size):
	# Load image ( BGR )
	img = cv.imread(filename)
	# convert to RGB
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	# Resize to target
	img = cv.resize ( img, (target_size,target_size) )
	# Subtract global dataset average to center pixels to 0
	img = img / 255.
	return img

# Given model, extract pre-last dense layer activations of X
def get_layer_activations(model, img_preped, layer):
    X = np.stack([img_preped])  # quick way to add a dimension
    #print("Getting activations of X shaped ", X.shape)
    func_activation = function([model.input], [layer.output])
    output_activation = func_activation([X])[0]
    return output_activation

# Extract model's input size (int)
def get_target_size(model):
    target_size = model.layers[0].input_shape[1] # shape is (m, height, width, channels), we care only width(assume =height)
    return target_size

# discover pre-last dense layer
def get_prelast_dense(model):
    dense_layer_ids = np.where ([type(layer) is keras.layers.core.Dense for layer in model.layers])[0]
    prelast_dense_layer_id = dense_layer_ids[-2]
    prelast_dense_layer = model.layers[prelast_dense_layer_id]
    return prelast_dense_layer

def get_products():
    return pd.read_csv(products_file, header=None, names=["product_name","barcode"])
