# Given a model and images folders: [Known, Unknown]:
#   1.SVM on [Known, Unknown].
#   2. Measure how well train data (known/unknown) separated: acc, conf mat


import itertools
import pickle
from Common import common as cm
from keras.models import load_model
from os import path
from datetime import datetime
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

class Last_Svm_Analysis:

    def __init__(self,
                 known_data_folder,     # folder where know class images are located of structure \class\file.[jpg,...]
                 unknown_data_folder,   # folder where unknown class images are located of structure \file.[jpg,...]
                 model_file             # model file location
                 ):
        #self.known_data_folder = known_data_folder
        #self.unknown_data_folder = unknown_data_folder
        #self.model_file = model_file
        self.model_loaded = False


    # def calc_save_last_activations(self, last_activations_file_name):
    #     # make sure model is loaded
    #     self.__load_model()
    #
    #     known_data_iterator = cm.get_data_iterator( self.known_data_folder, self.target_size, is_categorical=True)
    #     unknown_data_iterator = cm.get_data_iterator( self.unknown_data_folder, self.target_size, is_categorical=False)
    #
    #     now = datetime.now()
    #     known_preds = cm.get_preds(self.model, known_data_iterator)
    #     print ("Got known predictions in {} sec".format((datetime.now()-now).total_seconds() ))
    #
    #     now = datetime.now()
    #     unknown_preds = cm.get_preds(self.model, unknown_data_iterator)
    #     print ("Got unknown predictions in {} sec".format((datetime.now()-now).total_seconds() ))
    #
    #     act_file = open(last_activations_file_name, 'wb')
    #     pickle.dump( (known_preds,unknown_preds), act_file)
    #     act_file.close()
    #     print("Results saved to file {}".format(last_activations_file_name))
    #     return

    def make_svm_analysis(self, last_activations_file_name):
        #if not path.exists (last_activations_file_name):
        #    self.calc_save_last_activations(last_activations_file_name)
        (known_preds, unknown_preds) =  pickle.load(open(last_activations_file_name, 'rb'))
        print ("Loaded activations: known {} and unknown {}".format (known_preds.shape, unknown_preds.shape))

        # Combine Known+Unknown to a single structure
        last_layer_activations = np.concatenate (  (known_preds,               unknown_preds),                axis=0 )
        labels = np.concatenate ( (np.ones(len(known_preds)), np.zeros(len(unknown_preds))), axis=0)

        kernels = {"poly", "rbf", "sigmoid", "linear"} #
        C_s = {0.9, 1.0, 1.1}

        for kernel, C in itertools.product(kernels, C_s):

            # Fit SVC (takes too long to fit SVC("linear") - use LinearSVC)
            my_svc = LinearSVC(C=C, max_iter=10000) if kernel=="linear" else SVC(kernel=kernel, C=C)
            clf = make_pipeline(StandardScaler(), my_svc)

            print("Kernel={}, C={}".format( kernel, C))
            now=datetime.now()
            clf.fit(last_layer_activations, labels)
            print ("Fit in {} seconds".format( (datetime.now()-now).total_seconds() ) )

            # Get preds
            now=datetime.now()
            predictions = clf.predict(last_layer_activations)
            print ("Predicted in {} seconds".format( (datetime.now()-now).total_seconds()) )

            # Make confusion matrix, get accuracy
            acc=accuracy_score( labels, predictions)
            print ("Accuracy: {}".format(acc))
            conf_mat = confusion_matrix(labels, predictions)
            print (conf_mat)


    ###################################
    ### PRIVATE METHODS
    ###################################

    # def __load_model(self):
    #     if not self.model_loaded:
    #         print ("Loading model {}".format(self.model_file) )
    #         self.model = load_model(self.model_file)
    #         print ("Loaded model" )
    #
    #         # extract input size
    #         self.target_size = self.model.layers[0].input_shape[1]
