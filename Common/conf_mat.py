# Given a model and a images folder, make a confusion  matrix

from Common import common as cm
from datetime import datetime
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

class Conf_Mat:
    def __init__(self,
                 data_folder,           # folder where images are located of structure \class\file.[jpg,...]
                 model_file,            # model file location
                 ):

        self.data_folder = data_folder

        print ("Loading model {}".format(model_file) )
        self.model = load_model(model_file)
        print ("Loaded model" )

        # extract input size
        self.target_size = self.model.layers[0].input_shape[1]


    def make_conf_mat(self,
                      conf_mat_pattern,     # confusion matrix (result). Can contain up to 2 placeholders {} for date/time for generation, #products
                      products_names_file   # NULLABLE; csv file w/o header of structure [name,barcode,...]
                      ):

        # Prepare data generator
        data_iterator = cm.get_data_iterator( self.data_folder, self.target_size, is_categorical=True)

        # Predict highest classes
        (y_pred, y_true) = cm.get_pred_actual_classes(self.model, data_iterator)

        # Get product names (folder names are barcodes)
        df_products = None
        if products_names_file is not None:
            df_products = pd.read_csv(products_names_file, header=None, dtype=str)

        # Replace barcodes with product names, if names passed
        prod_names = list(data_iterator.class_indices.keys())
        print("sample barcodes {} (tot: {})".format(prod_names[:2], len(prod_names)))
        if df_products is not None:
            prod_names = [ df_products.loc [ df_products[1]==barcode, 0].values[0] for barcode in prod_names ]
            print("sample products {} (tot: {})".format(prod_names[:2], len(prod_names)))

            # Shorten to 15 characters
            prod_names = [prod[0:15] for prod in prod_names]
            #print (prods_short)

        # result confusion matrix file
        conf_mat_file = conf_mat_pattern.format(datetime.now().strftime("%Y%m%d %H%M%S"), data_iterator.num_classes)

        # When 0 images of certain labels, add 1 manually to avoid badly formatted conf mat
        for lbl in range(len(prod_names)):
            if lbl not in y_true:
                y_true = np.append(y_true, lbl)
                y_pred = np.append(y_pred, lbl)

        # Draw confusion matrix
        plt.figure(figsize=(int(len(prod_names)/15), int(len(prod_names))/15), dpi=80)
        conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
        print ("Shape: {}".format(conf_mat.shape))
        ax = sns.heatmap(conf_mat, annot=True, cbar=False,annot_kws={'size':5}, fmt='g')
        #for t in ax.texts: t.set_text(t.get_text() + " %")


        ax.set_xticks( np.arange(len(prod_names))+0.5 )
        ax.set_yticks( np.arange(len(prod_names))+0.5 )

        #prod_names = ["Product "+str(i) for i in range(len(prod_names))]
        ax.set_yticklabels(prod_names , horizontalalignment='right', rotation = 0, size=5)
        ax.set_xticklabels(prod_names , horizontalalignment='right', rotation = 90, size=5)

        ax.set_xlabel("PREDICTED", weight="bold")#, size=20)
        ax.set_ylabel("ACTUAL", weight="bold")#, size=20)
        plt.tight_layout()
        plt.savefig(conf_mat_file)
        plt.close()

        print ("Conf mat at: {}".format(conf_mat_file))