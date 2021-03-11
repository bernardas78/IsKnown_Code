# Given a model and a images folders: [Known, Unknown], make a histogram of top 1 distributions

from Common import common as cm
from datetime import datetime
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np

class Hist_top1:
    def __init__(self,
                 known_data_folder,     # folder where know class images are located of structure \class\file.[jpg,...]
                 unknown_data_folder,   # folder where unknown class images are located of structure \file.[jpg,...]
                 model_file             # model file location
                 ):

        self.known_data_folder = known_data_folder
        self.unknown_data_folder = unknown_data_folder

        print ("Loading model {}".format(model_file) )
        self.model = load_model(model_file)
        print ("Loaded model" )

        # extract input size
        self.target_size = self.model.layers[0].input_shape[1]


    def make_hist_top1(self, hist_pattern):  # historgam (result). Can contain up to 2 placeholders {} for date/time for generation, #products:

        # Prepare data generators
        known_data_iterator = cm.get_data_iterator( self.known_data_folder, self.target_size, is_categorical=True)
        print ("self.unknown_data_folder: {}".format(self.unknown_data_folder))
        unknown_data_iterator = cm.get_data_iterator( self.unknown_data_folder, self.target_size, is_categorical=False)

        # Get top 1 probabilities
        print ("Getting predictions...")
        now=datetime.now()
        preds_known = cm.get_preds_top1(self.model, known_data_iterator)
        preds_unknown = cm.get_preds_top1(self.model, unknown_data_iterator)
        print ("Predictions done in {} sec".format( (datetime.now()-now).total_seconds() ) )

        # Separate if=1
        eps = 1e-7
        preds_known = np.array(
            [pred_known + 0.01 if pred_known > 1 - eps else pred_known for pred_known in preds_known])
        preds_unknown = np.array(
            [pred_unknown + 0.01 if pred_unknown > 1 - eps else pred_unknown for pred_unknown in preds_unknown])

        plt.hist(preds_known, 200, alpha=0.5, label='known ({} samples)'.format(len(preds_known)))
        plt.hist(preds_unknown, 200, alpha=0.5, label='unknown ({} samples)'.format(len(preds_unknown)))

        plt.title("Top1 probability distribution for Known vs. Unknown")
        plt.legend(loc='upper right')
        # plt.show()
        hist_file = hist_pattern.format(datetime.now().strftime("%Y%m%d %H%M%S"),
                                        known_data_iterator.num_classes)
        plt.savefig(hist_file)

        print ("Hist at: {}".format(hist_file))