import tensorflow.keras as k
import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

class PairSequence (k.utils.Sequence):

    def __init__(self, batch_size=32, target_size=224, is_resnet=True, debug=True,
                 data_folder_pos="C:\\EmptyNot\\Train\\Empty",
                 data_folder_neg="C:\\EmptyNot\\Train\\NotEmpty"):
        self.batch_size = batch_size
        self.target_size = target_size
        self.is_resnet = is_resnet
        self.debug = debug
        self.data_folder_pos = data_folder_pos
        self.data_folder_neg = data_folder_neg

        # read file names
        self.pos_names = os.listdir(self.data_folder_pos)
        self.neg_names = os.listdir(self.data_folder_neg)

        # 1 positive sample always part of pair
        self.anchor_names = np.copy(self.pos_names)
        np.random.shuffle(self.anchor_names)

        # len used to measure how many times to request next() in a single epoch;
        self.len_value = np.ceil ( np.minimum ( len(self.pos_names), len(self.neg_names) ) / batch_size ).astype(int)

        # keep track how many times requested
        self.cntr = 0


    def __getitem__(self, idx):

        # Pair: 1 positive sample and 1 (positive|negative)
        X = [ np.zeros((self.batch_size,self.target_size,self.target_size,3),dtype=float),
              np.zeros((self.batch_size, self.target_size, self.target_size, 3), dtype=float) ]

        # Y = {1 if same class (positive); 0 if different}
        Y = np.zeros( (self.batch_size), dtype=float )

        # make a batch: 50% pos-pos and 50% pos-neg samples
        for sample_id in np.arange(self.batch_size):
            anchor_filename = os.path.join(self.data_folder_pos, self.anchor_names [ self.cntr % len(self.anchor_names) ] )
            if sample_id%2==0:
                non_anchor_filename = os.path.join(self.data_folder_pos, self.pos_names [ self.cntr % len(self.pos_names) ] )
                output_val = 1
            else:
                non_anchor_filename = os.path.join(self.data_folder_neg, self.neg_names [ self.cntr % len(self.neg_names) ] )
                output_val = 0

            # read and prepare images to [[256,256,3],[256,256,3]] format
            X [0][sample_id,:,:,:] = self.prepareImage ( anchor_filename )
            X [1][sample_id,:,:,:] = self.prepareImage ( non_anchor_filename )
            Y [sample_id] = output_val

            self.cntr += 1
        return (X, Y)

    def __len__( self ):
        return self.len_value


    def on_epoch_end(self):
        # randomize file order
        np.random.shuffle (self.pos_names)
        np.random.shuffle(self.neg_names)
        np.random.shuffle(self.anchor_names)

        if self.debug:
            print ("PairSequence.py, on_epoch_end")

    def __del__(self):
        if self.debug:
            print ("PairSequence.py, __del__")

    def prepareImage(self, filename):
        # Load image ( RGB )
        img = Image.open(filename)
        # Crop bottom square frame (assume H>W)
        w, h = img.size
        img = img.crop((0, h - w, w, h)) if h > w else img
        # Resize to target
        img = img.resize((self.target_size, self.target_size))
        # resnet prepare
        if self.is_resnet:
            img = resnet_preprocess_input(np.array(img))
        else:
            img = img / 255.
        # print ("img[0:2,:,:]: {}".format(img[:2,0,0]))
        return img