# Calc acc, f1 of Hierarchy 1-4 models

import os
import numpy as np
from tensorflow.keras.models import load_model
from Globals.globalvars import Glb, Glb_Iterators
from sklearn.metrics import f1_score,accuracy_score

model_filename = os.path.join(Glb.results_folder, "model_clsf_from_isVisible_20210511_gpu0.h5")  # Hier1-4

for hier_lvl in range(1,5):
    # Prep data, model
    model = load_model(model_filename)
    data_folder = os.path.join( Glb.images_folder, "Bal_v14", "Ind-0", "Test" )
    data_iterator = Glb_Iterators.get_iterator (data_folder, div255_resnet="div255", shuffle=False)

    # Predict
    preds = model.predict(data_iterator)
    pred_classes = np.argmax(preds)
    actual_classes = data_iterator.classes

    # Accuracy, f-score
    acc = accuracy_score(y_true=actual_classes, y_pred=pred_classes)
    f1 = f1_score(y_true=actual_classes, y_pred=pred_classes)
    print ("Model Hier-{}. Acc={}, F1={}".format(hier_lvl,acc,f1))


