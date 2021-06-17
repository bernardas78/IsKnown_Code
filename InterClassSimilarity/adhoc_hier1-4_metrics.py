# Calc acc, f1 of Hierarchy 1-4 models

import os
import numpy as np
from tensorflow.keras.models import load_model
from Globals.globalvars import Glb, Glb_Iterators
from sklearn.metrics import f1_score,accuracy_score

model_filenames = {
    0: os.path.join(Glb.results_folder, "model_clsf_from_isVisible_20210415_gpu1.h5"),
    1: os.path.join(Glb.results_folder, "model_clsf_from_isVisible_20210615_gpu0_hier1.h5"),  # Hier1-4
    2: os.path.join(Glb.results_folder, "model_clsf_from_isVisible_20210614_gpu0_hier2.h5"),
    3: os.path.join(Glb.results_folder, "model_clsf_from_isVisible_20210614_gpu0_hier3.h5"),
    4: os.path.join(Glb.results_folder, "model_clsf_from_isVisible_20210614_gpu0_hier4.h5") }

data_folders = {
    0: os.path.join( Glb.images_folder, "Bal_v14", "Ind-0", "Test" ),
    1: os.path.join( Glb.images_folder, "Bal_v14", "Ind-1", "Test" ),
    2: os.path.join( Glb.images_folder, "Bal_v14", "Ind-2", "Test" ),
    3: r"D:\IsKnown_Images\Bal_102030_v14_Ind-3\Ind-3\Test",
    4: r"D:\IsKnown_Images\Bal_102030_v14_Ind-4\Ind-4\Test"
}

for hier_lvl in range(0,1):
    # Prep data, model
    model = load_model(model_filenames[hier_lvl])
    data_folder = data_folders[hier_lvl]
    data_iterator = Glb_Iterators.get_iterator (data_folder, div255_resnet="div255", shuffle=False)

    # Predict
    preds = model.predict(data_iterator)
    pred_classes = np.argmax(preds, axis=1)
    actual_classes = data_iterator.classes

    # Accuracy, f-score
    acc = accuracy_score(y_true=actual_classes, y_pred=pred_classes)
    f1 = f1_score(y_true=actual_classes, y_pred=pred_classes, average="macro")
    print ("Model Hier-{}. Acc={}, F1={}".format(hier_lvl,acc,f1))


