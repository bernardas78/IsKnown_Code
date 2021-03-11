# Using IsVisible classifiers
#   2 best F-1 models
# Make Datasets from cropped data (each model)

#data_folder = "C:\\EmptyNot\\Train_NotEmptyUndersampled"
#data_folder = "C:\\EmptyNot\\Train"
data_folder = "C:\\EmptyNot\\Val"
#data_folder = "A:\\IsKnown_Images\\Affine"


# filename=sigmoid value (for sorting)
#dest_folder_sigmoid = "A:\\IsKnown_Images\\EmptyNotDebug\\Sigmoid_NotEmptyUndersampled\\"
#dest_folder_sigmoid = "A:\\IsKnown_Images\\EmptyNotDebug\\Sigmoid_Balanced\\"

# model to load from
model_path = r"A:\\IsKnown_Results\\model_emptyNot_20210222_NotEmptyUndersampled.h5"
#model_path = r"A:\\IsKnown_Results\\model_emptyNot_20210222_Balanced.h5"
is_resnet=False

from tensorflow.keras.models import load_model
from pathlib import Path    # recursive list of files
from PIL import Image
import numpy as np
import os
import shutil
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score


def prepareImage (filename, target_size):
	# Load image ( RGB )
    img = Image.open(filename)
    # Crop bottom square frame (assume H>W)
    w, h = img.size
    img = img.crop ((0,h-w,w,h)) if h>w else img
    # Resize to target
    img = img.resize ((target_size,target_size))
    # resnet prepare
    if is_resnet:
        img = resnet_preprocess_input(np.array(img))
    else:
        img = np.array(img) / 255.
        #print ("img[0:2,:,:]: {}".format(img[:2,0,0]))
    return np.stack([img])  # quick way to add a dimension


# Load model
print ("Loading model {}".format(model_path))
model = load_model ( model_path )
if is_resnet:
    target_size = model.layers[0].input_shape[0][1] # for resnet input_shape is [(None, 224, 224, 3)]
else:
    target_size = model.layers[0].input_shape[1] # for isVisible              (None, 256, 256, 3)
#print (model.summary())
print ("Done Loading model. Target size: {}".format(target_size))

# Get list of image files
img_file_lst = [str(thepath) for thepath in Path(data_folder).rglob("*.jpg")]
print ("Total files found: {}".format( len(img_file_lst) ) )

# Counter for files copied and loop
loop_cntr = 0

# Assemble sigmoid values by display
dic_sigmoids = {"Empty": [], "NotEmpty": []}

for img_file in img_file_lst:
    # Get "sigmoid" value and copy
    img_prepped = prepareImage(img_file, target_size)
    sigmoidvalue = model.predict (img_prepped)[0,0] # shape is [m,2], where [:,0]=Empty_prob, [:,1]=NotEmpty_prob

    # Assemble sigmoid values
    EmptyNot = img_file.split("\\")[-2]
    dic_sigmoids[EmptyNot].append(sigmoidvalue)

    # copy file: filename=sigmoid value
    #dest_fullname = os.path.join(dest_folder_sigmoid, '{:0>8}.jpg'.format( int(sigmoidvalue * 1e+7) ) )
    #shutil.copy(img_file, dest_fullname)

    loop_cntr +=1
    if loop_cntr%100==0:
        print ("Files processed {} / {}".format(loop_cntr,len(img_file_lst)))

print ("Total copied {}/{}".format(loop_cntr, len(img_file_lst)))

# Var1: range 0.99-1.00
dic_sigmoids["Empty"] = [np.maximum(0.99,val) for val in dic_sigmoids["Empty"]]
dic_sigmoids["NotEmpty"] = [np.maximum(0.99,val) for val in dic_sigmoids["NotEmpty"]]
plt.hist(x=dic_sigmoids["Empty"],label="Empty", bins=100, alpha = 0.5, range=(0.99,1))
plt.hist(x=dic_sigmoids["NotEmpty"],label="Not Empty", bins=100, alpha = 0.5, range=(0.99,1))

# Var2: range 0-1
#plt.hist(x=dic_sigmoids["Empty"],label="Empty", bins=100, alpha = 0.5)
#plt.hist(x=dic_sigmoids["NotEmpty"],label="Not Empty", bins=100, alpha = 0.5)

plt.ylabel("Count")
plt.xlabel("Sigmoid value (IsEmpty)")
plt.title(data_folder.split("\\")[-1])
plt.legend()
plt.show()
plt.close()

# ROC to find best accuracy
# Display ROC for mahalanobis ;
#Combine empty and not empty
y_score = dic_sigmoids["Empty"]+dic_sigmoids["NotEmpty"]
y_true = np.concatenate( (np.repeat(1,len(dic_sigmoids["Empty"])),np.repeat(0,len(dic_sigmoids["NotEmpty"] ) ) ) )
(fpr, tpr, thresholds) = roc_curve(y_score=y_score, y_true=y_true)

# Find best accuracy
accuracy_scores = []
for thresh in thresholds:
    accuracy_scores.append(accuracy_score(y_true, [1 if m > thresh else 0 for m in y_score]))
best_acc_ind = np.argmax(accuracy_scores)
best_acc = accuracy_scores[best_acc_ind]
threshhold_to_use = thresholds[best_acc_ind]
print("Threshold to use = {}".format(threshhold_to_use))

plt.figure()
#Roc curve
plt.plot(fpr, tpr, color='blue', lw=2)
#point for best acc
plt.plot(fpr[best_acc_ind], tpr[best_acc_ind], marker="s", color="red")
plt.text (x=fpr[best_acc_ind], y=tpr[best_acc_ind], s="@Thr={}, acc={}".format(threshhold_to_use,accuracy_scores[best_acc_ind]) )

plt.show()
