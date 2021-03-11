# Observed:
#   validation accuracy << test accuracy (e.g. 17.5% << 33.7% using model_isvisible_v62_Ind-0_20210213.h5)
#   hypothesis: val set balanced; test unbalanced; maybe mis-predicts samples where original cnt lower?
#
# Process of analysis:
#   get confusion matrix on val set (need to be balanced)
#   get #samples for each class in test-set (not balanced, thus represents actual distribution)
#   make graphs: acc=f(cnt), acc - separate for each class = (tn_class+tp_class) / total

# What get mixed with what?
#   using same conf mat
#   make distances matrix: cell=1/(fp+fn) - the more 2 classes get mixed up, the more similar they are
#   make dendrogram - do similar object get merged first?

model_path = r"A:\IsKnown_Results\model_isvisible_v62_Ind-0_20210213.h5"

val_folder = r"A:\IsKnown_Images\A_Balanced\v62\Ind-0\Val"
test_folder = r"A:\IsKnown_Images\A_Balanced\v62\Ind-0\Test"
# to speed up predictions
val_folder = r"c:\IsKnown_Images_IsVisible\v62\Ind-0\Val"

import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# get counts per class in test set (~actual distribution)
cnt_samples = [len(os.listdir(os.path.join(test_folder,theclass))) for theclass in os.listdir(test_folder)]
cnt_classes = len(cnt_samples)

data_gen = ImageDataGenerator(preprocessing_function=resnet_preprocess_input)
data_iterator = data_gen.flow_from_directory(
    directory=val_folder,
    target_size=(224, 224),
    batch_size=32,
    shuffle=False,
    class_mode='categorical')
actual_classes = data_iterator.classes
class_names = list(data_iterator.class_indices.keys())


predictions_filename=".\\predictions_val_a_balanced.h5"
if not os.path.exists(predictions_filename):
    model = load_model(model_path)

    # get predictions
    predictions = model.predict_generator(data_iterator, steps=len(data_iterator))
    pickle.dump(predictions, open(predictions_filename, 'wb') )
else:
    predictions = pickle.load( open(predictions_filename, 'rb') )

pred_classes = np.argmax(predictions, axis=1)
conf_mat = confusion_matrix(y_true=actual_classes, y_pred=pred_classes)
print ("Val Acc - Sanity check (should be 17.5%): {}".format(accuracy_score(y_true=actual_classes, y_pred=pred_classes)))

# get accuracy for each class in val set
fp_classes = np.zeros((cnt_classes), dtype=float)
fn_classes = np.zeros((cnt_classes), dtype=float)
for theclass in np.arange(cnt_classes):
    tp_thisclass = conf_mat[theclass,theclass]
    fp_thisclass = np.sum(conf_mat[:,theclass]) - tp_thisclass
    fn_thisclass = np.sum(conf_mat[theclass,:]) - tp_thisclass
    fp_classes[theclass] = fp_thisclass
    fn_classes[theclass] = fn_thisclass

    # draw a matching line between FN and FP point
    #plt.vlines(x=cnt_samples[theclass], ymin=np.minimum(fp_thisclass,fn_thisclass), ymax=np.maximum(fp_thisclass,fn_thisclass), )

plt.scatter (x=cnt_samples, y=fn_classes, s=1, marker='x', label='FN')
plt.scatter (x=cnt_samples, y=fp_classes, s=1, marker='x', color='red', label='FP')
plt.legend()
plt.title ("False Positives, Negatives = f (#samples)")

# top 3 FP barcodes
top3_fp = np.argsort(fp_classes)[-3:]
for top_fp in top3_fp:
    top_fp_classname = class_names[top_fp]
    plt.text(x=cnt_samples[top_fp]+5, y=np.maximum(fp_classes[top_fp],fn_classes[top_fp])+10, s= top_fp_classname)
#plt.show()
plt.close() #uncomment to see chart FN,FP=f(#samples)

# show only false negatives
plt.scatter (x=cnt_samples, y=fn_classes, s=1, marker='x', label='FN')
plt.legend()
plt.title ("False Negatives = f (#samples)")

# top 3 FP barcodes
top3_fn = np.argsort(fn_classes)[-3:]
for top_fn in top3_fn:
    top_fn_classname = class_names[top_fn]
    plt.text( x=cnt_samples[top_fn]+5, y=fn_classes[top_fn]+10, s= top_fn_classname)


#plt.show()
plt.close() #uncomment to see chart FN=f(#samples)

# make distance matrix: cell=1/(fp+fn)
eps = 1e-7
#dist_mat = np.zeros((cnt_classes,cnt_classes), dtype=float)
dist_vec = np.zeros( int(cnt_classes*(cnt_classes-1)/2), dtype=float)
k=0
for i in np.arange(cnt_classes-1):
    for j in np.arange(i+1,cnt_classes):
        #dist_mat[i,j] = 1. / (conf_mat [i,j] + conf_mat [j,i] + eps)
        dist_vec[k] = 1. / (conf_mat [i,j] + conf_mat [j,i] + eps)
        k+=1


clstrs = linkage(y=dist_vec, method='single') # method='single' ==> min_dist, i.e. ((mandarinai_1,mandarinai_2),apelsinai) if apelsinai close to any mandarinai
fig = plt.figure(figsize=(60, 10))


df_prods = pd.read_csv ('dendrogramai.csv', header=None, names=["ProductName","ProductCode","Cnt"], dtype=str)
prod_indices_in_df = [ list(df_prods["ProductCode"]).index(class_name) if class_name in list(df_prods["ProductCode"]) else -1 for class_name in class_names ]
labels = [ "{} {} ({})".format(class_names[i], df_prods["ProductName"][prod_index][:10] if prod_index>=0 else "", cnt_samples[i]) for i,prod_index in enumerate(prod_indices_in_df) ]
#labels = [ "{} {} ({})".format(row["ProductCode"], row["ProductName"], row["Cnt"]) for ind,row in df_prods.iterrows()]
#labels = class_names

dn = dendrogram(Z=clstrs, labels=labels, leaf_font_size=6)
#plt.show()
plt.savefig("dendro_mindist_full.png")
plt.close()

