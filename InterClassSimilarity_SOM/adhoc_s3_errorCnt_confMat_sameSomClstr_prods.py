# kiek klaidu % tarp pomidoru ir jonagold obuoliu (blogiausio SOM clusterio top 2 prekes)

from Globals.globalvars import Glb, Glb_Iterators
from tensorflow.keras.models import load_model
import os
import time
import numpy as np
import pandas as pd
import itertools
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

hier_lvl = 0

# prodcuts to compare (IDs from IsKnown_Results/SOM_Clstr_Str/Train_15x15_hier0/purity_0.197_[11_3].csv)
prods=[57,42]

set_name="Test"
#set_name="Val"

df_prodnames = pd.read_csv("df_prods_s3_148.csv", header=0)["class"].tolist()
df_classes = pd.read_csv("df_prods_s3_148.csv", header=0)["class"].tolist()

model_filename = r"A:\IsKnown_Results\model_clsf_from_isVisible_20220719.h5"
model = load_model(model_filename)

data_folder = os.path.join(r"C:\IsKnown_Images_IsVisible", set_name)
data_iterator = Glb_Iterators.get_iterator(data_folder, div255_resnet="div255", batch_size=64, target_size=256, shuffle=False)
total_classes = len(data_iterator.class_indices)

actual_classes = data_iterator.classes
now=time.time()
preds = model.predict(data_iterator, steps=len(data_iterator))
print ("Predicted in {} sec".format(time.time()-now))
pred_classes = np.argmax(preds,axis=1)

# Sanity check: overall accuracy
acc = len ( np.where (pred_classes==actual_classes)[0] ) / len(actual_classes)
total_errors = len( np.where (pred_classes!=actual_classes)[0] )
print ("{} accuracy: {}. Total errors: {}/{}".format(set_name, acc, total_errors, len(pred_classes)))

# for each pair of products
for prod_i,prod_j in itertools.combinations(prods, 2):
    errors_between_prods = len( np.where ((pred_classes==prod_i) & (actual_classes==prod_j) | (pred_classes==prod_j) & (actual_classes==prod_i))[0] )
    pct_thisErrors_vs_totalErrors = errors_between_prods / total_errors * 100
    prod_i_name,prod_j_name = df_prodnames[prod_i], df_prodnames[prod_j]
    prod_i_class,prod_j_class = df_classes[prod_i], df_classes[prod_j]
    print ("Errors between products [{}({}),{}({})] contribute {}/{} ({} pct)".format (prod_i_name,prod_i_class,prod_j_name,prod_j_class,errors_between_prods,total_errors, pct_thisErrors_vs_totalErrors) )
    print ("---------------------------")

conf_mat = confusion_matrix(y_true=actual_classes, y_pred=pred_classes)
# remove diagonal; sum elements [i,j]+[j,i]; set other to 0
for i in range(total_classes):
    conf_mat[i,i]=0
    for j in range(i+1,total_classes):
        conf_mat[i, j] += conf_mat[j, i]
        conf_mat[j, i] = 0


lst_pct_thisErrors_vs_totalErrors = []
prod_name_pairs = []

# Biggest 21 error contributors
print ("Top 21 error contributors")
#for top_i in range(21):
top_i=0
while np.sum(conf_mat>0):
    ind_ravel = np.argmax(conf_mat)
    inds = np.unravel_index( ind_ravel, (total_classes,total_classes))
    errors_between_prods = conf_mat [inds]
    pct_thisErrors_vs_totalErrors = errors_between_prods / total_errors*100
    prod_i_name,prod_j_name = df_prodnames[inds[0]], df_prodnames[inds[1]]
    prod_i_class,prod_j_class = df_classes[inds[0]], df_classes[inds[1]]
    print ("errors between [{}({}),   {}({})] contribute {}/{} ({} pct)".format (prod_i_name,prod_i_class,prod_j_name,prod_j_class,errors_between_prods,total_errors, pct_thisErrors_vs_totalErrors))
    conf_mat [inds] = 0
    # accum data for pie chart
    lst_pct_thisErrors_vs_totalErrors.append(pct_thisErrors_vs_totalErrors)
    prod_name_pairs.append( "{}/{}".format(prod_i_name,prod_j_name) if top_i<5 else "")
    top_i += 1

# Append other (not top n)
#lst_pct_thisErrors_vs_totalErrors.append( np.sum(conf_mat)/total_errors*100 )
#prod_name_pairs.append("Other")
#fig1, ax1 = plt.subplots()
#plt.pie(lst_pct_thisErrors_vs_totalErrors, labels=prod_name_pairs, autopct='%1.1f%%', shadow=True, startangle=90)
plt.pie(lst_pct_thisErrors_vs_totalErrors, labels=prod_name_pairs, autopct=lambda pct:r"{:.1f}%".format(pct) if pct>2.4 else "", shadow=True, startangle=90)
#plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title ("Biggest error contributors", fontsize=14, fontweight="bold")
plt.tight_layout()


plt.savefig("temp/pie_top10_err_contributors.png")
plt.close()