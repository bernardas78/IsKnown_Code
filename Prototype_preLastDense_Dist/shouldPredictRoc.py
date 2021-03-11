# Draws ROC on should/shoud not predict:
#   Roc1: aboslute distance from center
#   Reoc2: mahalanobis distance from center

from Prototype_preLastDense_Dist import common as cm
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import math

# Read distances file
df_distances = pd.read_csv( cm.get_distances_filename(), dtype={"actual":np.str, "top1":np.str}).fillna('')
#df_distances = pd.read_csv( cm.get_distances_filename(), dtype={"actual":np.str, "top1":np.str}, na_values="") # {"actual":""})

# Positive = class is known (one of topX classes)
y_true = (df_distances["actual"] != "").astype(np.float64)

# Predicted = normalized distance from top 1 prediction
#y_pred_abs = 1. - df_distances["dist_eucl"] / np.max(df_distances["dist_eucl"])
#y_pred_mah = 1. - np.log(df_distances["dist_mahalanobis"]) / np.log(np.max(df_distances["dist_mahalanobis"]) )
# use sigmoid
y_pred_mah = [ 1. - 1. / (1 + math.exp( -np.log(curr_dist)) ) for curr_dist in df_distances["dist_mahalanobis"]]
y_pred_mah_excl_0 = [ 1. - 1. / (1 + math.exp( -np.log(curr_dist)) ) for curr_dist in df_distances["dist_mahalanobis_excl_0"]]
# cosine is 0-1
y_pred_cosine = [ (1. - curr_dist ) for curr_dist in df_distances["dist_cosine"]]

# Display ROC for absolute
#(fpr,tpr,thresholds) = roc_curve (y_score=y_pred_abs, y_true=y_true)
#roc_auc = auc(fpr, tpr)

# Display ROC for mahalanobis ;
(fpr_mah,tpr_mah,thresholds_mah) = roc_curve (y_score=y_pred_mah, y_true=y_true)
roc_auc_mah = auc(fpr_mah, tpr_mah)

# Display ROC for mahalanobis excluding 0 sigmas;
(fpr_mah_excl_0,tpr_mah_excl_0,thresholds_mah_excl_0) = roc_curve (y_score=y_pred_mah_excl_0, y_true=y_true)
roc_auc_mah_excl_0 = auc(fpr_mah_excl_0, tpr_mah_excl_0)

# Display ROC for cosine
(fpr_cosine,tpr_cosine,thresholds_cosine) = roc_curve (y_score=y_pred_cosine, y_true=y_true)
roc_auc_cosine = auc(fpr_cosine, tpr_cosine)


# Find best accuracy
accuracy_scores = []
for thresh in thresholds_cosine:
    accuracy_scores.append(accuracy_score(y_true, [1 if m > thresh else 0 for m in y_pred_cosine]))
best_acc_ind=np.argmax(accuracy_scores)
best_acc = accuracy_scores[best_acc_ind]
threshhold_to_use=thresholds_cosine[best_acc_ind]
print ("Threshold to use = {}".format(threshhold_to_use))

plt.figure()
plt.plot(fpr_mah, tpr_mah, color='blue', lw=2, label='ROC (Mahalanobis) (area = %0.2f)' % roc_auc_mah)
plt.plot(fpr_mah_excl_0,tpr_mah_excl_0, color='red', lw=2, label='ROC (Mahalanobis no 0) (area = %0.2f)' % roc_auc_mah_excl_0)
plt.plot(fpr_cosine,tpr_cosine, color='green', lw=2, label='ROC (Cosine) (area = %0.2f)' % roc_auc_cosine)
plt.plot([0, 1], [0, 1], color='navy', lw=0.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Bandoma atpažinti %, kai nežinoma prekė')
plt.ylabel('Bandoma atpažinti %, kai žinoma prekė')
cnt_class = len ( np.unique(df_distances["actual"]) ) - 1 #1-unknown class
samples_known = len(np.where (df_distances["actual"]!="")[0])
samples_unknown = len(np.where (df_distances["actual"]=="")[0])
plt.title('{} žinomos klasės; {} žinomų prekių, {} nežinomų'.format (cnt_class, samples_known, samples_unknown ) )
plt.legend(loc="lower right")
#plt.show()

# Draw a point for best accuracy
plt.plot ( fpr_cosine[best_acc_ind], tpr_cosine[best_acc_ind], marker="s", color="red")
plt.text ( fpr_cosine[best_acc_ind]+0.02, tpr_cosine[best_acc_ind]-0.02, "Best accuracy: {:.1f}%".format(best_acc*100) )

# Save
plt.savefig(".\\roc.png")