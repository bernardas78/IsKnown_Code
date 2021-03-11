# Draw ROC Top 1 preds: known and unknown

from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
df_probs_filename = "top1_preds.csv"
df_probs = pd.read_csv(df_probs_filename)

#known_probs = df_probs.loc [ df_probs["IsKnown"] == "Known" ]["Top1_prob"]
#unknown_probs = df_probs.loc [ df_probs["IsKnown"] == "Unknown" ]["Top1_prob"]

known_values = [1 if known_value=="Known" else 0 for known_value in df_probs["IsKnown"] ]
known_preds = df_probs["Top1_prob"]

(fpr,tpr,thresholds) = roc_curve (y_score=known_preds, y_true=known_values)
roc_auc = auc(fpr, tpr)

# Find best accuracy
accuracy_scores = []
for thresh in thresholds:
    accuracy_scores.append(accuracy_score(known_values, [1 if m > thresh else 0 for m in known_preds]))
best_acc_ind=np.argmax(accuracy_scores)
best_acc = accuracy_scores[best_acc_ind]
threshhold_to_use=thresholds[best_acc_ind]
print ("Threshold to use = {}".format(threshhold_to_use))


plt.figure()
plt.plot(fpr,tpr, color='green', lw=2, label='ROC_AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=0.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Bandoma atpažinti %, kai nežinoma prekė')
plt.ylabel('Bandoma atpažinti %, kai žinoma prekė')
#cnt_class = len ( np.unique(df_distances["actual"]) ) - 1 #1-unknown class
#samples_known = len(np.where (df_distances["actual"]!="")[0])
#samples_unknown = len(np.where (df_distances["actual"]=="")[0])
#plt.title('{} žinomos klasės; {} žinomų prekių, {} nežinomų'.format (cnt_class, samples_known, samples_unknown ) )
plt.legend(loc="lower right")

# Draw a point for best accuracy
plt.plot(fpr[best_acc_ind], tpr[best_acc_ind], marker="s", color="red")
plt.text(fpr[best_acc_ind] + 0.02, tpr[best_acc_ind] - 0.02,
         "Best accuracy: {:.1f}%".format(best_acc * 100))

# Save
plt.savefig(".\\roc.png")

plt.show()
