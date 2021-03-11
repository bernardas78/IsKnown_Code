# Draw 2 histograms of Top 1 preds: known and unknown
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
df_probs_filename = "top1_preds.csv"
df_probs = pd.read_csv(df_probs_filename)

known_probs = df_probs.loc [ df_probs["IsKnown"] == "Known" ]["Top1_prob"]
unknown_probs = df_probs.loc [ df_probs["IsKnown"] == "Unknown" ]["Top1_prob"]

# Separate if=1
eps=1e-7
known_probs = np.array([known_prob+0.01 if known_prob>1-eps else known_prob for known_prob in known_probs])
unknown_probs = np.array([unknown_prob+0.01 if unknown_prob>1-eps else unknown_prob for unknown_prob in unknown_probs])

plt.hist(unknown_probs, 200, alpha=0.5, label='unknown')
plt.hist(known_probs, 200, alpha=0.5, label='known')

plt.title("Top1 probability for known vs. unknown classes")
plt.legend(loc='upper right')
#plt.show()
plt.savefig('hist.png')
