from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

metrics_file = r"metrics_aug20_aug102030.csv"

df_metrics = pd.read_csv(metrics_file)

lst_test_accs = []
lst_x_labels = []
for aug_version in np.unique( df_metrics.aug_version):
     test_accs = df_metrics.test_acc [df_metrics.aug_version==aug_version]
     lst_test_accs.append(test_accs)
     lst_x_labels.append(aug_version)

plt.boxplot (lst_test_accs)
plt.xticks(ticks=(np.arange(len(lst_x_labels)))+1, labels=lst_x_labels, rotation=90)
plt.title ("Test accuracy ~ Affine augmentation parameters")
plt.tight_layout()
plt.savefig("testacc_aug.png")
plt.close()
#plt.show()
