import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

metrics_file = r"metrics_aug20_aug102030_resnet.csv"

df_metrics = pd.read_csv(metrics_file)

lst_test_accs = []
lst_x_labels = []
for aug_version in np.unique( df_metrics.aug_version):
     test_accs = df_metrics.test_acc [df_metrics.aug_version==aug_version]
     lst_test_accs.append(test_accs)
     lst_x_labels.append(aug_version)

matplotlib.rc('font', family='calibri')
plt.boxplot (lst_test_accs)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(ticks=(np.arange(len(lst_x_labels)))+1, labels=lst_x_labels, rotation=90)
plt.title ("Test acc. by affine augmentation parameters", fontdict={'fontname':'calibri', 'fontsize':23})
#plt.ylabel("Test Accuracy")
plt.tight_layout()
plt.savefig("testacc_aug.pdf")
plt.close()
#plt.show()


lst_test_accs_by_visibility = []
lst_x_labels_by_visibility = []
for vis_version in np.unique( df_metrics.isVisible_version):
     for aug_version in np.unique( df_metrics.aug_version):
          test_accs = df_metrics.test_acc [(df_metrics.aug_version==aug_version) & (df_metrics.isVisible_version==vis_version)]
          lst_test_accs_by_visibility.append(test_accs)
          lst_x_labels_by_visibility.append("{}\nvis={}".format(aug_version,vis_version))

plt.boxplot (lst_test_accs_by_visibility)
plt.xticks(ticks=(np.arange(len(lst_x_labels_by_visibility)))+1, labels=lst_x_labels_by_visibility, rotation=90)
plt.title ("Test accuracy ~ Affine augmentation parameters, visibility")
plt.tight_layout()
plt.savefig("testacc_aug_by_visibility.png")
plt.close()
#plt.show()