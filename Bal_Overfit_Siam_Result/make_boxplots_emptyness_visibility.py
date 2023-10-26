import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

metrics_file = r"metrics_emptyness_visibility.csv"

df_metrics = pd.read_csv(metrics_file)

lst_test_accs = []
lst_x_labels = []
for isVisible_version in np.unique( df_metrics.isVisible_version):
    for emptyness_version in np.unique( df_metrics.emptyness_version):
        test_accs = df_metrics.test_acc [(df_metrics.emptyness_version==emptyness_version) & (df_metrics.isVisible_version==isVisible_version)]
        lst_test_accs.append(test_accs)
        lst_x_labels.append("Vis_"+str(isVisible_version) + " " + emptyness_version)

plt.boxplot (lst_test_accs)
plt.xticks(ticks=(np.arange(len(lst_x_labels)))+1, labels=lst_x_labels, rotation=90)
plt.title ("Test accuracy ~ Emptyness, Visibility")
plt.tight_layout()
plt.savefig("testacc_emptyness_visibility.png")
#plt.show()
plt.close()



vis_labels = {14: "Q1", 62: "Q1,\nBagR"}

lst_test_accs = []
lst_x_labels = []
for isVisible_version in np.unique( df_metrics.isVisible_version):
     test_accs = df_metrics.test_acc [df_metrics.isVisible_version==isVisible_version]
     lst_test_accs.append(test_accs)
     #lst_x_labels.append("Vis_"+str(isVisible_version))
     lst_x_labels.append( vis_labels[isVisible_version] )

matplotlib.rc('font', family='calibri')
plt.boxplot (lst_test_accs, labels=lst_x_labels)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xticks(ticks=(np.arange(len(lst_x_labels)))+1, labels=lst_x_labels, rotation=90)
plt.title ("Test accuracy ~ Visibility", fontdict={'fontname':'calibri', 'fontsize':20})
legend = plt.legend(["Q1: <1/4 product area visibility", "BagR: plastic bags with high glare"], loc='lower left', handlelength=0, fontsize=14)
#plt.ylabel("Test accurracy")
plt.tight_layout()
plt.savefig("testacc_visibility.pdf")
#plt.show()
plt.close()


lst_test_accs = []
lst_x_labels = []
for emptyness_version in np.unique( df_metrics.emptyness_version):
    test_accs = df_metrics.test_acc [df_metrics.emptyness_version==emptyness_version]
    lst_test_accs.append(test_accs)
    lst_x_labels.append(emptyness_version)

matplotlib.rc('font', family='calibri')
plt.boxplot (lst_test_accs)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xticks(ticks=(np.arange(len(lst_x_labels)))+1, labels=lst_x_labels, rotation=90)
plt.title ("Test accuracy ~ Emptyness classifier", fontdict={'fontname':'calibri', 'fontsize':20})
#plt.ylabel("Test Accuracy")
plt.tight_layout()
plt.savefig("testacc_emptyness.pdf")
#plt.show()
plt.close()
