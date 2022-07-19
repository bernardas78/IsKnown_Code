from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

metrics_file = r"metrics_resnet_vs_isVisible.csv"

df_metrics = pd.read_csv(metrics_file)

lbl_models = {"EffNet": "EfficientNet\nBO", "Resnet50": "Resnet50", "IsVisible": "Authors'\nVisibility\narchitecture", "autoenc": "Autoencoder\nbased"}
lst_test_accs = []
lst_x_labels = []

# Add results of resplit data
abl_visible_file = "../DataResplit_Ablation/metrics_data_resplit.csv"
df_metrics_visible_resplit = pd.read_csv(abl_visible_file)
lst_test_accs.append(df_metrics_visible_resplit.test_acc.tolist())
lst_x_labels.append ( "Authors'\nVisibility\narchitecture" )

for model_version in np.unique( df_metrics.model_version):
     #if model_version=="EffNet1":     #exclude results prior to resplitting data
     if model_version != "IsVisible":  # exclude results prior to resplitting data
          test_accs = df_metrics.test_acc [df_metrics.model_version==model_version]
          lst_test_accs.append(test_accs)
          lst_x_labels.append(lbl_models[model_version])
          #lst_x_labels.append("a\nb")



plt.boxplot (lst_test_accs, labels=lst_x_labels, showfliers=False)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xticks(ticks=(np.arange(len(lst_x_labels)))+1, labels=lst_x_labels, rotation=90)
#plt.title ("Test accuracy ~ Model architecture", fontsize=14, fontweight="bold")
#legend = plt.legend(["Q1: <1/4 product area visibility", "BagR: plastic bags with high glare"], loc='lower left', handlelength=0)
plt.ylabel("Test Accuracy")
plt.tight_layout()
plt.savefig("testacc_modelarch.png")
#plt.show()
plt.close()

