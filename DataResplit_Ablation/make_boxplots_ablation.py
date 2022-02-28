from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

lst_test_accs = []
lst_x_labels = []


# "visibility" architecture, no ablation
metrics_file = r"../Bal_Overfit_Siam_Result/metrics_resnet_vs_isVisible.csv"
df_metrics_no_abl = pd.read_csv(metrics_file)
df_metrics_no_abl = df_metrics_no_abl[ df_metrics_no_abl.model_version=="IsVisible" ]
lst_test_accs.append(df_metrics_no_abl.test_acc.tolist())
#lst_x_labels.append ( np.repeat( "Empty, Invisible\nRemoved", len(df_metrics_no_abl.test_acc)) )
lst_x_labels.append ( "Empty, Invisible\nRemoved" )

# ablation - empty
abl_visible_file = "metrics_visible.csv"
df_metrics_visible = pd.read_csv(abl_visible_file)
lst_test_accs.append(df_metrics_visible.test_acc.tolist())
#lst_x_labels.append ( np.repeat( "Invisible\nRemoved", len(df_metrics_visible.test_acc)) )
lst_x_labels.append ( "Invisible\nRemoved" )

# ablation - invisible
abl_notempty_file = "metrics_notEmpty.csv"
df_metrics_notempty = pd.read_csv(abl_notempty_file)
lst_test_accs.append(df_metrics_notempty.test_acc.tolist())
#lst_x_labels.append ( np.repeat( "Empty\nRemoved", len(df_metrics_notempty.test_acc)) )
lst_x_labels.append ( "Empty\nRemoved" )

plt.boxplot (lst_test_accs, labels=lst_x_labels)
#plt.tick_params(axis='both', which='major', labelsize=14)
#plt.xticks(ticks=(np.arange(len(lst_x_labels)))+1, labels=lst_x_labels, rotation=90)
plt.title ("Test accuracy, ablation study", fontsize=14, fontweight="bold")
#legend = plt.legend(["Q1: <1/4 product area visibility", "BagR: plastic bags with high glare"], loc='lower left', handlelength=0)
#plt.tight_layout()
plt.savefig("testacc_ablation.png")
#plt.show()
plt.close()

