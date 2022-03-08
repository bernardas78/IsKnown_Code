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
lst_x_labels.append ( "No data re-split" )

# resplit
abl_visible_file = "metrics_data_resplit.csv"
df_metrics_visible = pd.read_csv(abl_visible_file)
lst_test_accs.append(df_metrics_visible.test_acc.tolist())
#lst_x_labels.append ( np.repeat( "Invisible\nRemoved", len(df_metrics_visible.test_acc)) )
lst_x_labels.append ( "Data re-split" )


plt.boxplot (lst_test_accs, labels=lst_x_labels)
plt.title ("Test accuracy, data re-resplit", fontsize=14, fontweight="bold")
plt.savefig("testacc_resplit.png")
#plt.show()
plt.close()

