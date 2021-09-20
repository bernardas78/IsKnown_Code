# 27 Datasets (3 merge methods * 9 numbers_of_classes)
# Actual Metrics vs. hypothetically merged classes metrics
#
# Inputs:
#   mertrics_mrg.csv - results of classification of 27 datasets
#   <hypoth. metrics file>

import pandas as pd
from matplotlib import pyplot as plt

# read Actually merged results file
column_names = ["gpu","datetime","data_dir","test_acc","test_f1"]
df_metrics = pd.read_csv ("metrics_mrg.csv", header=None, names=column_names)
accuracy = df_metrics.test_acc
f_1 = df_metrics.test_f1
class_counts = [int(data_dir.split("_")[-1]) for data_dir in df_metrics.data_dir]
merge_methods = [ "_".join( data_dir.split("/")[-1].split("_")[:-1] ) for data_dir in df_metrics.data_dir]
labels = ['emb_dist','conf_mat','som_purity_impr']
color_codes = ["blue","orange","green"]
colors = [ color_codes[labels.index(merge_method)] for merge_method in merge_methods ]


# read Hypothetically merged metrics file
df_hypoth_merge_metrics = pd.read_csv("metrics_hypoth_merge.csv", header=0)
class_cnt = df_hypoth_merge_metrics.class_cnt
hypot_acc_emb_dist = df_hypoth_merge_metrics.hypot_acc_emb_dist
hypot_f1_emb_dist = df_hypoth_merge_metrics.hypot_f1_emb_dist
hypot_acc_conf_mat = df_hypoth_merge_metrics.hypot_acc_conf_mat
hypot_f1_conf_mat = df_hypoth_merge_metrics.hypot_f1_conf_mat
hypot_acc_purity_impr = df_hypoth_merge_metrics.hypot_acc_purity_impr
hypot_f1_purity_impr = df_hypoth_merge_metrics.hypot_f1_purity_impr

####################### BARCODE STRUCTURE ########################################
bc_structure_cnt_classes = [194,109,26,5,2]
# Values by running adhoc_hier1-4_metrics.py
bc_structure_acc = [0.8303471444568868,
                    0.8355729749906682,
                    0.8266144083613288,
                    0.9759238521836506,
                    0.979096677864875]
bc_structure_f1 = [0.5539615212579857,
                   0.5483381249067844,
                   0.6127574834934619,
                   0.5619988851741187,
                   0.9504490578900224]


############ ACCURACY GRAPH
# Actual
plt.scatter (x=class_counts, y=accuracy, c=colors)
# Hypothetical
x = class_cnt
plt.plot(x, hypot_acc_emb_dist, label="Embeddings distance", color=color_codes[labels.index('emb_dist')] )
plt.plot(x, hypot_acc_conf_mat, label="Error contribution", color=color_codes[labels.index('conf_mat')])
plt.plot(x, hypot_acc_purity_impr, label="SOM purity", color=color_codes[labels.index('som_purity_impr')])
plt.plot(bc_structure_cnt_classes, bc_structure_acc, label="Barcode hierarchy", color="red")
plt.xlabel("Number of classes")
plt.ylabel("Accuracy")
plt.legend(loc="upper right")
plt.title("Accuracy by merging classes")
#plt.show()
plt.savefig('actual_vs_hypth_merged_acc.jpg')
plt.close()


############ F_SCORE GRAPH
# Actual
plt.scatter (x=class_counts, y=f_1, c=colors)
# Hypothetical
x = class_cnt
plt.plot(x, hypot_f1_emb_dist, label="Embeddings distance", color=color_codes[labels.index('emb_dist')] )
plt.plot(x, hypot_f1_conf_mat, label="Error contribution", color=color_codes[labels.index('conf_mat')])
plt.plot(x, hypot_f1_purity_impr, label="SOM purity", color=color_codes[labels.index('som_purity_impr')])
plt.plot(bc_structure_cnt_classes, bc_structure_f1, label="Barcode hierarchy", color="red")
plt.xlabel("Number of classes")
plt.ylabel("F-score")
plt.legend(loc="upper right")
plt.title("F-score by merging classes")
#plt.show()
plt.savefig('actual_vs_hypth_merged_f1.jpg')
plt.close()
