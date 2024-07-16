# 27 Datasets (3 merge methods * 9 numbers_of_classes)
# Actual Metrics vs. hypothetically merged classes metrics
#
# Inputs:
#   mertrics_mrg.csv - results of classification of 27 datasets
#   <hypoth. metrics file>

import pandas as pd
from matplotlib import pyplot as plt

do_include_bc_structure=True

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

language="LT"
#language="EN"

lbl_embedingsdist = "Embeddings distance" if language=='EN' else "Atstumas tarp aktyvacijų"
lbl_errorcontr = "Error contribution" if language=='EN' else "Klaidų įnašas"
lbl_sompurity = "SOM purity" if language=='EN' else "SOM grynumas"
lbl_barcodestructure = "Barcode hierarchy" if language=='EN' else "Hierarchija pagal brūkšninį kodą"

############ ACCURACY GRAPH
xlabel = "Number of classes" if language=='EN' else "Klasių skaičius"
ylabel = "Accuracy" if language=='EN' else "Tikslumas"
title = "Accuracy by merging classes" if language=='EN' else "Tikslumas suliejant klases"

# Actual
plt.scatter (x=class_counts, y=accuracy, c=colors)
# Hypothetical
x = class_cnt
plt.plot(x, hypot_acc_emb_dist, label=lbl_embedingsdist, color=color_codes[labels.index('emb_dist')] )
plt.plot(x, hypot_acc_conf_mat, label=lbl_errorcontr, color=color_codes[labels.index('conf_mat')])
plt.plot(x, hypot_acc_purity_impr, label=lbl_sompurity, color=color_codes[labels.index('som_purity_impr')])
if do_include_bc_structure:
    plt.plot(bc_structure_cnt_classes, bc_structure_acc, label=lbl_barcodestructure, color="red")
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.legend(loc="upper right")
plt.title(title)
#plt.show()
plt.savefig('actual_vs_hypth_merged_acc{}_{}.pdf'.format("_inclBcHier" if do_include_bc_structure else "", language))
plt.close()


############ F_SCORE GRAPH
xlabel = "Number of classes" if language=='EN' else "Klasių skaičius"
ylabel = "F-score" if language=='EN' else "F-rodiklis"
title = "F-score by merging classes" if language=='EN' else "F-rodiklis suliejant klases"

# Actual
plt.scatter (x=class_counts, y=f_1, c=colors)
# Hypothetical
x = class_cnt
plt.plot(x, hypot_f1_emb_dist, label=lbl_embedingsdist, color=color_codes[labels.index('emb_dist')] )
plt.plot(x, hypot_f1_conf_mat, label=lbl_errorcontr, color=color_codes[labels.index('conf_mat')])
plt.plot(x, hypot_f1_purity_impr, label=lbl_sompurity, color=color_codes[labels.index('som_purity_impr')])
if do_include_bc_structure:
    plt.plot(bc_structure_cnt_classes, bc_structure_f1, label=lbl_barcodestructure, color="red")
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.legend(loc="upper right")
plt.title(title)
#plt.show()
plt.savefig('actual_vs_hypth_merged_f1{}_{}.pdf'.format("_inclBcHier" if do_include_bc_structure else "", language))
plt.close()
