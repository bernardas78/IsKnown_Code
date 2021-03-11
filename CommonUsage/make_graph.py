from Common import conf_mat, hist_top1_known_unknown, roc_top1, roc_prelast, roc_prelast_chosen_any, \
    last_kmeans_analysis, last_svm_analysis, last_dectree_analysis

##################################
### CONFUSION MATRIX
##################################

#model_file="A:\\RetellectModels\\model_20201128_54prekes_acc897_test707.h5"
model_file="A:\\RetellectModels\\model_20210131_54prekes_acc904_test906.h5"

products_names_file="D:\\Retellect\\data-prep\\temp\\prekes.selected.csv"

#data_folder = "C:\\RetellectImages\\Val"
conf_mat_pattern="D:\\Retellect\\errorAnalysis\\ConfMat_{}_{}prekes.png"
#my_conf_mat = conf_mat.Conf_Mat( data_folder=data_folder, model_file=model_file)
#my_conf_mat.make_conf_mat(conf_mat_pattern,products_names_file)


###################################
### LAST LAYER
###################################

train_data_folder = "C:\\RetellectImages\\Train"
val_data_folder = "C:\\RetellectImages\\Val"
test_data_folder = "C:\\RetellectImages\\Test"

#known_data_folder = val_data_folder #TrainValTest - unbalanced
unknown_data_folder = "A:\\RetellectImages\\UnKnown"

hist_pattern = "D:\\Retellect\\errorAnalysis\\Hist_Top1_{}_{}prekes.png"
#my_hist_top1 = hist_top1_known_unknown.Hist_top1 ( known_data_folder=known_data_folder, unknown_data_folder=unknown_data_folder, model_file=model_file )
#my_hist_top1.make_hist_top1 ( hist_pattern=hist_pattern )


last_activations_file_name = "D:\\Retellect\\errorAnalysis\\Last_Activations.h5"
roc_top1_file_pattern = "D:\\Retellect\\errorAnalysis\\Roc_Top1_{}_{}prekes.png"
#my_roc = roc_top1.Roc_top1 (known_data_folder=known_data_folder, unknown_data_folder=unknown_data_folder, model_file=model_file)
#my_roc.make_roc_top1(roc_top1_file_pattern)

###################################
### PRE-LAST LAYER
###################################
roc_prelast_file_pattern = "D:\\Retellect\\errorAnalysis\\Roc_Prelast_{}.png"
prelast_activations_file_name = "D:\\Retellect\\errorAnalysis\\Prelast_Activations.h5"
meansigmas_file_name = "D:\\Retellect\\errorAnalysis\\Prelast_Meansigmas.h5"
distances_file_name = "D:\\Retellect\\errorAnalysis\\Prelast_Dist_From_Top1.csv"

# Calc activations and means/sigmas on train data
#my_prelast_roc = roc_prelast.Roc_prelast (known_data_folder=train_data_folder, unknown_data_folder=unknown_data_folder, model_file=model_file)
#my_prelast_roc.calc_save_prelast_activations(prelast_activations_file_name)
#my_prelast_roc.calc_save_meansigmas_known(prelast_activations_file_name, meansigmas_file_name)

# To get best distance from mean - use validation data
#my_prelast_roc = roc_prelast.Roc_prelast (known_data_folder=val_data_folder, unknown_data_folder=unknown_data_folder, model_file=model_file)
#my_prelast_roc.calc_save_dist_from_top1(meansigmas_file_name, distances_file_name)
#my_prelast_roc.make_roc_prelast(distances_file_name, roc_prelast_file_pattern)


#####################################
### K-MEANS CLUSTER ANALYSIS
#####################################
unknown_balanced_data_folder="A:\\RetellectImages\\UnBalanced" #balanced manually to contain excatly same as validation balanced
val_balanced_data_folder="A:\\RetellectImages\\Balanced\\Val"
#my_lka = last_kmeans_analysis.Last_Kmeans_Analysis(known_data_folder=val_balanced_data_folder, unknown_data_folder=unknown_balanced_data_folder, model_file=model_file)
#my_lka.calc_save_last_activations(last_activations_file_name)
#my_lka.make_clusters_kmeans(last_activations_file_name)

#####################################
### SVM
#####################################
#my_l_svm = last_svm_analysis.Last_Svm_Analysis(None,None,None)
#my_l_svm.make_svm_analysis(last_activations_file_name)

#####################################
### DECISION TREES
#####################################
#my_l_dectree = last_dectree_analysis.Last_Dectree_Analysis(None,None,None)
#my_l_dectree.make_dectree_analysis(last_activations_file_name)


###################################
### CHOSEN ANY: PRE-LAST LAYER: VAL DATA
###################################
distances_chosen_any_file_name = "D:\\Retellect\\errorAnalysis\\Prelast_Dist_From_Chosen_Any.csv"
roc_prelast_chosen_any_file_pattern = "D:\\Retellect\\errorAnalysis\\Roc_Prelast_Chosen_Any_{}.png"
hist_prelast_chosen_any_file_pattern = "D:\\Retellect\\errorAnalysis\\Hist_Prelast_Chosen_Any_{}.png"
conf_mat_prelast_chosen_any_file_pattern = "D:\\Retellect\\errorAnalysis\\Conf_Mat_Prelast_Chosen_Any_{}.png"

# To get best distance from mean - use validation data
#my_prelast_roc_chosen_any = roc_prelast_chosen_any.Roc_prelast_chosen_any (known_data_folder=val_data_folder, model_file=model_file)
#my_prelast_roc_chosen_any.calc_save_dist_from_chosen_any(meansigmas_file_name, distances_chosen_any_file_name,is_categorical=True)
#my_prelast_roc_chosen_any.make_roc_prelast_chosen_any(distances_chosen_any_file_name, roc_prelast_chosen_any_file_pattern,
#                                                      hist_prelast_chosen_any_file_pattern, conf_mat_prelast_chosen_any_file_pattern,
#                                                      threshold_to_use=None)
##                                                     threshold_to_use = thresholds_to_use["95"])
thresholds_to_use = {"95":1.54894}

###################################
### CHOSEN ANY: PRE-LAST LAYER: TEST DATA
###################################
distances_chosen_any_file_name_test = "D:\\Retellect\\errorAnalysis\\Prelast_Dist_From_Chosen_Any_Test.csv"
roc_prelast_chosen_any_file_pattern_test = "D:\\Retellect\\errorAnalysis\\Roc_Prelast_Chosen_Any_{}_Test.png"
hist_prelast_chosen_any_file_pattern_test = "D:\\Retellect\\errorAnalysis\\Hist_Prelast_Chosen_Any_{}_Test.png"
conf_mat_prelast_chosen_any_file_pattern_test = "D:\\Retellect\\errorAnalysis\\Conf_Mat_Prelast_Chosen_Any_{}_Test.png"

my_prelast_roc_chosen_any = roc_prelast_chosen_any.Roc_prelast_chosen_any (known_data_folder=test_data_folder, model_file=model_file)
my_prelast_roc_chosen_any.calc_save_dist_from_chosen_any(meansigmas_file_name, distances_chosen_any_file_name_test,is_categorical=True)
my_prelast_roc_chosen_any.make_roc_prelast_chosen_any(distances_chosen_any_file_name_test, roc_prelast_chosen_any_file_pattern_test,
                                                      hist_prelast_chosen_any_file_pattern_test, conf_mat_prelast_chosen_any_file_pattern_test,
                                                      threshold_to_use=thresholds_to_use["95"])

###################################
### CHOSEN ANY: PRE-LAST LAYER: UNKNOWN DATA
###################################
distances_chosen_any_file_name_unknown = "D:\\Retellect\\errorAnalysis\\Prelast_Dist_From_Chosen_Any_unknown.csv"
roc_prelast_chosen_any_file_pattern_unknown = "D:\\Retellect\\errorAnalysis\\Roc_Prelast_Chosen_Any_{}_unknown.png"
hist_prelast_chosen_any_file_pattern_unknown = "D:\\Retellect\\errorAnalysis\\Hist_Prelast_Chosen_Any_{}_unknown.png"
conf_mat_prelast_chosen_any_file_pattern_unknown = "D:\\Retellect\\errorAnalysis\\Conf_Mat_Prelast_Chosen_Any_{}_unknown.png"

#my_prelast_roc_chosen_any = roc_prelast_chosen_any.Roc_prelast_chosen_any (known_data_folder=unknown_data_folder, model_file=model_file)
#my_prelast_roc_chosen_any.calc_save_dist_from_chosen_any(meansigmas_file_name, distances_chosen_any_file_name_unknown,is_categorical=False)
#my_prelast_roc_chosen_any.make_roc_prelast_chosen_any(distances_chosen_any_file_name_unknown, roc_prelast_chosen_any_file_pattern_unknown,
#                                                      hist_prelast_chosen_any_file_pattern_unknown, conf_mat_prelast_chosen_any_file_pattern_unknown,
#                                                      threshold_to_use=thresholds_to_use["95"])
