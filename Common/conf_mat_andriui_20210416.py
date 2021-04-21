import conf_mat

data_folder = "C:\\RetellectImages\\Val"
model_file="A:\\RetellectModels\\model_20210131_54prekes_acc904_test906.h5"
conf_mat_pattern="D:\\Retellect\\errorAnalysis\\ConfMat_{}_{}prekes.png"
products_names_file="A:\\AK Dropbox\\n20190113 A\\Models\\prekes_54.csv"

my_conf_mat = conf_mat.Conf_Mat( data_folder=data_folder, model_file=model_file)
my_conf_mat.make_conf_mat(conf_mat_pattern,products_names_file)
