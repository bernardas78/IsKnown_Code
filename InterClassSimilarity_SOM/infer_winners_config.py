from InterClassSimilarity_SOM.infer_winners import infer_winners



dim_size = 15
hier_lvl = 0

set_names = ["Test", "Train", "Val"]
#set_names = ["Train"]
do_predict = True
do_piecharts = True
do_clstr_str = True
do_clstr_dist = True
incl_filenames = True

infer_winners (set_names=set_names, dim_size=dim_size, hier_lvl=hier_lvl, do_predict=do_predict, do_piecharts=do_piecharts, do_clstr_str=do_clstr_str, do_clstr_dist=do_clstr_dist, incl_filenames=incl_filenames)