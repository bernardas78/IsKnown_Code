from InterClassSimilarity_SOM.infer_winners import infer_winners



dim_size = 2
hier_lvl = 4

set_names = ["Test", "Train", "Val"]
do_predict = True
do_piecharts = True

infer_winners (set_names=set_names, dim_size=dim_size, hier_lvl=hier_lvl, do_predict=do_predict, do_piecharts=do_piecharts)