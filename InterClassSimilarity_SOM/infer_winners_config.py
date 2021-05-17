from InterClassSimilarity_SOM.infer_winners import infer_winners



dim_size = 14
hier_lvl = 0

set_names = ["Test", "Train", "Val"]
do_predict = False
do_piecharts = True

infer_winners (set_names=set_names, dim_size=dim_size, hier_lvl=hier_lvl, do_predict=do_predict, do_piecharts=do_piecharts)