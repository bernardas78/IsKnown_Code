from InterClassSimilarity_SOM.make_som_clusterers import make_som_clusterers

hier_lvl=0
dim_size=15
n_iters=50  # 15x15 grid size 1 iter takes 78 seconds on 1080; takes ~1hour on my home machines

make_som_clusterers(hier_lvl=hier_lvl, dim_size=dim_size,n_iters=n_iters)