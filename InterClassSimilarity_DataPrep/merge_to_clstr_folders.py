# Merges Individual Products to Clustered folders:
#   Inputs:
#       Images: A:\IsKnown_Images\Aff_NE_TrainValTest\Bal_v14\[Train|Val|Test]
#       Clusters: InterClassSimilarity\merged_str\merged_classes_[conf_mat|emb_dist|som_purity_impr].csv
#   Outputs:
#       Images: A:\IsKnown_Images\Mrg_A_NE_TrainValTest\Bal_v14\[conf_mat|emb_dist|som_purity_impr]_<cnt>\[Train|Val|Test]\clstrId\*.jpg

import itertools
import pandas as pd
import os
from shutil import copyfile
from Globals.globalvars import Glb, find_files

merge_methods = [ "conf_mat", "emb_dist", "som_purity_impr" ]
class_counts = [20,26,36,38,132,156,162,170,187]

src_dir = os.path.join ( "A:", "IsKnown_Images", "Aff_NE_TrainValTest", "Bal_v14", "Ind-0" ) #\[Train|Val|Test]\barcode
dest_dir = os.path.join ( "A://IsKnown_Images", "Mrg_A_NE_TrainValTest", "Bal_v14" ) #\[conf_mat|emb_dist|som_purity_impr]_<cnt>\[Train|Val|Test]\clstrId

cntr=0
for merge_method,class_count in itertools.product (merge_methods,class_counts):
    #print ("{},{}".format(merge_method,class_count))

    # Load and filter cluster structure for the current class count
    df_clstr_str = pd.read_csv("../InterClassSimilarity/merged_str/merged_classes_{}.csv".format(merge_method), header=0, dtype=str )
    # columns: 'cnt_classes', 'clstr_id', 'product_id', 'product_barcode','product_name'
    df_clstr_str_this_class_cnt = df_clstr_str[ df_clstr_str.cnt_classes==str(class_count)]

    # Copy files from \barcode\* to \clstr_id\*
    for fullfilename in find_files(src_dir,"*.jpg"):
        #print (fullfilename)
        tmp,filename = os.path.split(fullfilename)
        tmp,barcode = os.path.split(tmp)
        _,set_name = os.path.split(tmp)
        #print (fullfilename+".."+set_name+".."+barcode+".."+filename)

        # Lookup cluster ID by barcode
        clstrId = df_clstr_str_this_class_cnt.clstr_id [ df_clstr_str_this_class_cnt.product_barcode==barcode].values[0]

        # Make dir if needed
        dest_dir_full = os.path.join( dest_dir, merge_method+"_"+str(class_count), set_name, clstrId)
        #print(dest_dir_full)
        if not os.path.exists(dest_dir_full):
            os.makedirs(dest_dir_full)

        # copy file
        dest_dir_fullfilename = os.path.join(dest_dir_full, filename)
        copyfile(fullfilename, dest_dir_fullfilename)

        cntr+=1
        #if cntr>1:
        #    break
        if cntr%100==0:
            print ("{} files copied".format(cntr))

