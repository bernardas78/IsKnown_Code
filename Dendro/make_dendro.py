import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import pyplot as plt

#df_prods = pd.read_csv ('dendrogramai.csv', header=None, names=["ProductName","ProductCode","Cnt"])
df_prods = pd.read_csv ('dendrogramai_IsKnownImages_Affine.csv', header=None, names=["ProductName","ProductCode","Cnt"])

cnt_prods = len(df_prods)

mat_dist = np.zeros ((cnt_prods,cnt_prods), dtype=float)
max_cnt = np.max(df_prods["Cnt"])
max_productCode_len = int (np.max ( np.log10(df_prods["ProductCode"]))) + 1

for i in range(cnt_prods):
    for j in np.arange(i+1,cnt_prods):    #only upper triangular is used in scipy.cluster.hierarchy.linkage
        # Distance between higher-level categories is higher than between lower level categories
        #   e.g. diff_cat(1XXXX,3XXXX)=5;  diff_cat(12XXX,13XXX)=4
        # Distance between counts joins less-images-having cats sooner
        #   e.g. diff_cnt (100, 1) = 100+1
        # Total distance: join same cat, less-images-having prods first; then higher level cats
        #   dist = diff_cat + sigmoid (diff_cnt)
        #dist_cat = int ( np.log10( abs( df_prods["ProductCode"][i] - df_prods["ProductCode"][j] ) ) ) + 1 # add 1 so that dist_cat > 1
        productCode_i, productCode_j = str(df_prods["ProductCode"][i]), str(df_prods["ProductCode"][j])
        dist_cat = np.max ( [ 0 if productCode_i[pos]==productCode_j[pos] else max_productCode_len-pos for pos in range(max_productCode_len) ] )
        diff_cnt = (df_prods["Cnt"][i] + df_prods["Cnt"][j] ) / (2*max_cnt+1      )                       # /max_cnt so that dist_cnt < 1
        mat_dist[i,j] = diff_cnt + dist_cat

def get_common_barcode_part_and_cnt (id):
    id=int(id)
    #print ("Calling get_common_barcode_part({})".format(id))
    if id >= cnt_prods:
        clstr_id=id-cnt_prods
        code_left,cnt_imgs_left,cnt_prods_left=get_common_barcode_part_and_cnt (clstrs[clstr_id,0])
        code_right,cnt_imgs_right,cnt_prods_right = get_common_barcode_part_and_cnt (clstrs[clstr_id, 1])
        #print ("left_str {}, right_str {}".format(left_str, right_str))
        code_common = "".join([code_left[pos] if code_left[pos]==code_right[pos] else "_" for pos in range(max_productCode_len) ])
        cnt_imgs_common = cnt_imgs_left + cnt_imgs_right
        cnt_prods_common = cnt_prods_left + cnt_prods_right
        #print ("common: {}".format(common_part))
        return code_common, cnt_imgs_common, cnt_prods_common
    else:
        return str(df_prods["ProductCode"][id]), df_prods["Cnt"][id], 1

# leaf label function: returns common barcode part and total count of images
def llf(id):
    code_common, cnt_common, cnt_prods = get_common_barcode_part_and_cnt(id)
    return "{} ({:3},{:5})".format (code_common, cnt_prods, cnt_common)

clstrs = linkage(y=mat_dist, method='complete') # method='complete' ==> cluster w/ higher max(lowest-cat-imgs) will be joined later

fig = plt.figure(figsize=(30, 10))
labels = [ "{} {} ({})".format(row["ProductCode"], row["ProductName"], row["Cnt"]) for ind,row in df_prods.iterrows()]
dn = dendrogram(Z=clstrs, labels=labels, leaf_font_size=4)
#plt.show()
plt.savefig("dendro_full.png")
plt.close()


fig = plt.figure(figsize=(10, 10))
dn = dendrogram(Z=clstrs, leaf_label_func=llf, p=50, truncate_mode='lastp',orientation='right')
#plt.show()
plt.savefig("dendro_last50.png")
plt.close()


plt.hist(df_prods["Cnt"], bins=100)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title ("Image counts in classes (All)", fontdict={'fontname':'calibri', 'fontsize':16})
plt.xlabel("Image count in class", fontdict={'fontname':'calibri', 'fontsize':16})
plt.ylabel("Number of classes", fontdict={'fontname':'calibri', 'fontsize':16})
plt.tight_layout()
plt.savefig("hist_prekiuFreq.pdf")
#plt.show()
plt.close()

cnts_less_100 = df_prods["Cnt"] [ df_prods["Cnt"] <100 ]
plt.hist(cnts_less_100, bins=100, color='red')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title ("Image counts in classes (<100 images/class)", fontdict={'fontname':'calibri', 'fontsize':16} )
plt.xlabel("Image count in class", fontdict={'fontname':'calibri', 'fontsize':16})
plt.ylabel("Number of classes", fontdict={'fontname':'calibri', 'fontsize':16})
plt.tight_layout()
plt.savefig("hist_prekiuFreq_less100.pdf")
#plt.show()
plt.close()