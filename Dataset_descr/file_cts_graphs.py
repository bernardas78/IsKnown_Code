import pandas as pd
from matplotlib import pyplot as plt

df_prods = pd.read_csv( "file_cnts.csv", header=0 )

# look readme.txt, Product taxonomy
cat_bc = {
    "18": "18",
    "19": "19",
    "21": "21",
    "22": "22_23",
    "23": "22_23"
}

cat_names = {
    "18": "Fresh Fruits and Vegetables",
    "19": "Dried Fruits and Nuts",
    "21": "Candies",
    "22_23": "Buns, Donuts, Bisquits"
}

prod_bc_left2 = [ str(bc)[:2] for bc in df_prods.barcode ]
prod_cat_bc = [ cat_bc[bc_left2] for bc_left2 in prod_bc_left2]
df_prods["prod_cat_names"] = [ cat_names[cat_bc] for cat_bc in prod_cat_bc]

# categories file count
df_prods_cnt_catname = df_prods[['prod_cat_names','filecnt']]
df_prods_cnt_catname_grouped = df_prods_cnt_catname.groupby('prod_cat_names').agg('sum').sort_values('filecnt',ascending=False)
file_cnt_bycat = df_prods_cnt_catname_grouped.filecnt.tolist()
cat_names_bycat = df_prods_cnt_catname_grouped.index.tolist()

plt.pie(file_cnt_bycat, labels=cat_names_bycat, autopct=lambda pct:r"{:.1f}%".format(pct) if pct>2.4 else "", shadow=True, startangle=90,  labeldistance=None)
plt.title ("Image count by category", fontsize=14, fontweight="bold")
plt.legend(loc=[1.0,0])
plt.tight_layout()
plt.savefig('filecnt_by_cat.png', bbox_inches='tight')
plt.close()

# categories products count
df_prods_bcs_catname_grouped = df_prods.groupby('prod_cat_names').agg(['count','sum']).sort_values(('filecnt','sum'),ascending=False)
bc_cnt_bycat = df_prods_bcs_catname_grouped.barcode["count"].tolist()
cat_names_bycat = df_prods_bcs_catname_grouped.index.tolist()

plt.pie(bc_cnt_bycat, labels=cat_names_bycat, autopct=lambda pct:r"{:.1f}%".format(pct) if pct>2.4 else "", shadow=True, startangle=90,  labeldistance=None)
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.title ("Product count by category", fontsize=14, fontweight="bold")
#plt.legend(loc=[0.8,0])
plt.tight_layout()
plt.savefig('bccnt_by_cat.png', bbox_inches='tight')
plt.close()
