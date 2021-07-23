import Globals.globalvars
import  train_multi_versions_single_gpu
import os
import shutil
import pandas as pd
from datetime import date, datetime
from Globals.globalvars import Glb
import py7zr
import boto3

gpu_id = 1

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)


df_traincombinations = pd.read_csv("comb_mergemethod_classcnt_gpu{}.csv".format(gpu_id), header=None, names=["merge_method","cnt_classes"])

for i,row in df_traincombinations.iterrows():
    merge_method, cnt_classes = row.merge_method, row.cnt_classes
    #print ("{} {}".format(merge_method, cnt_classes))


    model_filename = os.path.join(Glb.results_folder,
                                   "model_clsf_from_isVisible_{}_gpu{}_hier1_{}_{}.h5".format(date.today().strftime("%Y%m%d"),
                                                                                         gpu_id, merge_method, cnt_classes))
    lc_filename = os.path.join(Glb.results_folder,
                               "lc_clsf_from_isVisible_{}_gpu{}_hier1_{}_{}.csv".format(date.today().strftime("%Y%m%d"),
                                                                                   gpu_id, merge_method, cnt_classes))

    zip_filename = "{}_{}.7z".format(merge_method,str(cnt_classes) )
    src_zip = os.path.join(Glb.images_folder, "Mrg_A_NE_BalKerasAff102030_ZIP", zip_filename )
    unzip_dest_dir = os.path.join(Glb.images_folder, "Mrg_A_NE_BalKerasAff102030", "Bal_v14")

    data_dir = os.path.join(unzip_dest_dir, "{}_{}".format(merge_method,str(cnt_classes) ) )
    print (model_filename)
    print (lc_filename)
    print (src_zip)
    print (unzip_dest_dir)
    print (data_dir)

    # Download 7z file from S3
    amazon_filename = Globals.globalvars.Glb.amzn_file
    df_amazon_creds = pd.read_csv(amazon_filename, header=0)
    access_key=df_amazon_creds["Access key ID"][0]
    secret_key=df_amazon_creds["Secret access key"][0]
    client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    print ("Downloading .7z...")
    now = datetime.now()
    with open(src_zip,'wb') as f:
        client.download_fileobj('test.bucket.for.test.machine', zip_filename, f)
    print ("Downloaded in {} sec".format((datetime.now()-now).total_seconds()))


    # Unzip 7z file
    print ("Unzipping .7z...")
    now = datetime.now()
    with py7zr.SevenZipFile(src_zip, 'r') as archive:
        archive.extractall(path=unzip_dest_dir)
    print ("Unzipped in {} sec".format((datetime.now()-now).total_seconds()))

    # Delete 7z file
    os.remove (src_zip)

    # Train
    for i in range(1):
        model_clsf = train_multi_versions_single_gpu.train_on_single_gpu (gpu_id=gpu_id,
                                                                          model_filename=model_filename,
                                                                          lc_filename=lc_filename,
                                                                          data_dir=data_dir)

    # Delete Data folder
    print ("Removing Data folder ...")
    now = datetime.now()
    shutil.rmtree(data_dir)
    print ("Removed in {} sec".format((datetime.now()-now).total_seconds()))
