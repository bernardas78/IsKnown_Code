# Displays original+encoded image side by side from validation set
#

from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np

#model_file_path = "A:\\IsKnown_Results\\model_autoenc_20210403.h5"  #Latent 16x16x32
#model_file_path = "A:\\IsKnown_Results\\model_autoenc_20210407.h5"  #Latent 8x8x64
#model_file_path = "A:\\IsKnown_Results\\model_autoenc_20210409_v2.h5"  #Latent 4x4x128
model_file_path = "A:\\IsKnown_Results\\model_autoenc_20210409_v3.h5"  #Latent 8x8x64, BN
data_path = "C:\\AutoEnc_ImgsTmp\\Bal_v14\\Ind-0\\Train\\18002"
#autoenc_show_path = os.environ['GDRIVE'] + "\\PhD_Data\\IsKnow_ErrorAnalysis\\Autoenc"
autoenc_show_path = "a:\\IsKnown_ErrorAnalysis\\Autoenc"

cnt_files_to_display = 50

model = load_model( model_file_path )

for i,filename in enumerate(os.listdir(data_path)):

    image_filepath = os.path.join(data_path, filename)

    img_arr = np.asarray ( Image.open(image_filepath).resize((256,256)) ) /255
    imgs_arr = np.expand_dims (img_arr, axis=0) # need to pass array of images [?,height,width,channels]

    imgs_pred_arr = model.predict(imgs_arr)
    img_pred_arr = imgs_pred_arr[0,:,:,:] # back to [height,width,channels] from [?,height,width,channels]

    # 2 images side by side : original and predicted (endoded->decoded)
    img_both_arr = np.zeros((img_arr.shape[0], img_arr.shape[1] * 2, 3), dtype=np.uint8)
    img_both_arr[:, :img_arr.shape[1], :] = np.round(np.array(img_arr) * 255)
    img_both_arr[:, img_arr.shape[1]:, :] = np.round(img_pred_arr * 255).astype(np.uint8)

    img_both = Image.fromarray(img_both_arr)
    img_both.save( os.path.join(autoenc_show_path, filename) )

    if i>=cnt_files_to_display:
        break