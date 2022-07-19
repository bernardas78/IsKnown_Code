# Affine transformations of all files in a folder
#   Src: A:\IsKnown_Images\Selected\Ind-0\<sco>\<barcode>\*.jpg
#   Dest: A:\IsKnown_Images\Affine\<barcode>\*.jpg


import sco_four_points
import os
import cv2 as cv
import numpy as np

#src_folder = r"A:\IsKnown_Images\Selected\Ind-0"
#src_folder = r"A:\IsKnown_Images\Selected_TimeBarcodeSplit"Selected_TimeBarcodeSplit
#src_folder = r"A:\IsKnown_Images\Selected"
#dst_folder = r"A:\IsKnown_Images\Affine"

src_folder = r"A:\IsKnown_Images\Selected_s3"
dst_folder = r"A:\IsKnown_Images\Affine_S3"

i=0

# Loop structure <sco>\<barcode>\*.jpg
for sco_lvl in os.listdir(src_folder):
    sco_lvl_full_folder = os.path.join(src_folder,sco_lvl)

    # Original and transformed image locations mapping
    sco_src_pts = sco_four_points.src_pts[sco_lvl]
    #sco_dst_pts = sco_four_points.dst_pts[sco_lvl]
    sco_dst_pts = np.array( [
        [  0.,   0.],
        [255.,   0.],
        [255., 255.],
        [  0., 255.]] ).astype(np.float32)

    # Affine matrix
    M = cv.getPerspectiveTransform(sco_src_pts, sco_dst_pts)

    for barcode_lvl in os.listdir(sco_lvl_full_folder):
        barcode_lvl_folder = os.path.join(sco_lvl_full_folder,barcode_lvl)

        # make folder if not exists (otherwise, cv.imwrite() doesn't write
        dest_barcode_folder = os.path.join(dst_folder, barcode_lvl)
        if not os.path.exists (dest_barcode_folder):
            os.makedirs(dest_barcode_folder)

        for img_filename in os.listdir(barcode_lvl_folder):
            img_fullname = os.path.join(barcode_lvl_folder, img_filename)
            dest_img_fullname = os.path.join(dest_barcode_folder, img_filename)

            #print (img_fullname)

            # Transform and save
            src = cv.imread(img_fullname)
            dst_img = cv.warpPerspective(src, M, (256, 256))
            cv.imwrite(dest_img_fullname, dst_img)

            if i%100==0:
                #print ("dest_img_fullname:" + dest_img_fullname)
                print ("Processed {} files".format(i))
            i+=1

