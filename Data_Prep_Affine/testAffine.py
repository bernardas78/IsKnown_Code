
import cv2 as cv
import numpy as np
import os
from sco_four_points import src_pts
#import four_point_transform

#sample_file_names = [
    #r"A:\S3\photo\UTENA_KUPISKIO\SCO21\991800200000\20220312104804-991800200000-65970475-8-Ilgavaisiai_agurkai_1kg.jpg" ,
#    r"A:\S3\photo\UTENA_KUPISKIO\SCO21\991800200000\20220410155210-991800200000-18728237-7-Ilgavaisiai_agurkai_1kg.jpg"]

root_folder = r"A:\S3\photo"
root_folder = r"A:\IsKnown_Images\Selected_s3"
store_name = "VILNIUS_RYGOS"
pos_name = "SCO25"
barcode = "991800200000"
full_path = os.path.join(root_folder,store_name,pos_name,barcode)
full_path = os.path.join(root_folder,store_name+"_"+pos_name,barcode)

filenames = os.listdir(full_path)
first_file = sorted(filenames)[0]
last_file = sorted(filenames)[-1]
sample_file_names = [first_file, last_file]

for i,sample_file_name in enumerate(sample_file_names):
    src = cv.imread( os.path.join(full_path,sample_file_name))

    #src_pts = np.array([
    #    [37,247],
    #    [360,314],
    #    [580,640],
    #    [75,640]]).astype(np.float32)
    src_pts_this_sco = src_pts [store_name+"_"+pos_name]
    dst = np.array( [
        [  0.,   0.],
        [255.,   0.],
        [255., 255.],
        [  0., 255.]] ).astype(np.float32)

    #four_img =  four_point_transform.four_point_transform(src, four_pts)
    M = cv.getPerspectiveTransform(src_pts_this_sco, dst)
    four_img = cv.warpPerspective(src, M, (256, 256))

    cv.polylines(src,[src_pts_this_sco.astype(np.int)],True,(0,0,255))
    cv.imwrite(r"A:\TestPerspective\lines{}.jpg".format(i), src)

    cv.imwrite(r"A:\TestPerspective\transformed{}.jpg".format(i), four_img)

