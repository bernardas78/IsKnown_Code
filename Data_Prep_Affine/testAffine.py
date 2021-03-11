
import cv2 as cv
import numpy as np
import four_point_transform

sample_file_names = [
    r"A:\IsKnown_Images\Selected\Ind-0\21\18002\20201009165511-991800200000-Ilgavaisiai agurkai 1kg-cam1.jpg",
    r"A:\IsKnown_Images\Selected\Ind-0\22\18009\20201013120349-991800900000-Trumpavaisiai agurkai 914cm 1kg-cam1.jpg",
    r"A:\IsKnown_Images\Selected\Ind-0\23\18002\20201008201314-991800200000-Ilgavaisiai agurkai 1kg-cam1.jpg",
    r"A:\IsKnown_Images\Selected\Ind-0\24\18002\20200930152030-991800200000-Ilgavaisiai agurkai 1kg-cam1.jpg" ]

sco_id = 3
sample_file_name = sample_file_names[sco_id]

sco_id = 3
sample_file_name = r"A:\IsKnown_Images\Selected_Debug\23\18465\20201023112615-991846500000-Lenkikos kriaus CLAPSA 60 1kg-cam1.jpg"

new_file_pattern = r"temp\{}_{}___{}_{}___{}_{}.jpg"
orig_file_pattern = r"temp\orig_{}_{}___{}_{}___{}_{}.jpg"


src = cv.imread(sample_file_name)

#srcTri = np.array( [[0, 0],
#                    [src.shape[1] - 1, 0],
#                    [0, src.shape[0] - 1]] ).astype(np.float32)
#dstTri = np.array( [ [0, src.shape[1]*0.33],
#                     [src.shape[1]*0.85, src.shape[0]*0.25],
#                     [src.shape[1]*0.15, src.shape[0]*0.7]] ).astype(np.float32)

srcTri = np.array( [[160, 640],
                    [331, 227],
                    [480, 400] ] ).astype(np.float32)
dstTri = np.array( [ [0, 640],
                     [480, 0],
                     [480, 450] ] ).astype(np.float32)

srcTri = np.array( [[0, 260],
                    [313, 239],
                    [160, 640] ] ).astype(np.float32)
dstTri = np.array( [ [0, 0],
                     [480, 0],
                     [0, 500] ] ).astype(np.float32)

srcTri = np.array( [[195, 240],
                    [85, 527],
                    [472, 376] ] ).astype(np.float32)
dstTri = np.array( [ [300, 0],
                     [0, 320],
                     [480, 320] ] ).astype(np.float32)

# Slytis desine-zem
srcTri = np.array( [[0, 0],
                    [0, 640],
                    [480, 200] ] ).astype(np.float32)
dstTri = np.array( [ [0, 0],
                     [0, 640],
                     [480, 400] ] ).astype(np.float32)


warp_mat = cv.getAffineTransform(srcTri, dstTri)
warp_dst = cv.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))

#print (new_file_name)
#print (srcTri)
#cv.imshow('Affine', warp_dst)
#cv.waitKey()
#pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
dst_pts = dstTri.astype(np.int32).reshape((-1,1,2))
src_pts = srcTri.astype(np.int32).reshape((-1,1,2))

cv.polylines(warp_dst,[dst_pts],True,(0,255,0))
#cv.polylines(warp_dst,[src_pts],True,(255,0,0))
new_file_name = new_file_pattern.format(dstTri[0,0], dstTri[0,1], dstTri[1,0], dstTri[1,1], dstTri[2,0], dstTri[2,1])
cv.imwrite(new_file_name, warp_dst)

#cv.polylines(src,[dst_pts],True,(0,255,0))
#cv.polylines(src,[src_pts],True,(255,0,0))
orig_file_name = orig_file_pattern.format(dstTri[0,0], dstTri[0,1], dstTri[1,0], dstTri[1,1], dstTri[2,0], dstTri[2,1])
cv.imwrite(orig_file_name, src)

new2_file_name = r"temp\tr2.jpg"
src2Tri = np.array( [[80, 0],
                    [430, 52],
                    [240, 320] ] ).astype(np.float32)
dst2Tri = np.array( [ [0, 0],
                     [480, 0],
                     [240, 320] ] ).astype(np.float32)
warp_mat2 = cv.getAffineTransform(src2Tri, dst2Tri)
warp_dst2 = cv.warpAffine(warp_dst, warp_mat2, (warp_dst.shape[1], warp_dst.shape[0]))
cv.imwrite(new2_file_name, warp_dst2)

# sco 22
four_pts = np.array([
    [0,250],
    [325,226],
    [480,381],
    [170,685]]).astype(np.float32)
dst = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 255.],
    [  0., 255.]] ).astype(np.float32)

# sco 21
four_pts = np.array([
    [9,228],
    [340,215],
    [480,360],
    [108,495]]).astype(np.float32)
dst = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 200.],
    [  0., 200.]] ).astype(np.float32)

# sco 23
four_pts = np.array([
    [23,266],
    [346,246],
    [480,386],
    [123,525]]).astype(np.float32)
dst = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 180.],
    [  0., 180.]] ).astype(np.float32)

# sco 24_1
four_pts = np.array([
    [130,325],
    [430,312],
    [378,639],
    [0,490]]).astype(np.float32)
dst = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 230.],
    [  0., 200.]] ).astype(np.float32)

# sco 24_2
four_pts = np.array([
    [136,279],
    [457,279],
    [370,640],
    [0,443]]).astype(np.float32)
dst = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 230.],
    [  0., 220.]] ).astype(np.float32)

# sco 24_3
four_pts = np.array([
    [0,272],
    [312,267],
    [480,467],
    [96,640]]).astype(np.float32)
dst = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 200.],
    [  0., 210.]] ).astype(np.float32)

# sco 24_4
four_pts = np.array([
    [23,176],
    [352,209],
    [480,402],
    [117,640]]).astype(np.float32)
dst = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 210.],
    [  0., 250.]] ).astype(np.float32)


# sco 24_5
four_pts = np.array([
    [160,197],
    [480,209],
    [478,590],
    [265,640]]).astype(np.float32)
dst = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [130., 255.],
    [  0., 255.]] ).astype(np.float32)

# sco 24_6
four_pts = np.array([
    [62,174],
    [379,208],
    [480,350],
    [160,640]]).astype(np.float32)
dst = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 170.],
    [  0., 250.]] ).astype(np.float32)

# sco 24_7
four_pts = np.array([
    [44,180],
    [369,208],
    [480,366],
    [145,640]]).astype(np.float32)
dst = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 150.],
    [  0., 250.]] ).astype(np.float32)


# sco 23_1
four_pts = np.array([
    [20,290],
    [345,300],
    [480,480],
    [105,640]]).astype(np.float32)
dst = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 180.],
    [  0., 200.]] ).astype(np.float32)

# sco 23_2
four_pts = np.array([
    [76,270],
    [390,247],
    [480,340],
    [205,640]]).astype(np.float32)
dst = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 130.],
    [  0., 220.]] ).astype(np.float32)

# sco 23_3
four_pts = np.array([
    [0,253],
    [306,247],
    [480,430],
    [122,640]]).astype(np.float32)
dst = np.array( [
    [  10.,   0.],
    [255.,   0.],
    [255., 220.],
    [  0., 230.]] ).astype(np.float32)

#four_img =  four_point_transform.four_point_transform(src, four_pts)
M = cv.getPerspectiveTransform(four_pts, dst)
four_img = cv.warpPerspective(src, M, (256, 256))


cv.polylines(src,[four_pts.astype(np.int)],True,(0,0,255))
cv.imshow('',src)
cv.waitKey()

cv.imshow('',four_img)
cv.waitKey()
cv.imwrite(r"temp\four_{}.jpg".format(sco_id+1), four_img)

