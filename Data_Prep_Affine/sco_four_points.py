import numpy as np

src_pts = {}
dst_pts = {}

# sco 21
src_pts["21"] = np.array([
    [9,228],
    [340,215],
    [480,360],
    [108,495]]).astype(np.float32)
dst_pts["21"] = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 200.],
    [  0., 200.]] ).astype(np.float32)

# sco 22
src_pts["22"] = np.array([
    [0,250],
    [325,226],
    [480,381],
    [170,685]]).astype(np.float32)
dst_pts["22"] = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 255.],
    [  0., 255.]] ).astype(np.float32)

# sco 23
#src_pts["23"] = np.array([
#    [23,266],
#    [346,246],
#    [480,386],
#    [123,525]]).astype(np.float32)
#dst_pts["23"] = np.array( [
#    [  0.,   0.],
#    [255.,   0.],
#    [255., 180.],
#    [  0., 180.]] ).astype(np.float32)

# sco 23_1
src_pts["23_1"] = np.array([
    [20,290],
    [345,300],
    [480,480],
    [105,640]]).astype(np.float32)
dst_pts["23_1"] = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 180.],
    [  0., 200.]] ).astype(np.float32)

# sco 23_2
src_pts["23_2"] = np.array([
    [76,270],
    [390,247],
    [480,340],
    [205,640]]).astype(np.float32)
dst_pts["23_2"] = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 130.],
    [  0., 220.]] ).astype(np.float32)

# sco 23_3
src_pts["23_3"] = np.array([
    [0,253],
    [306,247],
    [480,430],
    [122,640]]).astype(np.float32)
dst_pts["23_3"] = np.array( [
    [  10.,   0.],
    [255.,   0.],
    [255., 220.],
    [  0., 230.]] ).astype(np.float32)

# sco 24
#src_pts["24"] = np.array([
#    [0,255],
#    [300,260],
#    [480,465],
#    [80,640]]).astype(np.float32)
#dst_pts["24"] = np.array( [
#    [  10.,   0.],
#    [255.,   0.],
#    [255., 200.],
#    [  0., 250.]] ).astype(np.float32)

# sco 24_1
src_pts["24_1"] = np.array([
    [130,325],
    [430,312],
    [378,639],
    [0,490]]).astype(np.float32)
dst_pts["24_1"] = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 230.],
    [  0., 200.]] ).astype(np.float32)

# sco 24_2
src_pts["24_2"] = np.array([
    [136,279],
    [457,279],
    [370,640],
    [0,443]]).astype(np.float32)
dst_pts["24_2"] = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 230.],
    [  0., 220.]] ).astype(np.float32)

# sco 24_3
src_pts["24_3"] = np.array([
    [0,272],
    [312,267],
    [480,467],
    [96,640]]).astype(np.float32)
dst_pts["24_3"] = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 200.],
    [  0., 210.]] ).astype(np.float32)

# sco 24_4
src_pts["24_4"] = np.array([
    [23,176],
    [352,209],
    [480,402],
    [117,640]]).astype(np.float32)
dst_pts["24_4"] = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 210.],
    [  0., 250.]] ).astype(np.float32)


# sco 24_5
src_pts["24_5"] = np.array([
    [160,197],
    [480,209],
    [478,590],
    [265,640]]).astype(np.float32)
dst_pts["24_5"] = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [130., 255.],
    [  0., 255.]] ).astype(np.float32)

# sco 24_6
src_pts["24_6"] = np.array([
    [62,174],
    [379,208],
    [480,350],
    [160,640]]).astype(np.float32)
dst_pts["24_6"] = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 170.],
    [  0., 250.]] ).astype(np.float32)

# sco 24_7
src_pts["24_7"] = np.array([
    [44,180],
    [369,208],
    [480,366],
    [145,640]]).astype(np.float32)
dst_pts["24_7"] = np.array( [
    [  0.,   0.],
    [255.,   0.],
    [255., 150.],
    [  0., 250.]] ).astype(np.float32)


src_pts["UTENA_KUPISKIO_SCO21"] = np.array([
       [37,247],
       [360,314],
       [580,640],
       [75,640]]).astype(np.float32)

src_pts["UTENA_KUPISKIO_SCO22"] = np.array([
       [10,255],
       [350,280],
       [585,570],
       [150,750]]).astype(np.float32)

src_pts["UTENA_KUPISKIO_SCO23"] = np.array([
       [20,325],
       [360,340],
       [570,580],
       [120,730]]).astype(np.float32)

src_pts["UTENA_KUPISKIO_SCO24"] = np.array([
       [25,250],
       [355,275],
       [600,560],
       [145,750]]).astype(np.float32)

src_pts["UTENA_KUPISKIO_SCO26"] = np.array([
       [20,290],
       [340,335],
       [530,615],
       [75,680]]).astype(np.float32)

src_pts["UTENA_KUPISKIO_SCO27"] = np.array([
       [5,300],
       [340,310],
       [580,595],
       [90,680]]).astype(np.float32)

src_pts["UTENA_KUPISKIO_SCO28"] = np.array([
       [30,330],
       [360,350],
       [570,600],
       [100,710]]).astype(np.float32)

src_pts["VILNIUS_ASANAVICIUTES_SCO21"] = np.array([
       [30,250],
       [370,265],
       [850,820],
       [150,730]]).astype(np.float32)

src_pts["VILNIUS_ASANAVICIUTES_SCO22"] = np.array([
       [20,255],
       [340,260],
       [630,590],
       [160,730]]).astype(np.float32)

src_pts["VILNIUS_ASANAVICIUTES_SCO23"] = np.array([
       [-70,200],
       [270,220],
       [530,530],
       [100,770]]).astype(np.float32)

src_pts["VILNIUS_ASANAVICIUTES_SCO24"] = np.array([
       [110,310],
       [420,380],
       [560,620],
       [130,670]]).astype(np.float32)

src_pts["VILNIUS_JUSTINISKIU_SCO21"] = np.array([
       [20,235],
       [350,225],
       [640,525],
       [200,735]]).astype(np.float32)

src_pts["VILNIUS_JUSTINISKIU_SCO22"] = np.array([
       [10,270],
       [350,240],
       [610,540],
       [200,720]]).astype(np.float32)

src_pts["VILNIUS_JUSTINISKIU_SCO23"] = np.array([
       [0,260],
       [330,260],
       [600,580],
       [160,730]]).astype(np.float32)

src_pts["VILNIUS_JUSTINISKIU_SCO24"] = np.array([
       [50,170],
       [380,200],
       [620,550],
       [170,720]]).astype(np.float32)

src_pts["VILNIUS_JUSTINISKIU_SCO25"] = np.array([
       [180,0],
       [520,0],
       [540,260],
       [170,270]]).astype(np.float32)


src_pts["VILNIUS_RYGOS_SCO21"] = np.array([
       [30,270],
       [360,280],
       [600,580],
       [150,730]]).astype(np.float32)

src_pts["VILNIUS_RYGOS_SCO22"] = np.array([
       [40,270],
       [360,280],
       [600,580],
       [150,730]]).astype(np.float32)

src_pts["VILNIUS_RYGOS_SCO23"] = np.array([
       [30,270],
       [360,280],
       [600,580],
       [150,730]]).astype(np.float32)

src_pts["VILNIUS_RYGOS_SCO24"] = np.array([
       [40,280],
       [370,300],
       [590,580],
       [150,730]]).astype(np.float32)

src_pts["VILNIUS_RYGOS_SCO25"] = np.array([
       [40,280],
       [370,300],
       [590,580],
       [150,730]]).astype(np.float32)
