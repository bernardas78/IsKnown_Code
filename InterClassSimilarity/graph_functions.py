from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
import numpy as np

# find 2 line intersection points
def find_intersections(line1_pts, line2_pts):
    # Must be same length
    assert (len(line1_pts)==len(line2_pts))
    intersection_pts = []
    for i in range(len(line1_pts)-1):
        if line1_pts[i]<=line2_pts[i] and line1_pts[i+1]>=line2_pts[i+1] or \
                line1_pts[i] >= line2_pts[i] and line1_pts[i + 1] <= line2_pts[i + 1] :
            intersection_pts.append(i+0.5)
    return intersection_pts

def find_all_intersection_points (lst_lines):
    intersection_pts = []
    # Find all intersection points
    for i in range(len(lst_lines)-1):
        for j in range(i+1,len(lst_lines)):
            intersection_pts += find_intersections (lst_lines[i], lst_lines[j])
            #intersection_pts += find_intersections (hypot_f1_conf_mat,hypot_f1_emb_dist)
            #intersection_pts += find_intersections (hypot_f1_purity_impr,hypot_f1_emb_dist)
            #intersection_pts += find_intersections (hypot_f1_conf_mat,hypot_f1_purity_impr)
    intersection_pts = np.array(intersection_pts)
    intersection_pts = np.unique (np.sort(intersection_pts) )
    # eliminate 0, all classes
    intersection_pts = intersection_pts[ np.where (intersection_pts>0)[0] ]
    # reverse to math points in accuracy/f1
    #intersection_pts = len(hypot_f1_conf_mat) - intersection_pts
    intersection_pts = len(lst_lines[0]) - intersection_pts
    return intersection_pts


# Function to draw graphs and intersection points
def metric_lines_and_intersections ( lst_lines, lst_labels, y_label, title,
                                     intersection_pts=None ):
    #x = np.arange(test_conf_mat.shape[0],0,-1)
    x = np.arange(len(lst_lines[0]),0,-1)
    y_min = 1.0
    for i in range(len(lst_lines)):
        plt.plot( x, lst_lines[i], label=lst_labels[i] )
        y_min = np.minimum(y_min,np.min(lst_lines[i]))
        #print ("y_min={}".format(y_min))
    plt.xlabel ("Number of classes")
    plt.ylabel (y_label)
    plt.legend(loc="upper right")
    plt.title(title)
    if intersection_pts is not None:
        plt.vlines(intersection_pts, ymin=y_min, ymax=1, linewidth=0.5, color="black")
        for int_pt in intersection_pts:
            plt.text(x=int_pt+0.5, y=y_min+0.1, s=int_pt, rotation=90, fontsize='xx-small')
    plt.show()

# "Smooth" lines by taking average of cnt_neighbors+1+cnt_neighbors
def smooth_lines(lst_lines, cnt_neighbors):
    new_lst_lines = []
    for lst_line in lst_lines:
        new_line = np.copy(lst_line)
        #print("Members1:{}".format(new_line[0:3]))
        for neighbour in range(1, cnt_neighbors+1):
            new_line += np.concatenate( ( lst_line[neighbour:],np.repeat(lst_line[-1],neighbour) ) ) + \
                        np.concatenate( ( np.repeat(lst_line[0],neighbour), lst_line[:-neighbour]))
            #print ("Members:{}".format( new_line[0:3] ))
        new_lst_lines.append(new_line / (2*cnt_neighbors+1) )
    return new_lst_lines

# Maxima points
# Find local maxima
def find_local_maximums(lst_lines):
    extrema_pts = set()
    # Find all intersection points
    for i in range(len(lst_lines)):
        extrema_pts = extrema_pts | (set ( argrelextrema(np.array(lst_lines[i]), np.greater)[0] ))
    # Remove dupes
    #extrema_pts = np.unique(extrema_pts)
    # reverse to math points in accuracy/f1
    extrema_pts = { len(lst_lines[0]) - extrema_pt for extrema_pt in extrema_pts}
    return np.array([ extrema_pt for extrema_pt in extrema_pts ])

# "Smooth" lines by taking average of cnt_neighbors+1+cnt_neighbors
def smooth_lines_weighted(lst_lines, cnt_neighbors, weights_big_small_ratio):
    # norm weights
    weights = np.linspace(weights_big_small_ratio,1,cnt_neighbors+1)
    weights_normed = weights / np.sum( np.concatenate ( [weights, weights[1:] ] ) )

    new_lst_lines = []
    for lst_line in lst_lines:
        new_line = np.copy(lst_line) * weights_normed[0]
        #print("Members1:{}".format(new_line[0:3]))
        for neighbour in range(1, cnt_neighbors+1):
            new_line += (np.concatenate( ( lst_line[neighbour:],np.repeat(lst_line[-1],neighbour) ) ) + \
                         np.concatenate( ( np.repeat(lst_line[0],neighbour), lst_line[:-neighbour])) ) * weights_normed[neighbour]
            #print ("Members:{}".format( new_line[0:3] ))
        new_lst_lines.append(new_line )
    return new_lst_lines