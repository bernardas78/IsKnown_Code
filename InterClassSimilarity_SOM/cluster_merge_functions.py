import time
import numpy as np
from collections import Counter

def somClusterPurity (pred_winner_neurons, lbls):
    cnt_total = 0
    cnt_winning_class_samples = 0
    for i in range(8):
        for j in range(8):
            #print ("i={}, j={}".format(i,j))
            # filter samples where this is winner neuron
            this_neuron_lbls = lbls [ (pred_winner_neurons[:,0]==i) & (pred_winner_neurons[:,1]==j) ].astype(int)
            #winning_class = Counter(this_neuron_lbls).most_common() [0][0]  #
            most_common_classes = Counter(this_neuron_lbls).most_common()
            cnt_winning_class_samples +=  most_common_classes[0][1] if len(most_common_classes)>0 else 0
            cnt_total += len(this_neuron_lbls)
    # purity formula: (cnt samples in most freq class in each cluster, summed for all clusters) / (cnt total)
    return cnt_winning_class_samples/cnt_total

def hypotheticalMergePurity(pred_winner_neurons, lbls):
    purity = somClusterPurity(pred_winner_neurons, lbls)

    # distance matrix: distance = inverse purity improvement
    purity_impr_mat = np.zeros((194,194), dtype=float)

    now = time.time()
    cntr = 0
    best_i, best_j, best_purity = -1,-1, 0.
    cluster_id = 1000
    for class_i in range(194-1):
        for class_j in range(class_i+1,194):
            lbls_hypo = np.copy(lbls)
            # Replace individual product labels with cluster IDs
            lbls_hypo = np.where( np.isin( lbls_hypo, [class_i, class_j]), cluster_id, lbls_hypo)
            hypot_purity = somClusterPurity (pred_winner_neurons, lbls_hypo)
            purity_impr_mat[class_i, class_j] = hypot_purity-purity
            if hypot_purity>best_purity:
                #print ("better purity found: {}".format(hypot_purity))
                best_i, best_j, best_purity = class_i,class_j,hypot_purity
            if cntr%10==0:
                print ("cntr={}".format(cntr))
                print("Time elapsed: {} sec".format(time.time() - now))
            cntr +=1
    print ("before: {}; after merge({},{}): {}".format( somClusterPurity (pred_winner_neurons, lbls_hypo), best_i, best_j,best_purity) )
    print ("Time elapsed: {} sec".format(time.time()-now))
    return purity_impr_mat