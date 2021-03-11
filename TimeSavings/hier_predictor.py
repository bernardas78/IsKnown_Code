import pickle
import os
import numpy as np
from matplotlib import pyplot as plt

Visible_versions = [14, 62]
Hier_lvls = np.arange(5)
Extract_sets = ["Val", "Test"]

Time_savings_correct = [ 10, 8, 6, 3, 0.1]
Time_loss_incorrect = 4
time_savings_avg = lambda acc,hier,pct_predicted: (acc*Time_savings_correct[hier] - (1-acc)*Time_loss_incorrect)*pct_predicted

eps = 1e-7

for version in Visible_versions:
    for the_set in Extract_sets:
        for hier in Hier_lvls:

            last_activations_filename = 'A:\\IsKnown_Results\\lastAct_v' + str(version) + '_Ind-' + str(hier) + '_' + the_set + '.h5'
            (actual_classes, pred_classes, activations) = pickle.load(open(last_activations_filename, 'rb'))

            # Predicted class confidence
            max_activations = np.max(activations, axis=1)

            Thresholds = np.linspace(0, 1 - 1 / 1000, 1000)
            acc_over_thr = np.zeros((len(Thresholds)), dtype=np.float)
            time_savings_over_thr = np.zeros((len(Thresholds)), dtype=np.float)

            for i, thres in enumerate(Thresholds):
                #print (thres)
                over_thr_indexes = np.where ( max_activations >= np.quantile(max_activations, thres ) ) [0]
                #print (len(over_thr_indexes))

                over_thr_actual_classes = actual_classes [over_thr_indexes]
                over_thr_pred_classes = pred_classes [over_thr_indexes]

                # calc metrics
                true_cnt = len ( np.where ( over_thr_actual_classes == over_thr_pred_classes )[0] )
                acc_over_thr[i] = true_cnt / len(over_thr_actual_classes)+eps
                #print ("Acc={} @Thr={}".format(acc_over_thr[i], thres))

                # calc time savings
                pct_predicted = len(over_thr_indexes) / len(max_activations)
                time_savings_over_thr[i] = time_savings_avg (acc_over_thr[i], hier, pct_predicted)

            plt.plot(Thresholds, acc_over_thr, label = 'Hier-'+str(hier))
            #plt.plot(Thresholds, time_savings_over_thr, label='Hier-' + str(hier) + '. TS='+str(Time_savings_correct[hier])+'sec')

        plt.xlabel('Top 1 probability threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy = F (Threshold on Top 1). ' + the_set + '. v'+str(version))
        #plt.ylabel('Avg. Time saved')
        #plt.title('Time saved = F (Threshold on Top 1). ' + the_set + '. v'+str(version)+ '. Incorrect=-'+str(Time_loss_incorrect)+'sec')
        plt.legend()

        plt.show()
