# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:56:18 2020

@author: laramos
"""
import numpy as np
from sklearn.metrics import confusion_matrix

ts = np.load(r"F:\DeepEEG\6_months_final_Results5Stim002pre\Thresholds_LR.npy")


sens = list()
spec = list()
ppv = list()
npv =list()

test = list()
preds = list()

    
for i in range(0,10):
    probas = np.load(r"F:\DeepEEG\6_months_final_Results5Stim002pre\probabilities_LR_"+str(i)+".npy")
    

    t_spec = ts[i]
    y_test = probas[:,1]
    tn, fp, fn, tp = confusion_matrix(y_test, probas[:,0]>t_spec).ravel()    
    
    sens.append(tp/(tp+fn))
    spec.append(tn/(tn+fp))
    ppv.append(tp/(tp+fp))
    npv.append(tn/(tn+fn))
    
    test.append(y_test)
    preds.append(probas[:,0])
    
    
arr = np.concatenate(test)   
