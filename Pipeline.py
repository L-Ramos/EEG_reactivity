 
"""
This code is for the EEg project, it contains feature extraction and data pre-processing

The filter fit in python is different from matlab, it is giving me different values from marjolein
@author: laramos
"""
import numpy as np
import os
os.chdir('E:\DeepEEG')
import glob
import Methods_EEG as mt
import Data_Preprocessing as dp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score,confusion_matrix,brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import welch
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.calibration import calibration_curve
from scipy import interp  

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
from sklearn.metrics import auc
from random import shuffle
import pandas as pd

class Measures:       
    def __init__(self, splits):
        self.clf_auc=np.zeros(splits)
        self.clf_brier=np.zeros(splits)
        self.clf_sens=np.zeros(splits)
        self.clf_spec=np.zeros(splits)
        self.clf_tpr=list()
        self.clf_fpr=list()
        self.mean_tpr=0.0
        self.frac_pos_rfc=np.zeros(splits)
        self.run=False

        
def normalize_min_max(x):
    
    max_val=np.max(x,axis=0)
    min_val=np.min(x,axis=0)
    for i in range(x.shape[1]):
        x[:,i]=(x[:,i]-min_val[i])/(max_val[i]-min_val[i])
    return(x)
    
def Combine_datasets(path_data,stimulus,secs,pre_post):
 
    X_final=np.array(pd.read_csv(path_data+'data_'+stimulus[0]+'_'+secs+'_'+pre_post+'.csv'),dtype='float32')
    Y_final=np.array(pd.read_csv(path_data+'label_'+stimulus[0]+'_'+secs+'_'+pre_post+'.csv'),dtype='float32')
    
    for k in range(1,5):
            path_data='E:\\DeepEEG\\data\\'
                
            X=np.array(pd.read_csv(path_data+'data_'+stimulus[k]+'_'+secs+'_'+pre_post+'.csv'),dtype='float32')
            #X2=np.array(pd.read_csv(path_data+'data_'+stimulus[k]+'_'+secs+'_'+'post'+'.csv'),dtype='float32')
            #X=X2-X
            #X=np.array(pd.read_csv(path_data+'databaseline_'+stimulus[k]+'_'+secs+'_'+pre_post+'.csv'),dtype='float32')
                
            Y=np.array(pd.read_csv(path_data+'label_'+stimulus[k]+'_'+secs+'_'+pre_post+'.csv'),dtype='float32')
            X_final=np.concatenate((X_final,X),axis=0)
            Y_final=np.concatenate((Y_final,Y),axis=0)
    return(X_final,Y_final)
    
def Save_fpr_tpr(path_results,names,measures):
    for i in range(0,len(names)): 
        for k in range(0,len(measures[i].clf_fpr)):
            f=np.array(measures[i].clf_fpr[k],dtype='float32')
            t=np.array(measures[i].clf_tpr[k],dtype='float32')
            save_f=path_results+'fpr_'+names[i]+'_'+str(k)
            np.save(save_f,f)
            save_t=path_results+'tpr_'+names[i]+'_'+str(k)
            np.save(save_t,t)           
    
if __name__ == '__main__':
    
    secs='5'
    #secs='10'
    
    stimulus=['Stim001','Stim002','Stim003','Stim004','Stim005']
    #stimulus=['baseline']
    
    #pre_post='pre'
    pre_post='post'
    #layers_nn=[[24,12],[24,30,24],[18,12],[18,36,18],[24,30,12]]
    for k in range(3,len(stimulus)):
        
        path_data="E:\\DeepEEG\\data\\"
            
        path_results="E:\\DeepEEG\\Results_"+secs+stimulus[k]+pre_post+"\\"
        if not os.path.exists(path_results):
             os.makedirs(path_results)
        
        X_pre=np.array(pd.read_csv(path_data+'data_all'+stimulus[k]+'_'+secs+'_'+pre_post+'.csv'),dtype='float32')
        #X_post=np.array(pd.read_csv(path_data+'data'+stimulus[k]+'_'+secs+'_post'+'.csv'),dtype='float32')
        #X=X_post-X_pre 
        #Y=np.array(pd.read_csv(path_data+'label_'+stimulus[k]+'_'+secs+'_'+pre_post+'.csv'),dtype='float32')
        
        X=X_pre
        
        Y=np.array(pd.read_csv(path_data+'label_all'+stimulus[k]+'_'+secs+'_'+pre_post+'.csv'),dtype='float32')
        
        ids=np.array(pd.read_csv(path_data+'IDs_all'+stimulus[k]+'_'+secs+'_'+pre_post+'.csv'),dtype='float32')
        
        label_ids=np.array(pd.read_csv(path_data+'Label_Ids'+stimulus[k]+'_'+secs+'_'+pre_post+'.csv'),dtype='float32')
        
        Y=Y.ravel()
    
        #Y=np.random.permutation(Y)
        
        X=normalize_min_max(X)
        
        splits=100
        cv=10
        
        skf = StratifiedShuffleSplit(n_splits=splits, test_size=0.20,random_state=1)
    
        mean_tprr = 0.0
        
        rfc_m=Measures(splits)
        svm_m=Measures(splits)
        lr_m=Measures(splits)
        nn_m=Measures(splits)
        
        for l, (train, test) in enumerate(skf.split(X, Y)):            
            x_train=X[train]
            x_test=X[test]
            y_train=Y[train]
            y_test=Y[test]
            
    
            #scaler = preprocessing.StandardScaler().fit(x_train)
            #x_train=scaler.transform(x_train)
            #x_test=scaler.transform(x_test) 
                    
            class_rfc=mt.RFC_Pipeline(True,x_train,y_train,x_test,y_test,l,cv,mean_tprr,rfc_m,path_results)   
            class_svm=mt.SVM_Pipeline(True,x_train,y_train,x_test,y_test,l,svm_m,cv,mean_tprr,path_results)   
            class_lr=mt.LR_Pipeline(True,x_train,y_train,x_test,y_test,l,mean_tprr,lr_m)   
            #class_nn=mt.NN_Pipeline(True,x_train,y_train,x_test,y_test,l,nn_m,cv,mean_tprr,layers_nn,path_results) 
            #Naive Bayes
            #KNN
            
        final_m=[rfc_m,svm_m,lr_m]
        final_m=[x for x in final_m if x.run != False]
        names=[class_rfc.name,class_svm.name,class_lr.name]
        names=[x for x in names if x != 'NONE'] 
        mt.Print_Results_Excel(final_m,splits,names,path_results)
        
        Save_fpr_tpr(path_results,names,final_m)
