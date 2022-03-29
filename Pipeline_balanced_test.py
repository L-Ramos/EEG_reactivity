 
"""
This code is for the EEg project, it contains feature extraction and data pre-processing

The filter fit in python is different from matlab, it is giving me different values from marjolein
@author: laramos
"""
import numpy as np
import os
os.chdir(r'L:\basic\Personal Archive\L\laramos\DeepEEG\Code')
import glob
import Methods_EEG as mt
#import Data_Preprocessing as dp
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
import sys

class Measures:       
    def __init__(self, splits,num_feats):
        self.clf_auc=np.zeros(splits)
        self.clf_brier=np.zeros(splits)
        self.clf_sens=np.zeros(splits)
        self.clf_spec=np.zeros(splits)
        self.clf_thresholds=np.zeros(splits)
        self.clf_tpr=list()
        self.clf_fpr=list()
        self.mean_tpr=0.0
        self.frac_pos_rfc=np.zeros(splits)
        self.run=False
        self.feat_imp=np.zeros((splits,num_feats))

        
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
            path_data='L:\\basic\\Personal Archive\\L\\laramos\\DeepEEG\\data\\'
                
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

def Read_Data(x,y,k,secs):
    import random
    random.seed(1)
    val=random.randint(1,30)
    
    n_feats=12
    channels=14
    files=list()
    labels=list()
    
    for i in range(0,x.shape[0]):    
        #read the files of one of the patients
        #f=(glob.glob(r"E:\DeepEEG\data\\"+secs+"sec\\frequencyFeatures_fft\\"+x[i]+"_"+stimulus[k]+"*"+pre_post+"Stim.txt"))        
        #f=(glob.glob(r"E:\DeepEEG\data\\"+secs+"sec\\frequencyFeatures_fft\\"+x[i]+"_baseline_epoch"+str(val)+".txt"))        
        f=(glob.glob(r"E:\DeepEEG\data\\"+secs+"sec\\frequencyFeatures_fft\\"+x[i]+"_24h_epoch"+str(val)+".txt"))        
        #append the file names and the same label for each file available
        for j in range(0,len(f)):
            labels.append(y[i])
            files.append(f[j])
    #print(files)            
    #create the data read to read all the values and convertthe albels
    data=np.zeros((len(files),n_feats,channels))
    labels=np.array(labels)
    
    
    for l,file in enumerate(files):
        file_input=open(file)
        lines_input=file_input.readlines()
        for j in range(0,n_feats):
            data[l,j,:]=np.fromstring(lines_input[j], dtype='float32', sep=";").reshape(channels) 
    #average the data over all the channels            
    data=np.mean(data,axis=2) 
    nan_pos=np.argwhere(np.isnan(data))
    #print(nan_pos)
    if nan_pos.size:
        data=np.delete(data,nan_pos[0][0],0)
        labels=np.delete(labels,nan_pos[0][0],0)    
    return(data,labels)
    
    
def Read_Data_Subtract(x,y,k,secs):
    n_feats=12
    channels=14
    files=list()
    files2=list()
    labels=list()
    
    for i in range(0,x.shape[0]):    
        #read the files of one of the patients
        f=(glob.glob(r"L:\\basic\\Personal Archive\\L\\laramos\\DeepEEG\\data\\"+str(secs)+"sec\\frequencyFeatures_fft\\"+x[i]+"_"+stimulus[k]+"*"+"preStim.txt"))        
        f2=(glob.glob(r"L:\\basic\\Personal Archive\\L\\laramos\\DeepEEG\\data\\"+str(secs)+"sec\\frequencyFeatures_fft\\"+x[i]+"_"+stimulus[k]+"*"+"postStim.txt"))      
        #append the file names and the same label for each file available
        for j in range(0,len(f)):
            labels.append(y[i])
            files.append(f[j])
            files2.append(f2[j])
    #create the data read to read all the values and convertthe albels
    data=np.zeros((len(files),n_feats,channels))
    data2=np.zeros((len(files),n_feats,channels))
    labels=np.array(labels)
    
    datatest=np.zeros((n_feats,channels))
    datatest2=np.zeros((n_feats,channels))
    for l,file in enumerate(files):
        file_input=open(file)
        lines_input=file_input.readlines()
        for j in range(0,n_feats):
            #data[l,j,:]=np.fromstring(lines_input[j], dtype='float32', sep=";").reshape(channels)          
            datatest2[j,:]=np.fromstring(lines_input[j], dtype='float32', sep=";").reshape(channels)          
        datatest2=np.mean(datatest2,axis=1)  
    for l,file in enumerate(files2):
        file_input=open(file)
        lines_input=file_input.readlines()
        for j in range(0,n_feats):
            data2[l,j,:]=np.fromstring(lines_input[j], dtype='float32', sep=";").reshape(channels)  
    #average the data over all the channels            
    data=np.mean(data,axis=2) 
    data2=np.mean(data2,axis=2) 
    data=data2-data
    nan_pos=np.argwhere(np.isnan(data))
    #print(nan_pos)
    if nan_pos.size:
        data=np.delete(data,nan_pos[0][0],0)
        labels=np.delete(labels,nan_pos[0][0],0)    

    return(data,labels)
    
if __name__ == '__main__':
    
    secs='5'
    #secs='10'
    
    stimulus=['Stim001','Stim002','Stim003','Stim004','Stim005']
    #stimulus=['baseline']
    
    pre_post='pre'
    #pre_post='post'
    layers_nn=[[12],[24],[36],[24,12],[24,36]]
    
    for k in range(0,1):
        

        path_outcome="L:\\basic\\divd\\knf\\Onderzoek_projecten\\EEG_IC\\DeepEEG\\final_DeepEEGdata\\Outcome_binary_3months_final.txt"
        path_data=r"E:\DeepEEG\data"
            
        path_results="E:\\DeepEEG\\Results\\new_24hrs"+secs+stimulus[k]+pre_post+"\\"
        if not os.path.exists(path_results):
             os.makedirs(path_results)
        
   
        new_label=np.loadtxt("L:\\basic\\Personal Archive\\L\\laramos\\DeepEEG\\data\\Outcome_binary_3months_final.txt")
        names_subj=pd.read_table(path_outcome,delimiter=',')
        
        splits=100
        cv=5
        
        skf = StratifiedShuffleSplit(n_splits=splits, test_size=0.20,random_state=1)
    
        mean_tprr = 0.0
        num_feats=12
        rfc_m=Measures(splits,num_feats)
        svm_m=Measures(splits,num_feats)
        lr_m=Measures(splits,num_feats)
        xgb_m=Measures(splits,num_feats)
        nn_m=Measures(splits,num_feats)
        
        X=names_subj['names']
        Y=names_subj['label']=new_label
        
        for l, (train, test) in enumerate(skf.split(X, Y)): 
      
            x_train=np.array(X[train])
            x_test=np.array(X[test])
            y_train=np.array(Y[train])
            y_test=np.array(Y[test])
            
            x_train,y_train=Read_Data(x_train,y_train,k,secs)
            print(x_train.shape)
            x_test,y_test=Read_Data(x_test,y_test,k,secs)
            print("Reading Data!")
            #x_train,y_train=Read_Data_Subtract(x_train,y_train,k,secs)
            #x_test,y_test=Read_Data_Subtract(x_test,y_test,k,secs)
            print("Data Complete, preparing network!")    
            scaler = preprocessing.StandardScaler().fit(x_train)
            x_train=scaler.transform(x_train)
            x_test=scaler.transform(x_test) 
            
            class_rfc=mt.RFC_Pipeline(True,x_train,y_train,x_test,y_test,l,cv,mean_tprr,rfc_m,path_results,k)   
            class_svm=mt.SVM_Pipeline(True,x_train,y_train,x_test,y_test,l,svm_m,cv,mean_tprr,path_results,k)   
            class_lr=mt.LR_Pipeline(True,x_train,y_train,x_test,y_test,l,mean_tprr,lr_m,path_results,k)
            #class_xgb=mt.XGBoost_Pipeline(False,x_train,y_train,x_test,y_test,l,cv,mean_tprr,xgb_m,path_results)   
            class_nn=mt.NN_Pipeline(True,x_train,y_train,x_test,y_test,l,nn_m,cv,mean_tprr,layers_nn,path_results,k) 
            #Naive Bayes
            #KNN
            np.save(r"L:\basic\Personal Archive\L\laramos\DeepEEG\data\sets\train"+str(l)+".npy",x_train)
            np.save(r"L:\basic\Personal Archive\L\laramos\DeepEEG\data\sets\test"+str(l)+".npy",x_test)
            
        final_m=[rfc_m,svm_m,lr_m,xgb_m,nn_m]
        final_m=[x for x in final_m if x.run != False]
        #names=[class_rfc.name,class_svm.name,class_lr.name,class_xgb.name]
        names=[class_rfc.name,class_svm.name,class_lr.name,class_nn.name]
        names=[x for x in names if x != 'NONE'] 
        mt.Print_Results_Excel(final_m,splits,names,path_results)
        
        Save_fpr_tpr(path_results,names,final_m)

