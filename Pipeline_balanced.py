 
"""
This code is for the EEg project, it contains feature extraction and data pre-processing

The filter fit in python is different from matlab, it is giving me different values from marjolein
@author: laramos
"""
import numpy as np
import os
#os.chdir(r'L:\basic\Personal Archive\L\laramos\DeepEEG\Code')
import glob
#import Methods_EEG as mt
import new_methods as mt
#import Data_Preprocessing as dp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler
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
import random
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold

class Measures:       
    def __init__(self, splits,itera):
        self.clf_auc=np.zeros((splits,itera))
        self.clf_brier=np.zeros((splits,itera))
                        
        self.clf_f1_score=np.zeros((splits,itera))
        self.clf_sens=np.zeros((splits,itera))
        self.clf_spec=np.zeros((splits,itera))
        self.clf_ppv=np.zeros((splits,itera))
        self.clf_npv=np.zeros((splits,itera))
        self.clf_fpr_val=np.zeros((splits,itera))
        
        self.sens_f1=np.zeros((splits,itera))
        self.spec_f1=np.zeros((splits,itera))
        self.f1_score_f1=np.zeros((splits,itera))
        self.clf_ppv_f1=np.zeros((splits,itera))
        self.clf_npv_f1=np.zeros((splits,itera))
        
        self.sens_spec=np.zeros((splits,itera))
        self.spec_spec=np.zeros((splits,itera))
        self.f1_score_spec=np.zeros((splits,itera))
        self.clf_ppv_spec=np.zeros((splits,itera))
        self.clf_npv_spec=np.zeros((splits,itera))
        self.clf_fpr_spec=np.zeros((splits,itera))
                    
        self.clf_thresholds=list()
        self.clf_tpr=list()
        self.clf_fpr=list()
        self.mean_tpr=0.0
        self.frac_pos_rfc=np.zeros((splits,itera))
        self.run=False
        self.feat_imp=list() 
        self.feat_imp_lr = list()
        self.probas=list()
        self.preds=list()
        
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


def Read_Data(path_data,x,y,k,secs,miss_pat,val,pre_post):
    
    used_files = list()
    n_feats=12
    channels=14
    files=list()
    labels=list()
    missing=list()

    for i in range(0,x.shape[0]):    
        if x[i] not in miss_pat:

      
        #read the files of one of the patients
            #f=(glob.glob(path_data+"\\"+x[i]+"_"+stimulus[k]+"*"+pre_post+"Stim.txt"))  
            if pre_post=='24_hours':
                f = (path_data+"\\"+x[i]+"_24h_epoch1.txt")
            else:
                f = (path_data+"\\"+x[i]+"_"+stimulus[k]+"_No001_"+pre_post+"Stim.txt")
                
            #f=(glob.glob(r"E:\DeepEEG\data\\"+secs+"sec\\frequencyFeatures_fft\\"+x[i]+"_baseline_epoch"+str(val)+".txt"))        
            #f=(glob.glob(path_data+"\\"+x[i]+"_24h_epoch"+str(val)+".txt"))        
            #f=(glob.glob(path_data+"\\"+x[i]+"_baseline_epoch"+str(val)+".txt"))        
            #append the file names and the same label for each file available
            
            if os.path.isfile(f):
                labels.append(y[i])
                files.append(f)
                used_files.append(x[i])
            else:
                #print(x[i],f)
                missing.append(x[i])
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
    return(data,labels,used_files,missing)



# def Read_Data(path,x,y,k,secs,missing,val,pre_post):

#     used_files = list()
#     n_feats=12
#     channels=14
#     files=list()
#     labels=list()
#     missing=list()
#     for i in range(0,x.shape[0]):    
#         #if x[i] not in missing:
#         if len(x[i])>0:
           
#         #read the files of one of the patients
#             f=(glob.glob(path+"\\"+x[i]+"_"+stimulus[k]+"*"+pre_post+"Stim.txt"))        
#             f = (path+"\\"+x[i]+"_"+stimulus[k]+"*"+pre_post+"Stim.txt")
#             #f=(glob.glob(r"E:\DeepEEG\data\\"+secs+"sec\\frequencyFeatures_fft\\"+x[i]+"_baseline_epoch"+str(val)+".txt"))        
#             #f=(glob.glob(path_data+"\\"+x[i]+"_24h_epoch"+str(val)+".txt"))        
#             #f=(glob.glob(path_data+"\\"+x[i]+"_baseline_epoch"+str(val)+".txt"))        
#             #append the file names and the same label for each file available
#             if len(f)>0:
#                 labels.append(y[i])
#                 files.append(f[0])
#                 used_files.append(x[i])
#             else:
#                 #print(x[i],f)
#                 missing.append(x[i])
#     #print(files)            
#     #create the data read to read all the values and convertthe albels
#     data=np.zeros((len(files),n_feats,channels))
#     labels=np.array(labels)
    
    
#     for l,file in enumerate(files):
#         file_input=open(file)
#         lines_input=file_input.readlines()
#         for j in range(0,n_feats):
#             data[l,j,:]=np.fromstring(lines_input[j], dtype='float32', sep=";").reshape(channels) 
#     #average the data over all the channels            
#     data=np.mean(data,axis=2) 
#     nan_pos=np.argwhere(np.isnan(data))
#     #print(nan_pos)
#     if nan_pos.size:
#         data=np.delete(data,nan_pos[0][0],0)
#         labels=np.delete(labels,nan_pos[0][0],0)    
#     return(data,labels,used_files,missing)
    
    
def Read_Data_Subtract(path,x,y,k,secs):
    n_feats=12
    channels=14
    files=list()
    files2=list()
    labels=list()
    
    for i in range(0,x.shape[0]):    
        #read the files of one of the patients
        f=(glob.glob(path+"\\"+x[i]+"_"+stimulus[k]+"*"+"preStim.txt"))        
        f2=(glob.glob(path+"\\"+x[i]+"_"+stimulus[k]+"*"+"postStim.txt"))      
        #append the file names and the same label for each file available
        for j in range(0,len(f)):
            labels.append(y[i])
            files.append(f[j])
            files2.append(f2[j])
    #create the data read to read all the values and convertthe albels
    data=np.zeros((len(files),n_feats,channels))
    data2=np.zeros((len(files),n_feats,channels))
    labels=np.array(labels)
    
    
    for l,file in enumerate(files):
        file_input=open(file)
        lines_input=file_input.readlines()
        for j in range(0,n_feats):
            data[l,j,:]=np.fromstring(lines_input[j], dtype='float32', sep=";").reshape(channels)          
            
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
    
    
   
def Read_Data_Subtract_Single(x,y,k,secs):
    val=random.randint(1,3)
    n_feats=12
    channels=14
    files=list()
    files2=list()
    labels=list()
    
    for i in range(0,x.shape[0]):    
        #read the files of one of the patients
        f=(glob.glob(r"E:\DeepEEG\data\\"+str(secs)+"sec\\frequencyFeatures_fft\\"+x[i]+"_"+stimulus[k]+"_No00"+str(val)+"_preStim.txt"))        
        f2=(glob.glob(r"E:\DeepEEG\data\\"+str(secs)+"sec\\frequencyFeatures_fft\\"+x[i]+"_"+stimulus[k]+"_No00"+str(val)+"_postStim.txt"))      
        #append the file names and the same label for each file available
        for j in range(0,len(f)):
            if len(f)>0:
                labels.append(y[i])
                files.append(f[j])
                files2.append(f2[j])
        #else:
        #    print(x[i])
    #create the data read to read all the values and convertthe albels
    data=np.zeros((len(files),n_feats,channels))
    data2=np.zeros((len(files),n_feats,channels))
    labels=np.array(labels)
    
    
    for l,file in enumerate(files):
        file_input=open(file)
        lines_input=file_input.readlines()
        for j in range(0,n_feats):
            data[l,j,:]=np.fromstring(lines_input[j], dtype='float32', sep=";").reshape(channels)          
            
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

def remove_outliers(x,y,x_orig):
    
    #sns.boxplot(x=X_train[:,0])
    #sns.boxplot(x=X_train[:,1])
    #sns.boxplot(x=X_train[:,2])
    #sns.boxplot(x=X_train[:,3])
    #sns.boxplot(x=X_train[:,4])
    #sns.boxplot(x=X_train[:,5])
    #sns.boxplot(x=X_train[:,6])
    
    #sns.boxplot(x=X_train[:,11])
    #sns.boxplot(x=X_test[:,11])
    
    s = np.zeros((x.shape),dtype='bool')
    
    s[:,0] = x[:,0]>10000
    s[:,1] = x[:,1]>5000
    s[:,2] = x[:,2]>2500
    s[:,3] = x[:,3]>2500
    s[:,4] = x[:,4]>5000
    s[:,10] = x[:,10]>50
    
    
    # x_0 = x[s[:,0]==False,0]
    # x_1 = x[s[:,1]==False,1]
    # x_2 = x[s[:,2]==False,2]
    # x_3 = x[s[:,3]==False,3]
    # x_4 = x[s[:,4]==False,4]
    # x_10 = x[s[:,10]==False,10]
    
    
    # sns.boxplot(x = x_0)
    # sns.boxplot(x = x_1)
    # sns.boxplot(x = x_2)
    # sns.boxplot(x = x_3)
    # sns.boxplot(x = x_4)
    # sns.boxplot(x = x_10)
    
    
    x[s==True] = np.nan
    x = pd.DataFrame(x)
    x['y'] = y
    x['id'] = x_orig
    x = x.dropna()
    y = np.array(x['y'],dtype='int16')
    x_orig = x['id']
    x = x.drop(['y','id'],axis=1)
    
    return(np.array(x),y,x_orig,x)




def read_all_stim(path, secs, stimulus, pre_post, fold,train_test):
     
     path_results = path+secs+stimulus[0]+pre_post+"\\"
     X_train0 = np.load(path_results+'x_'+train_test+'_'+stimulus[0]+str(fold)+".npy")      
     Y_train0 = np.load(path_results+'y_'+train_test+'_'+stimulus[0]+str(fold)+".npy")
     
     for i in range(1,5):
         path_results = path+secs+stimulus[i]+pre_post+"\\"
         X_train1 = np.load(path_results+'x_'+train_test+'_'+stimulus[i]+str(fold)+".npy")      
         Y_train1 = np.load(path_results+'y_'+train_test+'_'+stimulus[i]+str(fold)+".npy")
         X_train0 = np.concatenate((X_train0,X_train1),axis=0)
         Y_train0 = np.concatenate((Y_train0,Y_train1),axis=0)
     return X_train0, Y_train0


    
if __name__ == '__main__':
    
    secs='5'
    #secs='10'
    
    #stimulus=['Stim001','Stim002','Stim003','Stim004','Stim005']
    stimulus=['Stim001','Stim002','Stim003','Stim005']
    #stimulus=['baseline']
    
    #pre_post='pre'
    pre_post='post'    
    #pre_post='24_hours'
    #pre_post='baseline'
    #miss_pat=['Pat010','Pat012','Pat018','Pat032','Pat035','Pat060','Pat079','Pat091','Pat111'] #REEG
    miss_pat=['Pat006','Pat009','Pat010','Pat016','Pat018','Pat021','Pat025','Pat32','Pat33','Pat035','Pat079','Pat091','Pat092','Pat096'
              ,'Pat101','Pat106','Pat112','Pat133','Pat147','Pat151','Pat152','Pat154'] #REEG
    #missing = np.load('E:\missing_ecg.npy')
    missing = []
    
    optimizers = 'roc_auc'
    
    comb = False
    
    itera = 10
    splits = 5
    cv = 3
        
    for k in range(0,len(stimulus)):
        
        
        rfc_m = Measures(splits,itera)
        svm_m = Measures(splits,itera)
        lr_m = Measures(splits,itera)
        xgb_m = Measures(splits,itera)
        nn_m = Measures(splits,itera)
                        

    
        for i in range(0,itera):
       
                path_outcome="L:\\basic\\divd\\knf\\Onderzoek_projecten\\EEG_IC\\DeepEEG\\final_DeepEEGdata\\Outcome_binary_3months_final.txt"
                #path_data=r"L:\basic\divd\knf\Onderzoek_projecten\EEG_IC\DeepEEG\final_DeepEEGdata\5Sec_MLpart2\frequencyFeatures_fft"
                path_data=r"L:\basic\divd\knf\Onderzoek_projecten\EEG_IC\DeepEEG\final_DeepEEGdata\5Sec\frequencyFeatures_fft"
                         
                path = r"F:\DeepEEG\6_months_final_Results"      
                if comb:
                    path_results=r"F:\DeepEEG\comb_6_months_final_Results"+secs+stimulus[k]+pre_post+"\\"
                else:
                    path_results=r"F:\DeepEEG\6_months_final_Results"+secs+stimulus[k]+pre_post+"\\"
                if not os.path.exists(path_results):
                     os.makedirs(path_results)
                
           
                #new_label=np.loadtxt("L:\\basic\\Personal Archive\\L\\laramos\\DeepEEG\\data\\Outcome_binary_3months_final.txt")
                new_label = pd.read_csv(r"\\amc.intra\users\L\laramos\home\Desktop\EEG Rebuttal\Outcome_binary_6months_final.txt", sep="	", header=None)
                new_header = new_label.iloc[0] #grab the first row for the header
                new_label = new_label[1:] #take the data less the header row
                new_label.columns = new_header #set the header row as the df header
                
        #        names_subj=pd.read_table(path_outcome,delimiter=',')
                #names_subj=pd.read_table(r"L:\basic\divd\knf\Onderzoek_projecten\EEG_IC\DeepEEG\final_DeepEEGdata\Outcome_binary_3months_final_V3.txt",delimiter=',')

                
                #skf = StratifiedShuffleSplit(n_splits=splits, test_size=0.20,random_state=1)
                #skf = LeaveOneOut()
                skf = KFold(n_splits=splits, shuffle=True, random_state=i)  
                      
                X = np.array(new_label['names'])
                Y = np.array(new_label['label'],dtype='int16')
                splits=skf.get_n_splits(X)
            
                mean_tprr = 0.0
                num_feats=12

                
                # X = np.array(X)
                # missing = np.array(missing)
                # for i in range(missing.shape[0]):
                #     w = np.where(X==missing[i])
                #     X[w[0][0]] = ''            
                
                for l, (train, test) in enumerate(skf.split(X, Y)):                
                        val = random.randint(1,60)
        #                X_train = np.load(r'E:\DeepEEG\verne_final_Results5Stim00424_hours\\'+"train_pat"+str(l)+".npy")               
        #                X_test = np.load(r'E:\DeepEEG\verne_final_Results5Stim00424_hours\\'+"test_pat"+str(l)+".npy")
        #                y_test = np.load(r'E:\DeepEEG\verne_final_Results5Stim00424_hours\\'+"test_lab"+str(l)+".npy")
        #                y_train = np.load(r'E:\DeepEEG\verne_final_Results5Stim00424_hours\\'+"train_lab"+str(l)+".npy")
                        
                        X_train = np.array(X[train])
                        X_test = np.array(X[test])
                        y_train = np.array(Y[train])
                        y_test = np.array(Y[test])    
                        x_test_orig = X[test]              
                        x_train_orig = X[train]              
                        if l>=0:
                            print("Reading Data!")
                            #if x_test not in miss_pat:
                            #print(X_test)
                            #X_train,y_train=Read_Data_Subtract(X_train,y_train,k,secs)
                            #X_test,y_test=Read_Data_Subtract(X_test,y_test,k,secs)
                             # l = 0 
                             # k = 0
                             # path_results=r"F:\DeepEEG\6_months_final_Results"+secs+stimulus[0]+pre_post+"\\"
                             # X_train1 = np.load(path_results+'train_'+stimulus[0]+str(l)+".npy")                    
                             # path_results=r"F:\DeepEEG\6_months_final_Results"+secs+stimulus[1]+pre_post+"\\"
                             # X_train2 = np.load(path_results+'train_'+stimulus[1]+str(l)+".npy")
                             # path_results=r"F:\DeepEEG\6_months_final_Results"+secs+stimulus[2]+pre_post+"\\"
                             # X_train3 = np.load(path_results+'train_'+stimulus[2]+str(l)+".npy")
                             # path_results=r"F:\DeepEEG\6_months_final_Results"+secs+stimulus[3]+pre_post+"\\"
                             # X_train4 = np.load(path_results+'train_'+stimulus[3]+str(l)+".npy")
                             # path_results=r"F:\DeepEEG\6_months_final_Results"+secs+stimulus[4]+pre_post+"\\"
                             # X_train5 = np.load(path_results+'train_'+stimulus[4]+str(l)+".npy")
                            
                            if comb:
                                X_train, y_train = read_all_stim(path, secs, stimulus, pre_post, l, 'train')
                                X_test, y_test = read_all_stim(path, secs, stimulus, pre_post, l,'test')
                            else:                        
                                X_train,y_train,v_train,m_train = Read_Data(path_data,X_train,y_train,k,secs,miss_pat,val,pre_post)
                                
                                X_test,y_test,v_test,m_test = Read_Data(path_data,X_test,y_test,k,secs,miss_pat,val,pre_post)
                                
                                
                                new_x_test = list()
                                for ids in x_test_orig:
                                    if ids not in m_test:
                                        new_x_test.append(ids)
                                        
                                new_x_train = list()
                                for ids in x_train_orig:
                                    if ids not in m_train:
                                        new_x_train.append(ids)                                
                                        
                                
                                print("Before:",X_train.shape)
                                X_train,y_train,x_train_orig,xt = remove_outliers(X_train,y_train,list(v_train))
                                X_test,y_test,x_test_orig,xt2 = remove_outliers(X_test,y_test,list(v_test))
                                print("After:",X_train.shape)
        
                                np.save(path_results+'x_train_'+stimulus[k]+str(l)+".npy",X_train)
                                np.save(path_results+'x_test_'+stimulus[k]+str(l)+".npy",X_test)
                                
                                np.save(path_results+'y_train_'+stimulus[k]+str(l)+".npy",y_train)
                                np.save(path_results+'y_test_'+stimulus[k]+str(l)+".npy",y_test)    
                                
                                df = pd.DataFrame()
                                df['ID'] = x_test_orig
                                df['y'] = y_test
                                df.to_csv(path_results+"frame_test_"+str(l)+".csv")
        
        
                            print("Data Complete, preparing network!")    
                            #scaler = preprocessing.StandardScaler().fit(X_train)
                            scaler = RobustScaler().fit(X_train)
                            X_train=scaler.transform(X_train)
                            X_test=scaler.transform(X_test) 
                           
                            class_rfc = mt.Pipeline(True,'RFC',X_train,y_train,X_test,y_test,l,i,cv,mean_tprr,rfc_m,path_results,optimizers,'s')   
                            class_svm = mt.Pipeline(True,'SVM',X_train,y_train,X_test,y_test,l,i,cv,mean_tprr,svm_m,path_results,optimizers,'s')   
                            class_lr = mt.Pipeline(True,'LR',X_train,y_train,X_test,y_test,l,i,cv,mean_tprr,lr_m,path_results,optimizers,'s')
                            class_nn = mt.Pipeline(True,'NN',X_train,y_train,X_test,y_test,l,i,cv,mean_tprr,nn_m,path_results,optimizers,'s') 
                            class_xgb = mt.Pipeline(True,'XGB',X_train,y_train,X_test,y_test,l,i,cv,mean_tprr,xgb_m,path_results,optimizers,'s')   
                            #Naive Bayes
                            #KNN
    

        #            sys.exit()
        final_m=[rfc_m,svm_m,lr_m,xgb_m,nn_m]
        final_m=[x for x in final_m if x.run != False]
        names=[class_rfc.name,class_svm.name,class_lr.name,class_xgb.name,class_nn.name]
        #names=[class_rfc.name,class_svm.name,class_lr.name,class_nn.name]
        names=[x for x in names if x != 'NONE'] 
        mt.Print_Results_Excel(final_m,splits,names,path_results,stimulus[k])
        
        Save_fpr_tpr(path_results,names,final_m)


# #missing_train = sorted(missing_train)
# x=X_train
# #np.save('E:\missing_ecg.npy',missing_train)
# n_feats=12
# channels=14
# files=list()
# labels=list()
# missing=list()
# for i in range(0,x.shape[0]):    
#     if x[i] not in missing:
#         print(x[i])
#     #read the files of one of the patients
#         #f=(glob.glob(path+"\\"+x[i]+"_"+stimulus[k]+"*"+pre_post+"Stim.txt"))        
#         #f=(glob.glob(r"E:\DeepEEG\data\\"+secs+"sec\\frequencyFeatures_fft\\"+x[i]+"_baseline_epoch"+str(val)+".txt"))        
#         f=(glob.glob(path_data+"\\"+x[i]+"_24h_epoch"+str(val)+".txt"))        
#         #f=(glob.glob(path_data+"\\"+x[i]+"_baseline_epoch"+str(val)+".txt"))        
#         #append the file names and the same label for each file available
#         if len(f)>0:
#             #labels.append(y[i])
#             files.append(f[0])
#         else:
#             #print(x[i],f)
#             missing.append(x[i])






# x = pd.DataFrame(X_train)
# Q1 = x.quantile(0.25)
# Q3 = x.quantile(0.75)
# IQR = Q3 - Q1
# print(IQR)

# s = ((x < (Q1 - 1.5 * IQR))| (x > (Q3 + 1.5 * IQR)))
# c = x[~((x < (Q1 - 1.5 * IQR)) |(x > (Q3 + 1.5 * IQR))).any(axis=1)]

# s = np.array(s)
# x = np.array(x)
# x[s[:,0]==True,0]

x = new_label['names']

x = set(x)
vt = v_train+v_test
vt = set(vt)
f = x - vt

x = sorted(x_train_orig)
v_train = sorted(v_train)
i=0


x = set(xt['id'])
vt = v_train
vt = set(vt)
f = vt-x
print(f)



x = set(x_train_orig)
vt = set(v_train)
f = vt-x

print(f)

f = sorted(f)
