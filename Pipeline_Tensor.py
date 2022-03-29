# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:00:03 2017

@author: laramos
"""
import os
cwd = os.getcwd()
os.chdir(cwd)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import glob

from sklearn.model_selection import ShuffleSplit
from sklearn import preprocessing

#from sklearn.preprocessing import normalize
import time
from scipy import interp                      

#import methods_Prospective as mp
import warnings

import pandas as pd
from sklearn.calibration import calibration_curve
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

import random
from sklearn.metrics import roc_auc_score
from scipy import interp
from sklearn.metrics import roc_curve
from keras.utils import np_utils
import scipy as sp
from sklearn.model_selection import StratifiedShuffleSplit

class Measures:       
    def __init__(self, splits,num_feats):
        self.clf_auc=np.zeros(splits)
        self.clf_brier=np.zeros(splits)
        self.clf_sens=np.zeros(splits)
        self.clf_spec=np.zeros(splits)
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

def Read_Data(x,y,k,secs):
    import random
    val=random.randint(1,30)
    
    n_feats=12
    channels=14
    files=list()
    labels=list()
    
    for i in range(0,x.shape[0]):    
        #read the files of one of the patients
        f=(glob.glob(r"E:\DeepEEG\data\\"+secs+"sec\\frequencyFeatures_fft\\"+x[i]+"_"+stimulus[k]+"*"+pre_post+"Stim.txt"))        
        #f=(glob.glob(r"E:\DeepEEG\data\\"+secs+"sec\\frequencyFeatures_fft\\"+x[i]+"_baseline_epoch"+str(val)+".txt"))        
        #f=(glob.glob(r"E:\DeepEEG\data\\"+secs+"sec\\frequencyFeatures_fft\\"+x[i]+"_24h_epoch"+str(val)+".txt"))        
        #append the file names and the same label for each file available
        for j in range(0,len(f)):
            labels.append(y[i])
            files.append(f[j])
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

def mean_confidence_interval(data, confidence=0.95):
    """
    Compute 95% confidence interval
    Input:
        data: values
        confidence (optional)
    Output:
        m: average
        m-h: lower limit from CI
        m+h: upper limit of CI
    """
    
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h
    

def read_batch(X,Y,batch_size,start):
    end=start+batch_size
    return(X[start:end,:],Y[start:end,:])
  
         

secs='5'
#secs='10'

stimulus=['Stim001','Stim002','Stim003','Stim004','Stim005']

pre_post='pre'
#pre_post='post'
layers_nn=[[24,12],[24,30,24],[18,12],[18,36,18],[24,30,12]]

for k in range(0,5):
        
    path_outcome="L:\\basic\\divd\\knf\\Onderzoek_projecten\\EEG_IC\\DeepEEG\\final_DeepEEGdata\\Outcome_binary_3months_final.txt"
    path_data=r"L:\basic\Personal Archive\L\laramos\DeepEEG\data"
        
    path_results="L:\\basic\\Personal Archive\\L\\laramos\\DeepEEG\Results\\featimportance"+secs+stimulus[k]+pre_post+"\\"
    if not os.path.exists(path_results):
         os.makedirs(path_results)
    
   
    new_label=np.loadtxt("L:\\basic\\Personal Archive\\L\\laramos\\DeepEEG\\data\\Outcome_binary_3months_final.txt")
    names_subj=pd.read_table(path_outcome,delimiter=',')
    
    splits=100
    
    
    skf = StratifiedShuffleSplit(n_splits=splits, test_size=0.20,random_state=1)

    mean_tprr = 0.0
    num_feats=12
    rfc_m=Measures(splits,num_feats)
    svm_m=Measures(splits,num_feats)
    lr_m=Measures(splits,num_feats)
    xgb_m=Measures(splits,num_feats)
    
    X=names_subj['names']
    Y=names_subj['label']=new_label
    
    Y=Y.ravel()
    
    seed = 128   
     
    
    
    hidden_num_units_1 = 18
    hidden_num_units_2 = 36
    hidden_num_units_3 = 18
    
    splits=300
    acc_m=np.zeros(splits)
    acc_t=np.zeros(splits)
    
    auc_m=np.zeros(splits)
    
    
    learning_rate = 0.01
    
    mean_tprr = 0.0
    
    mean_tprn = 0.0
    
    mean_fpr = np.linspace(0, 1, 100)

    
    
    for l, (train, test) in enumerate(skf.split(X, Y)): 
        start_pipeline = time.time()                   
        
        x_train=np.array(X[train])
        x_test=np.array(X[test])
        y_train=np.array(Y[train])
        y_test=np.array(Y[test])
        print("Reading Data!")
        x_train,y_train=Read_Data_Subtract(x_train,y_train,k,secs)
        x_test,y_test=Read_Data_Subtract(x_test,y_test,k,secs)
        print("Data Complete, preparing network!")
        
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train=scaler.transform(x_train)
        x_test=scaler.transform(x_test) 
        
        y_test = np_utils.to_categorical(y_test,2)
        y_train = np_utils.to_categorical(y_train,2)
    
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train=scaler.transform(x_train)
        x_test=scaler.transform(x_test)
    
        #w=np.sum(Y_train[:,0])/np.sum(Y_train[:,1])
        input_num_units=x_train.shape[1]
        # define placeholders
        x = tf.placeholder(tf.float32, [None, input_num_units])
        y = tf.placeholder(tf.float32, [None, 2])
        prob = tf.placeholder_with_default(1.0, shape=())
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        weights = {
        	'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units_1]), name='wh_1'),
        	'hidden2': tf.Variable(tf.random_normal([hidden_num_units_1, hidden_num_units_2])),
        	'hidden3': tf.Variable(tf.random_normal([hidden_num_units_2, hidden_num_units_3])),
        	'output': tf.Variable(tf.random_normal([hidden_num_units_1, 2]))
        }
        biases = {
        	'hidden': tf.Variable(tf.random_normal([hidden_num_units_1])),
        	'hidden2': tf.Variable(tf.random_normal([hidden_num_units_2])),
        	'hidden3': tf.Variable(tf.random_normal([hidden_num_units_3])),
        	'output': tf.Variable(tf.random_normal([2]))
        }
        
        hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'], name='h1_add')
        hidden_layer = tf.nn.sigmoid(hidden_layer, name='h1_sg')
        hidden_layer = tf.nn.dropout(hidden_layer, keep_prob=prob, name='h1_dp')
        # layer 2
        hidden_layer = tf.add(tf.matmul(hidden_layer, weights['hidden2']), biases['hidden2'], name='h2_add')
        hidden_layer = tf.nn.sigmoid(hidden_layer, name='h2_sg')
        hidden_layer = tf.nn.dropout(hidden_layer, keep_prob=prob, name='h2_dp')
        # layer 3
        hidden_layer = tf.add(tf.matmul(hidden_layer, weights['hidden3']), biases['hidden3'], name='h3_add')
        hidden_layer = tf.nn.sigmoid(hidden_layer, name='h3_sg')
        hidden_layer = tf.nn.dropout(hidden_layer, keep_prob=prob, name='h3_dp')
        # output
        output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
                                 
        probs = tf.nn.softmax(output_layer, name="softmax_tensor")
        
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=y))
        
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        
        init = tf.global_variables_initializer()
        
        batch_size=32
    
        training_steps = int(np.ceil(x_train.shape[0] / batch_size))
        
        epochs=200
        with tf.Session(config=config) as sess:
            sess.run(init)      
            for j in range(0,epochs):           
                avg_loss=0
                start=0
                for batch in range(training_steps):
                    avg_cost = 0
                    x_batch,y_batch=read_batch(x_train,y_train,batch_size,start)
                    start=start+batch_size       
                    _, c = sess.run([optimizer, cost], feed_dict = {x: x_batch, y: y_batch,prob: 0.7})
                    avg_loss += (c / training_steps)
                    
                #print("Epoch:", (j), "cost =", "{:.5f}".format(avg_loss))
        
            print("\nTraining complete!")
            
            _,_,p=sess.run([optimizer, cost, probs], feed_dict = {x: x_test, y: y_test,prob: 1})
            pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
            predict = tf.argmax(output_layer, 1)
            pred = predict.eval({x: x_test})
            acc_m[l]= accuracy.eval({x: x_test, y: y_test})
            acc_t[l]= accuracy.eval({x: x_train, y: y_train})
            auc_m[l]=roc_auc_score(y_test[:,1],p[:,1])
            
            
            print("Training Accuracy:",acc_t[l])
            print("Testing Accuracy:",acc_m[l])
            print("Testing AUC:",auc_m[l])
            fpr,tpr,_= roc_curve(y_test[:,1],p[:,1])
            mean_tprn += interp(mean_fpr, fpr, tpr)
            sess.close()
        #train_writer.close()

    thefile = open(path_results+'Sensitivity.txt', 'a')
    
    thefile.write("NN \n")
    
    thefile.write("Average AUC From testing set %f and AUC-CI %f - %f \n" % (mean_confidence_interval(auc_m)))
    
    thefile.close()
    
    
    
    thefile = open(path_results+'AUC-NN.txt', 'w')
    
    for i in range(0,splits):
        thefile.write("%f\n" %(auc_m[i]))
    
    thefile.close()
    
    
    mean_tprn /= splits
    mean_tprn[-1] = 1.0 
    
    print("Average ACC Training: ",np.mean(acc_t))
    print("Average ACC Testing: ",np.mean(acc_m))
    print("Average AUC Testing: ",np.mean(auc_m))
    
    
    plt.figure()
    lw = 2
    plt.plot(mean_fpr, mean_tprn, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % np.mean(auc_m))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
