# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 16:12:01 2018

@author: laramos
"""
import numpy as np
#import Data_Preprocessing as dp
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


  
#loading Marjolein Features and checking for missing labels




#secs='5'
secs='10'
stimulus=['Stim001','Stim002','Stim003','Stim004','Stim005']
#stimulus=['baseline']

pre_post='pre'
#pre_post='post'

for p in range(2,3):
#for p in range(0,1):
    
    path_features=r"E:\DeepEEG\data\\"+secs+"sec\\frequencyFeatures_fft\\*"+stimulus[p]+"*"+pre_post+"*Stim.txt"
    #path_features="E:\\DeepEEG\\data\\"+secs+"secs\\frequencyFeatures_fft\\*"+"baseline_"+pre_post+"*Stim.txt"
        
    files=sorted((glob.glob(path_features)))
    
    path_outcome="L:\\basic\\divd\\knf\\Onderzoek_projecten\\EEG_IC\\DeepEEG\\final_DeepEEGdata\\Outcome_binary_3months_final.txt"
    path_feats=r"L:\basic\divd\knf\Onderzoek_projecten\EEG_IC\DeepEEG\final_DeepEEGdata\10Sec\featureNames.txt"
    new_label=np.loadtxt("E:\\DeepEEG\\data\\Outcome_binary_3months_final.txt")
    names_subj=pd.read_table(path_outcome,delimiter=',')
    names_subj['label']=new_label
    names_feat=pd.read_table(path_feats)
    
    
    files=(glob.glob(path_features))
       
    samples=len(files)
    samples_include=len(files)
    #8 features for 14 signals
    channels=14
    n_feats=12
    #pre and pos data files
    data=np.zeros((samples_include,n_feats,channels))        
    labels=np.zeros(samples_include)
    
    cont_pre=0
    cont_pos=0

    cont=0
    
    for i in range(0,samples):
            file_input=open(files[i])
            lines_input=file_input.readlines()  
            m=re.search(r"Pat[0-9][0-9][0-9]",files[i])
            m=m.group(0)[3:6]
            
            found=False    
            for k in range(0,len(names_subj)):            
                    num_subj=(names_subj['names'].iloc[k])
                    if (num_subj in files[i]): 
                        l=names_subj['label'].iloc[k]#copy outcome

                        found=True#found patient label
                        for j in range(0,n_feats):
                                data[cont,j,:]=np.fromstring(lines_input[j], dtype='float32', sep=";").reshape(14) 
                        labels[cont]=l
                        cont=cont+1
                            
            if found==False: 
                print("Missing file: ", files[i])#print file with missing label
    
    #computing average over the 14 channels
    data_avg=np.mean(data,axis=2)        
    #data_pos_avg=np.mean(data_pos,axis=2)
    #data_pre_avg=data_pre_avg[0:cont_pre,:]
    #data_pos_avg=data_pos_avg[0:cont_pos,:]
    
    data_avg=pd.DataFrame(data_avg,columns=names_feat['featureNames'])
    labels=pd.DataFrame(labels,columns=['Outcome'])
    
    
    data_avg.to_csv('data_all'+stimulus[p]+'_'+secs+'_'+pre_post+'.csv',index_label=False)
    labels.to_csv('label_all'+stimulus[p]+'_'+secs+'_'+pre_post+'.csv',index_label=False)
   
    #This is to visualize which are some outliers, it lists the same and the value of the outlier
    
    d=np.array(data_avg)
    feat=8
    for i in range(0,20):
        s=np.argmax(d[:,feat])
        print(files[s],d[s,feat])
        d[s,feat]=0
    
        
#---------------------------
#-------------------------VISUALIZING--------------------------------------------
#--------------------------

"""
#Visualize the outcomes distribution
labels['Outcome'].value_counts()
data_avg['Outcome']=labels['Outcome']
#sns.countplot(x='Outcome',data=data_avg[names_feat['featureNames'].iloc[0]],palette='hls')
sns.countplot(x='Outcome',data=data_avg,palette='hls')

mean_vars=data_avg.groupby('Outcome').mean()
std_vars=data_avg.groupby('Outcome').std()

pd.crosstab(data_avg[names_feat['featureNames'].iloc[0]],data_avg['Outcome']).plot(kind='bar')
s=data_avg[names_feat['featureNames'].iloc[0]]
s.hist()

s=np.array(data_avg['Outcome'],dtype='bool')
data_pos=data_avg[s]

s=np.array(1-s,dtype='bool')
data_neg=data_avg[s]

#visualizing the whole data features in boxplots
for i in range(names_feat.shape[0]):
    #s=data_avg[names_feat['featureNames'].iloc[i]]
    #plt.boxplot(s)
    #plt.title(names_feat['featureNames'].iloc[i])
    #plt.show()
    
    s=[data_pos[names_feat['featureNames'].iloc[i]],data_neg[names_feat['featureNames'].iloc[i]]]
    plt.boxplot(s)
    plt.title(names_feat['featureNames'].iloc[i])
    plt.show()
    
    
    

    
#deleting some outliers
s=data_avg[names_feat['featureNames'].iloc[0]]
f_list=s<10000
new_data=data_avg[f_list]
new_labels=labels[f_list]


for i in range(names_feat.shape[0]):
    s=new_data[names_feat['featureNames'].iloc[i]]
    plt.boxplot(s)
    plt.title(names_feat['featureNames'].iloc[i])
    plt.show()

    
plot_data=[new_data['Total Power (1-25Hz)'],new_data['Alpha Power (8-13Hz)']]
plt.boxplot(plot_data)
"""