# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:36:33 2018

@author: laramos
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy as sp
import pandas as pd

def Mean_Confidence_Interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

"""
path=r'E:\\DeepEEG\\Results_5Stim001pre\\'


names=['SVM','LR','RFC']

for i in range(0,3):
    
    f=np.load(path+'fpr_'+names[i]+'.npy')
    t=np.load(path+'tpr_'+names[i]+'.npy')
    #print(names[i],f[np.argmax(t-f)],t[np.argmax(t-f)])
    
    plt.plot(f,t)
    plt.plot(f[np.argmax(t-f)],t[np.argmax(t-f)],'ro')
    spec=1-f
    
    spec=-f[np.argmax(t-f)]+1
    #val=31
    #spec=-f[val]+1
    #print(names[i]+' Sensitivity %f, Specificity %f' % (t[val],spec))
    print(names[i]+' Sensitivity %f, Specificity %f' % (t[np.argmax(t-f)],spec))
    
   



    
path=r"E:\DeepEEG\Results\Results_10Stim001pre\"

names=['SVM','LR','RFC']
sens=np.zeros(100)
spec=np.zeros(100)
auc=np.zeros(100)
result_sens=np.zeros((100,3))
result_spec=np.zeros((100,3))
for i in range(0,3):
    for j in range(0,100):
        

        f=np.load(path+'fpr_'+names[i]+'_'+str(j)+'.npy')
        t=np.load(path+'tpr_'+names[i]+'_'+str(j)+'.npy')
        auc[j]=metrics.auc(f,t)

        
        s=np.round(1-f,decimals=2)
        s_bol=s>0.95
        k=0
        while s_bol[k]==True:
            k=k+1
    
        sens[j]=t[k-1]
        spec[j]=s[k-1]
    result_sens=Mean_Confidence_Interval(sens)
    result_spec=Mean_Confidence_Interval(spec)
    print(names[i]+' Sensitivity %f, %f, %f' %(Mean_Confidence_Interval(sens)))
    print(names[i]+' Specificity %f, %f, %f' %(Mean_Confidence_Interval(spec)))
"""    


    
#Combining all the .xls files  
secs="5"
#secs="10"

pre_post="pre"
#pre_post="post"
#pre_post="pre-post"
   

#local="E:\\DeepEEG\\Results\\Results"
local="E:\\DeepEEG\\Results\\24h_epochs"

xls = pd.ExcelFile(local+secs+"Stim001"+pre_post+"\\Results.xls")
#xls = pd.ExcelFile(r"E:\DeepEEG\Results\Results_5Stim001post\\Results.xls")

df1 = xls.parse('Sheet 1')

df1.insert(0,'Stimulus', 1)

names=['RFC','SVM','LR']
sens=np.zeros(100)
spec=np.zeros(100)
auc=np.zeros(100)
result_sens=np.zeros((15,3))
result_spec=np.zeros((15,3))
ks=np.zeros((100,5))

cont=0
for m in range(1,6):
    print("Stim: ",m)
    for i in range(0,3):
        for j in range(0,100):
            
            path=local+secs+"Stim00"+str(m)+pre_post+"\\"
            #path=r"E:\DeepEEG\Resuls\Results_5baselinepre\\"
            f=np.load(path+'fpr_'+names[i]+'_'+str(j)+'.npy')
            t=np.load(path+'tpr_'+names[i]+'_'+str(j)+'.npy')
            auc[j]=metrics.auc(f,t)
                
            s=np.round(1-f,decimals=2)
            s_bol=s>=0.94
            k=0
            while s_bol[k]==True:
                k=k+1
            ks[j,m-1]=k-1
            sens[j]=t[k-1]
            spec[j]=s[k-1]
    
        result_sens[cont,:]=Mean_Confidence_Interval(sens)
        result_spec[cont,:]=Mean_Confidence_Interval(spec)
        cont=cont+1
    break
        
result_sens=np.round(result_sens,decimals=2)               
result_spec=np.round(result_spec,decimals=2)                    
     
for i in range(2,6):       
    xls2 = pd.ExcelFile(local+secs+"Stim00"+str(i)+pre_post+"\\Results.xls")

    df2 = xls2.parse('Sheet 1')
    df2.insert(0,'Stimulus', i)

    df1=pd.concat((df1,df2),axis=0,ignore_index=True)

sens=list()
spec=list()

for i in range(0,15):
    sens.append(str(result_sens[i,0])+" ("+str(result_sens[i,1])+" - "+str(result_sens[i,2])+")")
    spec.append(str(result_spec[i,0])+" ("+str(result_spec[i,1])+" - "+str(result_spec[i,2])+")")
    
df1['Optimized Sensitivity']=pd.Series(sens)
df1['Optimized Specificity']=pd.Series(spec)
  
writer = pd.ExcelWriter(local+secs+"secs_"+pre_post+".xlsx", engine='xlsxwriter')
#writer = pd.ExcelWriter("E:\\DeepEEG\\Results\\5sec_Baseline.xlsx", engine='xlsxwriter')
df1.to_excel(writer, 'Sheet1')
writer.save()

