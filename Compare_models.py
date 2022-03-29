# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 12:49:11 2019

@author: laramos
"""
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import 

prob_reac=np.zeros((30,2))
file=np.load(r"L:\basic\Personal Archive\L\laramos\DeepEEG\Results\skNN_featimportance5Stim001pre\RFC_prob_stim_0_2.npy")
cont=0
cont2=0
for i in range(0,31):
    if i!=:
        prob_reac[cont2,:]=file[cont,:]
        cont2+=1
    cont+=3
    print(cont)
    

prob_base=np.load(r"L:\basic\Personal Archive\L\laramos\DeepEEG\Results\24hrs5Stim001pre\RFC_prob_stim_0_2.npy")

spec=0
total_vals=1000
y=prob_reac[:,1]
tresh=np.linspace(0.99,0,num=total_vals)
cont=0
#First I find which threshold gives me a good spec, i have to add it to the code below by had, just look at the prints
while spec<=0.95 and cont<total_vals:
    
    t=tresh[cont]
    
    p=prob_reac[:,0]>t
    
    mat=confusion_matrix(y,p)

    sens=mat[0,0]/(mat[0,0]+mat[1,0])

    spec=mat[1,1]/(mat[1,1]+mat[0,1])
    cont+=1
    
    print(spec,sens,t)
#Now I cna find where the spec was good, get that threshold and use it here to compute the predictions for pred_read    
pred_reac=prob_reac[:,0]>0.301 #add t here
    
mat=confusion_matrix(y,pred_reac)

sens=mat[0,0]/(mat[0,0]+mat[1,0])

spec=mat[1,1]/(mat[1,1]+mat[0,1])

print(spec,sens) 


spec=0
total_vals=1000
y=prob_base[:,1]
tresh=np.linspace(0.99,0,num=total_vals)
cont=0

#doing same for probas_base
while spec<=0.95 and cont<total_vals:
    
    t=tresh[cont]
    
    p=prob_base[:,0]>t
    
    mat=confusion_matrix(y,p)

    sens=mat[0,0]/(mat[0,0]+mat[1,0])

    spec=mat[1,1]/(mat[1,1]+mat[0,1])
    cont+=1
    
    print(spec,sens,t)
    
pred_base=prob_base[:,0]> 0.268
    
mat=confusion_matrix(y,pred_base)

sens=mat[0,0]/(mat[0,0]+mat[1,0])

spec=mat[1,1]/(mat[1,1]+mat[0,1])

print(spec,sens) 


#Now I'll check how different they are

m1=confusion_matrix(y,pred_base)
print(m1)

m2=confusion_matrix(y,pred_reac)
print(m2)

mat=confusion_matrix(pred_reac,pred_base)


print("Equal:",np.sum(pred_base==pred_reac))


data=np.zeros((30,5))


c=0
for i in range(0,30):
    if pred_base[i]!=y[i] or pred_reac[i]!=y[i]:
        data[c,0]=prob_reac[i,0]
        data[c,1]=pred_reac[i]        
        data[c,2]=prob_base[i,0]
        data[c,3]=pred_base[i]
        data[c,4]=y[i]
        c+=1
    

data=pd.DataFrame(data,columns=['Prob_reac','Pred_reac','Prob_base','Pred_base','Y'])







t=np.load(r"L:\basic\Personal Archive\L\laramos\DeepEEG\Results\skNN_featimportance5Stim001pre\Thresholds_RFC.npy")
sens=np.zeros(100)
spec=np.zeros(100)
for i in range(0,100):

    p=np.load(r"L:\basic\Personal Archive\L\laramos\DeepEEG\Results\skNN_featimportance5Stim005pre\RFC_prob_stim_4_"+str(i)+".npy")
    
    
    y=np.array(p[:,1],dtype="int16")
    pred=p[:,0]
    pred_bin=np.array(pred>t[i],dtype="int16")
    
    mat=confusion_matrix(y,pred_bin)
    
    sens[i]=mat[0,0]/(mat[0,0]+mat[1,0])
    
    spec[i]=mat[1,1]/(mat[1,1]+mat[0,1])
    
    print(spec[i],sens[i]) 

print("Average",np.mean(spec),np.mean(sens))




path=r"E:\DeepEEG\Results\skNN_featimportance5Stim001pre\\"

names=["RFC","NN","LR","SVM"]
which=2
auc=np.zeros(100)
sens=np.zeros(100)
spec=np.zeros(100)

t=np.load(path+"Thresholds_"+names[which]+".npy")

for i in range(100):
    probas=np.load(path+names[which]+"_prob_stim_0_"+str(i)+".npy")
    y=probas[:,1]
    probas=probas[:,0]
    
    auc[i]=roc_auc_score(y,probas)
    probas=probas>t[i]
    
    conf_m=confusion_matrix(y, probas)    
    sens[i]=conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])
    spec[i]=conf_m[1,1]/(conf_m[1,1]+conf_m[0,1])
    #tn, fp, fn, tp = confusion_matrix(y, probas).ravel()
    #sens=tp/(tp+fn)
    #spec=tn/(tn+fp)
    
print("Auc:",np.mean(auc))
print("sens:",np.mean(sens))
print("spec:",np.mean(spec))




