# -*- coding: utf-8 -*-
"""

VENN DISGRAMS HERE
Created on Tue Jan 29 14:10:04 2019

@author: laramos
"""

from matplotlib_venn import venn2 

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt  
from sklearn.metrics import roc_auc_score



r_df = list()
b_df = list()

#ts = np.load(r"F:\DeepEEG\final_results_10cv\6_months_final_Results5Stim004post\Thresholds_LR.npy")
ts = np.load(r"F:\DeepEEG\6_months_final_Results5Stim004post\Thresholds_RFC.npy")
ts2 = np.load(r"F:\DeepEEG\6_months_final_Results5Stim00424_hours\Thresholds_RFC.npy")
l = 45
for i in range(0,5):
    reac = np.load(r"F:\DeepEEG\6_months_final_Results5Stim004post\probabilities_RFC_"+ str(i)+"9.npy")
    back = np.load(r"F:\DeepEEG\6_months_final_Results5Stim00424_hours\probabilities_RFC_"+ str(i)+"9.npy")
    df_reac = pd.read_csv(r"F:\DeepEEG\6_months_final_Results5Stim004post\frame_test_"+str(i)+".csv")
    df_back = pd.read_csv(r"F:\DeepEEG\6_months_final_Results5Stim00424_hours\frame_test_"+str(i)+".csv")
    
    print(reac.shape,df_reac.shape)
    print(back.shape,df_back.shape)

    p = reac[:,0]>ts[l]

    

    #reac[:,0] = p
            
    df_reac['pred_R'] = p
    df_reac['prob_R'] = reac[:,0]
    
    r_df.append(df_reac)
    
    p = back[:,0]>ts2[l]
    
    l = l+1

    #back[:,0] = p
    
    df_back['pred_B'] = p
    df_back['prob_B'] = back[:,0]
    b_df.append(df_back)

    
r_df = pd.concat(r_df)
b_df = pd.concat(b_df)
   
merge = r_df.merge(b_df,on='ID')
merge['FP_R']= ''
for i in range(0,merge.shape[0]):
    if merge.y_x.iloc[i]==0 and merge.pred_R.iloc[i]==1:
        merge['FP_R'].iloc[i]=1
    else:
        merge['FP_R'].iloc[i]=0
        
merge['FN_R']= ''
for i in range(0,merge.shape[0]):
    if merge.y_x.iloc[i]==1 and merge.pred_R.iloc[i]==0:
        merge['FN_R'].iloc[i]=1
    else:
        merge['FN_R'].iloc[i]=0   
        
        
merge['FP_B']= ''
for i in range(0,merge.shape[0]):
    if merge.y_x.iloc[i]==0 and merge.pred_B.iloc[i]==1:
        merge['FP_B'].iloc[i]=1
    else:
        merge['FP_B'].iloc[i]=0
        
merge['FN_B']= ''
for i in range(0,merge.shape[0]):
    if merge.y_x.iloc[i]==1 and merge.pred_B.iloc[i]==0:
        merge['FN_B'].iloc[i]=1
    else:
        merge['FN_B'].iloc[i]=0           
        
 
fp_r = merge[merge.FP_R==1] 
fp_b = merge[merge.FP_B==1]        
        
fn_r = merge[merge.FN_R==1] 
fn_b = merge[merge.FN_B==1]  

v=venn2([set(fp_r['ID']), set(fp_b['ID'])],set_labels = ('qEEG-R', 'qEEG'))
v.get_patch_by_id('C').set_color('orange')
#v.get_label_by_id('A').set_size(20)
#v.get_label_by_id('B').set_size(20)
for text in v.set_labels:
    text.set_fontsize(20)
for x in range(len(v.subset_labels)):
    if v.subset_labels[x] is not None:
        v.subset_labels[x].set_fontsize(20)

plt.show()

v=venn2([set(fn_r['ID']), set(fn_b['ID'])],set_labels = ('qEEG-R', 'qEEG'))
v.get_patch_by_id('C').set_color('orange')
#v.get_label_by_id('A').set_size(20)
#v.get_label_by_id('B').set_size(20)
for text in v.set_labels:
    text.set_fontsize(20)
for x in range(len(v.subset_labels)):
    if v.subset_labels[x] is not None:
        v.subset_labels[x].set_fontsize(20)

plt.show()

tn, fp, fn, tp = confusion_matrix(merge.y_x, merge.pred_R).ravel()
print(tn/(tn+fp),tp/(tp+fn),fp/(fp+tn),tp/(tp+fp),fp/(fp+tn))
#0.8461538461538461 0.5211267605633803 0.15384615384615385 0.8222222222222222 0.15384615384615385


tn, fp, fn, tp = confusion_matrix(merge.y_x, merge.pred_B).ravel()
print(tn/(tn+fp),tp/(tp+fn),fp/(fp+tn),tp/(tp+fp),fp/(fp+tn))
#0.9423076923076923 0.6056338028169014 0.057692307692307696 0.9347826086956522 0.057692307692307696

merge['avg_pred'] = (merge['prob_R']+merge['prob_B'])/2


tresh=np.linspace(0,0.99,num=1000)
cont=0        
spec=0
#doing same for probas_base
while spec<=0.95 and cont<1000:
            
        t=tresh[cont]
    
        p=merge['avg_pred'][:]>t
            
        tn, fp, fn, tp = confusion_matrix(merge.y_x,p).ravel()

        sens=tp/(tp+fn)
        spec=tn/(tn+fp)
        
        cont+=1
        print(spec,sens,t)

tn, fp, fn, tp = confusion_matrix(merge.y_x,merge.pred_R).ravel()
print(tn/(tn+fp),tp/(tp+fn),fp/(fp+tn))

cnf_matrix = confusion_matrix(merge.y_x, merge.pred_R)

np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Good','Poor'],
                      title='Confusion matrix: EEG-R')



cnf_matrix = confusion_matrix(merge.y_x, merge.pred_B)

np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Good','Poor'],
                      title='Confusion matrix: EEG background')




cnf_matrix = confusion_matrix(merge.y_x,merge['avg_pred'][:]>t)

np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Good','Poor'],
                      title='Confusion matrix: Combination')




#GETTING STD FOR THE TABLES

#ts = np.load(r"F:\DeepEEG\final_results_10cv\6_months_final_Results5Stim004post\Thresholds_LR.npy")






#stimulus=['Stim001','Stim002','Stim003','Stim004','Stim005']
stimulus=['Stim004']

for st in stimulus:
    
    colors=['darkorange','blue','green','black','yellow']
    book = xlwt.Workbook(encoding="utf-8")    
    sheet1 = book.add_sheet("Sheet 1")
    
    sheet1.write(0, 0, "Methods")
    sheet1.write(0, 1, "AUC 95% CI ")
    sheet1.write(0, 2, "Sensitivity")
    sheet1.write(0, 3, "Specificity")
    sheet1.write(0, 4, "PPV")
    sheet1.write(0, 5, "FPR")

    names = ['RFC','SVM','LR','XGB','NN']
    
    for k in range(0,5):
        aucs = list()
        sens = list()
        spec = list()
        ppv = list()
        fpr = list()
        l=0
        #ts1 = np.load(r"F:\DeepEEG\6_months_final_Results5"+st+"post\Thresholds_"+names[k]+".npy")
        ts1 = np.load(r"F:\DeepEEG\6_months_final_Results5"+st+"24_hours\Thresholds_"+names[k]+".npy")
        for s in range(0,10):
            for i in range(0,5):
                
                #reac = np.load(r"F:\DeepEEG\6_months_final_Results5"+st+"post\probabilities_"+names[k]+"_"+ str(i)+str(s)+".npy")
                reac = np.load(r"F:\DeepEEG\6_months_final_Results5"+st+"24_hours\probabilities_"+names[k]+"_"+ str(i)+str(s)+".npy")
        
                #df_reac = pd.read_csv(r"F:\DeepEEG\6_months_final_Results5Stim004post\frame_test_"+str(i)+".csv")
                #df_back = pd.read_csv(r"F:\DeepEEG\6_months_final_Results5Stim00424_hours\frame_test_"+str(i)+".csv")
                
                probas = reac[:,0]
                preds = reac[:,0]>ts1[l]
                y_test = reac[:,1]
                #p2 = back[:,0]>ts2[l]
                l = l+1
                tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()  
                aucs.append(roc_auc_score(y_test,probas))
                sens.append(tp/(tp+fn))
                spec.append(tn/(tn+fp))
                ppv.append(tp/(tp+fp))
                fpr.append(fp/(fp+tn))
    
        sheet1.write(k+1,0,(names[k])) 
        #sheet1.write(k+1,1,str("%0.2f ± (%0.2f)"%(np.nanmean(aucs),np.nanstd(aucs))))              
        #sheet1.write(k+1,2,str("%0.2f ± (%0.2f)"%(np.nanmean(sens),np.nanstd(sens))))              
        #sheet1.write(k+1,3,str("%0.2f ± (%0.2f)"%(np.nanmean(spec),np.nanstd(spec))))              
        #sheet1.write(k+1,4,str("%0.2f ± (%0.2f)"%(np.nanmean(ppv),np.nanstd(ppv))))              
        #sheet1.write(k+1,5,str("%0.2f ± (%0.2f)"%(np.nanmean(fpr),np.nanstd(fpr))))
        sheet1.write(k+1,1,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(aucs))))              
        sheet1.write(k+1,2,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(sens))))              
        sheet1.write(k+1,3,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(spec))))              
        sheet1.write(k+1,4,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(ppv))))              
        sheet1.write(k+1,5,str("%0.2f (%0.2f - %0.2f)"%(Mean_Confidence_Interval(fpr))))
          
    book.save(r"F:\DeepEEG\measures_std_95ci_24hours"+st+".xls")

#final=np.concatenate((rfc_m.probas[:,0].reshape(-1,1),new_label.reshape(-1,1)),axis=1)
#w=np.where(final[:,0]==0)

#final2=np.delete(final2,(w[0]),axis=0)

#a=roc_auc_score(final2[:,1],final2[:,0])

#np.save(r"\\amc.intra\users\L\laramos\home\Desktop\EEG\post2.npy",final)

#0.9142857142857143 0.625 0.6867567567567568  B_EEG
#0.8857142857142857 0.3194444444444444 0.7115315315315316 R_EEG

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix: ')

    print(cm)
    plt.rcParams.update({'font.size': 14})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

post=np.load(r"\\amc.intra\users\L\laramos\home\Desktop\EEG\post2.npy")
r_eeg=np.load(r"L:\basic\Personal Archive\L\laramos\DeepEEG\paper\REEG.npy")
b_eeg=np.load(r"\\amc.intra\users\L\laramos\home\Desktop\EEG\EEG_complete.npy")

w=np.where(r_eeg[:,0]==0)
r_eeg=np.delete(r_eeg,(w[0]),axis=0)
b_eeg=np.delete(b_eeg,(w[0]),axis=0)

w=np.where(b_eeg[:,0]==0)
r_eeg=np.delete(r_eeg,(w[0]),axis=0)
b_eeg=np.delete(b_eeg,(w[0]),axis=0)
post=np.delete(post,(w[0]),axis=0)

w=np.where(post[:,0]==0)
r_eeg=np.delete(r_eeg,(w[0]),axis=0)
b_eeg=np.delete(b_eeg,(w[0]),axis=0)
post=np.delete(post,(w[0]),axis=0)


y_test=b_eeg[:,1]
probas=b_eeg[:,0]

tresh=np.linspace(0,0.99,num=1000)
cont=0        
spec=0
#doing same for probas_base

probas = merge['avg_pred']
y_test = merge['y_x']
while spec<=0.95 and cont<1000:
            
        t=tresh[cont]
    
        p=probas[:]>t
            
        tn, fp, fn, tp = confusion_matrix(y_test,p).ravel()

        aucs = roc_auc_score(y_test,probas)
        sens = tp/(tp+fn)
        spec = tn/(tn+fp)
        ppv = tp/(tp+fp)
        fpr = fp/(fp+tn) 
        cont+=1
        print(aucs,spec,sens,ppv,fpr)
        
        
   
y_test=post[:,1]
probas=post[:,0]        
        
t_final=0.6282882882882883
p_r=probas[:]>t_final
errors_pos_reeg=list()
errors_neg_reeg=list()
for i in range(p_r.shape[0]):
    if y_test[i]==0:
        if y_test[i]!=p_r[i]:
            errors_neg_reeg.append(i)
    if y_test[i]==1:
        if y_test[i]!=p_r[i]:
            errors_pos_reeg.append(i)
            
y_test=b_eeg[:,1]
probas=b_eeg[:,0]        
        
t_final=0.6867567567567568
p_b=probas[:]>t_final
errors_pos_beeg=list()
errors_neg_beeg=list()
for i in range(p_b.shape[0]):
    if y_test[i]==0:
        if y_test[i]!=p_b[i]:
            errors_neg_beeg.append(i)
    if y_test[i]==1:
        if y_test[i]!=p_b[i]:
            errors_pos_beeg.append(i)            

#from matplotlib_venn import venn2   
#venn2(subsets = (3, 2, 3), set_labels = ('R-EEG Mistakes', 'B-EEG Mistakes'))
#plt.show()
            
            
            
            
    

v=venn2([set(errors_neg_reeg), set(errors_neg_beeg)],set_labels = ('qEEG-R', 'qEEG'))
v.get_patch_by_id('C').set_color('orange')
v.get_label_by_id('A').set_size(20)
v.get_label_by_id('B').set_size(20)
plt.show()

venn2([set(errors_pos_reeg), set(errors_pos_beeg)],set_labels = ('qEEG-R', 'qEEG'))
v.get_patch_by_id('C').set_color('orange')
v.get_label_by_id('A').set_size(20)
v.get_label_by_id('B').set_size(20)
plt.show()

print(roc_auc_score(y_test,p_r))

print(roc_auc_score(y_test,p_b))

p_final=np.zeros(p_r.shape[0])

for i in range(p_r.shape[0]):
    p_final[i]=p_r[i] or p_b[i]

pf_2=(b_eeg[:,0] +post[:,0] )/2
    
print(roc_auc_score(y_test,p_final))
print(roc_auc_score(y_test,pf_2))


tn, fp, fn, tp = confusion_matrix(y_test,p_final).ravel()
 
sens=tp/(tp+fn)
spec=tn/(tn+fp)
print(sens,spec)

tn, fp, fn, tp = confusion_matrix(y_test,p_r).ravel()
 
sens=tp/(tp+fn)
spec=tn/(tn+fp)
print(sens,spec)

tn, fp, fn, tp = confusion_matrix(y_test,p_b).ravel()
 
sens=tp/(tp+fn)
spec=tn/(tn+fp)
print(sens,spec)

tn, fp, fn, tp = confusion_matrix(y_test,pf_2).ravel()
 
sens=tp/(tp+fn)
spec=tn/(tn+fp)
print(sens,spec)


cnf_matrix = confusion_matrix(y_test,p_r)

np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Good','Poor'],
                      title='Confusion matrix: qEEG-R')


cnf_matrix = confusion_matrix(y_test,p_b)

np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Good','Poor'],
                      title='Confusion matrix: qEEG')

cnf_matrix = confusion_matrix(y_test,p_final)

np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Good','Poor'],
                      title='Confusion matrix, without normalization')


cnf_matrix = confusion_matrix(y_test,pf_2)

np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Good','Poor'],
                      title='Confusion matrix: Combination')



probas=pf_2[:]

tresh=np.linspace(0,0.99,num=1000)
cont=0        
spec=0
#doing same for probas_base
while spec<=0.95 and cont<1000:
            
        t=tresh[cont]
    
        p=probas[:]>t
            
        tn, fp, fn, tp = confusion_matrix(y_test,p).ravel()

        sens=tp/(tp+fn)
        spec=tn/(tn+fp)
        
        cont+=1
        print(spec,sens,t)

   
        
t_final=0.6401801801801802
t_final=0.667927927927928

pf_f=pf_2[:]>t_final

cnf_matrix = confusion_matrix(y_test,pf_f)

np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Good','Poor'],
                      title='Confusion matrix, without normalization')


final_dataframe=pd.DataFrame()
final_dataframe['qEEG-R_probas']=post[:,0]
final_dataframe['qEEG-R_pred']=post[:,0]>0.6282882882882883
final_dataframe['qEEG_probas']=b_eeg[:,0]
final_dataframe['qEEG_pred']=b_eeg[:,0]>0.6867567567567568
final_dataframe['Comb_probas']=pf_2[:]
final_dataframe['Comb_pred']=pf_2[:]>0.667927927927928
final_dataframe['labels']=post[:,1]
final_dataframe.to_csv(r"\\amc.intra\users\L\laramos\home\Desktop\EEG\predictions.csv",index=False)





















#y_test = np.load(r"E:\DeepEEG\verne_final_Results5Stim001post\original_mrs_test00.npy")

ts_REEG = np.load(r"E:\DeepEEG\verne_final_Results5Stim004post\Thresholds_RFC.npy")
ts_EEG = np.load(r"E:\DeepEEG\verne_final_Results5Stim00424_hours\Thresholds_RFC.npy")
i=0
data_REEG = np.load(r"E:\DeepEEG\verne_final_Results5Stim004post\probabilities_RFC_0.npy")
probas_REEG = data_REEG[:,0] > ts_REEG[i]
y_test_REEG = data_REEG[:,1]

X_test_REEG = np.load(r'E:\DeepEEG\verne_final_Results5Stim004post\\'+"test_pat"+str(i)+".npy")

data_EEG = np.load(r"E:\DeepEEG\verne_final_Results5Stim00424_hours\probabilities_RFC_0.npy")
probas_EEG = data_EEG[:,0] > ts_EEG[i]
y_test_EEG = data_EEG[:,1]

X_test_EEG = np.load(r'E:\DeepEEG\verne_final_Results5Stim00424_hours\\'+"test_pat"+str(i)+".npy")

for i in range(1,5):

    data_REEG = np.load(r"E:\DeepEEG\verne_final_Results5Stim004post\probabilities_RFC_"+str(i)+".npy")
    test_REEG = np.load(r'E:\DeepEEG\verne_final_Results5Stim004post\\'+"test_pat"+str(i)+".npy")
    probas_REEG = np.concatenate((probas_REEG,data_REEG[:,0] > ts_REEG[i]),axis=0)
    y_test_REEG = np.concatenate((y_test_REEG,data_REEG[:,1]),axis=0)
    X_test_REEG = np.concatenate((X_test_EEG,test_REEG),axis=0)
    
    data_EEG = np.load(r"E:\DeepEEG\verne_final_Results5Stim00424_hours\probabilities_RFC_"+str(i)+".npy")
    test_EEG = np.load(r'E:\DeepEEG\verne_final_Results5Stim00424_hours\\'+"test_pat"+str(i)+".npy")
    probas_EEG = np.concatenate((probas_EEG,data_EEG[:,0] > ts_EEG[i]),axis=0)
    y_test_EEG = np.concatenate((y_test_EEG,data_EEG[:,1]),axis=0)
    X_test_EEG = np.concatenate((X_test_EEG,test_EEG),axis=0)
     
tn, fp, fn, tp = confusion_matrix(y_test_REEG,probas_REEG).ravel()
cnf_matrix = confusion_matrix(y_test_REEG,probas_REEG)
sens_reeg = tp/(tp+fn)
spec_reeg = tn/(tn+fp)
plt.figure()
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, classes=['Good','Poor'],
                      title='Confusion matrix: qEEG-R')

tn, fp, fn, tp = confusion_matrix(y_test_EEG,probas_EEG).ravel()
cnf_matrix = confusion_matrix(y_test_EEG,probas_EEG)
sens_eeg = tp/(tp+fn)
spec_eeg = tn/(tn+fp)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, classes=['Good','Poor'],
                      title='Confusion matrix: 24-hour EEG')

errors_neg_reeg = list()
errors_neg_eeg = list()

errors_pos_reeg = list()
errors_pos_eeg = list()

for i in range(y_test_REEG.shape[0]):    
    if y_test_REEG[i]==0:
        if y_test_REEG[i]!=probas_REEG[i]:
            errors_neg_reeg.append(X_test_REEG[i])
          
            
    if y_test_REEG[i]==1:
        if y_test_REEG[i]!=probas_REEG[i]:
            errors_pos_reeg.append(X_test_REEG[i])
            
for i in range(y_test_EEG.shape[0]):    
    if y_test_EEG[i]==0:
        if y_test_EEG[i]!=probas_EEG[i]:
            errors_neg_eeg.append(X_test_EEG[i])                        
    if y_test_EEG[i]==1:
        if y_test_EEG[i]!=probas_EEG[i]:
            errors_pos_eeg.append(X_test_EEG[i]) 
            

v=venn2([set(errors_neg_reeg), set(errors_neg_eeg)],set_labels = ('qEEG-R', 'qEEG'))
v.get_patch_by_id('C').set_color('orange')
v.get_label_by_id('A').set_size(20)
v.get_label_by_id('B').set_size(20)
plt.show()


v=venn2([set(errors_pos_reeg), set(errors_pos_eeg)],set_labels = ('qEEG-R', 'qEEG'))
v.get_patch_by_id('C').set_color('orange')
v.get_label_by_id('A').set_size(20)
v.get_label_by_id('B').set_size(20)
plt.show()




pf_f = np.logical_and(probas_REEG,probas_EEG)
cnf_matrix = confusion_matrix(y_test_EEG,pf_f)

np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Good','Poor'],
                      title='Confusion matrix: Combination')




for i in range(0,y_test_REEG.shape[0]):
    if y_test_REEG[i] != y_test_EEG[i]:
        print(i)


y_test_EEG = np.delete(y_test_EEG,26)
probas_EEG = np.delete(probas_EEG,26)
y_test_EEG = np.delete(y_test_EEG,82)
probas_EEG = np.delete(probas_EEG,82)

tot=list()
for name in X:
    if name not in X_test_EEG:    
            print(name)
            tot.append(name)





