# -*- coding: utf-8 -*-
"""

This file creates ROc curves and feature importance  curves

@author: laramos
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import pandas as pd


def Compute_Odds(odds):

    final_odds=np.zeros(odds.shape[1])
    for j in range(0,odds.shape[1]):
        cont=0
        for i in range(0,odds.shape[0]):
            if odds[i,j]!=0:
                cont=cont+1
                final_odds[j]=final_odds[j]+odds[i,j]
            
        
        if cont>0:
            final_odds[j]=final_odds[j]/cont
    return(final_odds)

            
            

def Plot_ROC(path_ml,path_logit,pdf_title,path_prior,all_variables,lr_test,posttici):

    
    if all_variables:
        mean_auc_sl = 0.92
        mean_auc_rfc = 0.92
        mean_auc_svm = 0.90
        mean_auc_log = 0.90
        mean_auc_nn = 0.90
        mean_auc_prior = 0.89      
        name_curve='Average ROC using all variables available'
    else:
        mean_auc_sl = 0.83
        mean_auc_rfc = 0.84
        mean_auc_svm = 0.79
        mean_auc_log = 0.79
        mean_auc_nn = 0.79
        mean_auc_prior = 0.78
        name_curve='Average ROC using Baseline variables available'
    if posttici:
        mean_auc_sl = 0.65
        mean_auc_rfc = 0.66
        mean_auc_svm = 0.62
        mean_auc_log = 0.65
        mean_auc_nn = 0.63
        mean_auc_prior = 0.89       
        
    fpr_svm=np.load(path_ml+'fprsvm.npy')
    tpr_svm=np.load(path_ml+'tprsvm.npy')
    
    fpr_rfc=np.load(path_ml+'fprrfc.npy')
    tpr_rfc=np.load(path_ml+'tprrfc.npy')
    
    fpr_log=np.load(path_logit+'fprLogit.npy')
    tpr_log=np.load(path_logit+'tprLogit.npy')
    
    if posttici:
        fpr_nn=np.load(path_ml+'fprNN.npy')
        tpr_nn=np.load(path_ml+'tprNN.npy')
    else:
        fpr_nn=np.load(path_ml+'fpr_nn_tf.npy')
        tpr_nn=np.load(path_ml+'tpr_nn_tf.npy')
    
    
    fpr_sl=np.load(path_ml+'fprSL.npy')
    tpr_sl=np.load(path_ml+'tprSL.npy')
    
    fpr_prior=np.load(path_ml+'fprLogit.npy')
    tpr_prior=np.load(path_ml+'tprLogit.npy')

    f,ax=plt.subplots(figsize=(10,10))
 
    lw=2

    #mean_auc_rfc = auc(fpr_rfc, tpr_rfc)
    
    
    ax.plot(fpr_rfc, tpr_rfc, color='darkblue',lw=lw, linestyle=':',marker='v', label='Random Forest (area = %0.2f)' % mean_auc_rfc,markersize=5,linewidth=1.0)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    
    #mean_auc_sl = auc(fpr_sl, tpr_sl)

    
    ax.plot(fpr_sl, tpr_sl, color='magenta',lw=lw,marker='|', label='Super Learner (area = %0.2f)' % mean_auc_sl,markersize=10,linewidth=1.0)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw)
    
   # mean_auc_svm = auc(fpr_svm, tpr_svm)

    
    ax.plot(fpr_svm, tpr_svm, color='darkorange',lw=lw,marker='.', label='Support Vector Machine (area = %0.2f)' % mean_auc_svm,markersize=7,linewidth=1.0)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    
    
    #mean_auc_log = auc(fpr_log, tpr_log)
   
    
    ax.plot(fpr_log, tpr_log, color='darkgreen',lw=lw, label=(lr_test+' (area = %0.2f)') % mean_auc_log,markersize=5,linewidth=1.0)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw)
    
    #mean_auc_nn = auc(fpr_nn, tpr_nn)
  
    
    
    ax.plot(fpr_nn, tpr_nn, color='black',lw=lw,marker='x',linestyle='-.', label='Neural Network (area = %0.2f)' % mean_auc_nn,markersize=5,linewidth=1.0)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    
    #ax.plot(fpr_prior, tpr_prior, color='blue',lw=lw,marker='4',linestyle='-.', label='Prior knowledge (area = %0.2f)' % mean_auc_prior,markersize=10,linewidth=1.0)
    #ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    
     
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    
   
    #plt.axis([0,1,0,1])
    ax.set_xlabel('False Positive Rate',fontsize=14)
    ax.set_ylabel('True Positive Rate',fontsize=14)
    #ax.set_title('Average ROC from Machine Learning experiments')
    #ax.set_title(name_curve,fontsize=12)
    ax.legend(loc="lower right",fontsize=14)
    
    #ax.show() 
    fig = ax.get_figure()
    fig.savefig(pdf_title, format='pdf')
    
    #fig.savefig("test.tiff", format='tiff')
    




#All variables
path_ml="E:\\Results MRs\\Center-All_vars_contscore-mrs\\"
path_logit="E:\\Results MRs\\Logistic Regression Results\\LR-Center-All_vars_contscore-mrs-back\\"
path_prior="E:\\Results MRs\\Logistic Regression Results\\LR-Center-Knowledge_all-mrs\\"
lr_test="LR: Back. Elim."
pdf_title="All_Variables.pdf"    
Plot_ROC(path_ml,path_logit,pdf_title,path_prior,True,lr_test,False)  


#Baseline
path_ml="E:\\Results MRs\\Center-Baseline_contscore-mrs\\"
path_logit="E:\\Results MRs\\Logistic Regression Results\\LR-Center-Baseline_contscore-mrs-rfc\\"
path_prior="E:\\Results MRs\\Logistic Regression Results\\LR-Center-Knowledge_baseline-mrs\\"
lr_test="LR: Random Forest"
pdf_title="Baseline_Variables.pdf"    
Plot_ROC(path_ml,path_logit,pdf_title,path_prior,False,lr_test,False) 
  
#E:\Results MRs\Results postici\Center-Baseline_contscore-posttici_c-rfc
#Postici
path_ml="E:\\Results MRs\\Results postici\\Center-Baseline_contscore-posttici_c-rfc\\"
path_logit="E:\\Results MRs\\Results postici\\LR-Center-Baseline_contscore-posttici_c-lasso\\"
path_prior="E:\\Results MRs\\Logistic Regression Results\\LR-Center-Knowledge_baseline-mrs\\"
lr_test="LR: Lasso"
pdf_title="Postici.pdf"    
Plot_ROC(path_ml,path_logit,pdf_title,path_prior,False,lr_test,True) 



"""
----------------------
This is to organize and order feature importance files

"""

#imp=np.load(r"C:\Users\laramos\Dropbox\Mrcleanlearning\results\Center-Baseline_contscore-mrs-rfc\Feat_ImportanceRFC.npy")
#cols=np.load(r"C:\Users\laramos\Dropbox\Mrcleanlearning\results\Center-Baseline_contscore-mrs-rfc\cols.npy")
#imp=np.load(r"E:\Mrclean\Results\Center-All_contscore-mrs-rfc\Feat_ImportanceRFC.npy")
#cols=np.load(r"E:\Mrclean\Results\Center-All_contscore-mrs-rfc\cols.npy")


base=r'C:\Users\laramos\Dropbox\Mrcleanlearning\results\New100-All_contscore-mrs-rfc'
cols=np.load(base+"\cols.npy")
cols=pd.Index(cols)

imp_lr=np.load(base+"\Feat_ImportanceLR.npy")
imp_lr=Compute_Odds(imp_lr)
data_lr = pd.Series(imp_lr, index=cols)
data_lr.to_csv(base+"\LR_featIMP.csv")

imp_rfc=np.load(base+"\Feat_ImportanceRFC.npy")
imp_rfc=np.mean(imp_rfc,axis=0)
data_rfc = pd.Series(imp_rfc, index=cols)
data_rfc=data_rfc.sort_values()
data_rfc.to_csv(base+"\RFC_featIMP.csv")

#Top 15 with silvias comments
imp_rfc=np.load(base+"\Feat_ImportanceRFC.npy")
iterations=imp_rfc.shape[0]
features=imp_rfc.shape[1]

top_15=np.zeros((1,imp_rfc.shape[1]))
top_15=pd.DataFrame(top_15,columns=cols)


for i in range(0,iterations):
    data_rfc = pd.Series(imp_rfc[i,:], index=cols)
    data_rfc=data_rfc.sort_values(ascending =False)
    data_rfc=data_rfc[0:15].index
    top_15[data_rfc]=top_15[data_rfc]+1

#s=np.zeros((1,imp_rfc.shape[1]))
imp_rfc=np.mean(imp_rfc,axis=0)
imp_rfc=imp_rfc.reshape(1,-1)
imp_rfc=pd.DataFrame(imp_rfc,columns=cols)
top_15=top_15.append(imp_rfc)

for i in range(0,features):
    if top_15.iloc[0][cols[i]]<=0:
        top_15=top_15.drop(cols[i],axis=1)

top_15=top_15.transpose()
top_15.columns=['a','b']
top_15 = top_15.sort_values(by='a',ascending=False)
top_15 = top_15[0:15]
top_15 = top_15.sort_values(by='b',ascending=False)
top_15.to_csv(base+"\RFC_15feat.csv")




features = cols
importances = imp_rfc
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


importances = pd.Series(importances, index=cols)
importances.nlargest(12,columns=cols).plot(kind='barh')



#plots the feature importance with only one resutllt
cols=np.load(r"L:\basic\Personal Archive\L\laramos\DeepEEG\Results\cols.npy")
base=r"F:\DeepEEG\6_months_final_Results5Stim001pre"
####Plotting the figure for importance
imp_rfc=np.load(base+"\Feat_ImportanceRFC.npy")
imp_rfc=np.mean(imp_rfc,axis=0)

feat_imp = pd.DataFrame({'importance':imp_rfc})    
feat_imp['feature'] = cols
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
feat_imp.sort_values(by='importance', inplace=True)
feat_imp = feat_imp.set_index('feature', drop=True)

f=plt.figure()
feat_imp.plot.barh(title="Average Feature Ranking Importance", figsize=(8,8))
plt.xlabel('Importance')
#plt.show()
plt.savefig(r"F:\DeepEEG\Results\foo.pdf", bbox_inches='tight')






#plot feature importance combining two results


import numpy as np
import matplotlib.pyplot as plt

cols=np.load(r"L:\basic\Personal Archive\L\laramos\DeepEEG\Results\cols.npy")
base1=r"F:\DeepEEG\final_results_10cv\6_months_final_Results5Stim004post"
#base2=r"F:\DeepEEG\final_results_10cv\6_months_final_Results5Stim00424_hours"
base2 = r"F:\DeepEEG\final_results_10cv\6_months_final_Results5Stim00124_hours"

N = cols.shape[0]
imp_rfc=np.load(base1+"\Feat_ImportanceRFC.npy")
men_means = np.mean(imp_rfc,axis=0)
men_std = np.std(imp_rfc,axis=0)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars



imp_rfc2=np.load(base2+"\Feat_ImportanceRFC.npy")
women_means = np.mean(imp_rfc2,axis=0)
women_std = np.std(imp_rfc2,axis=0)

feat_imp = pd.DataFrame({'importance_qeegr':men_means})    
feat_imp['features'] = cols
feat_imp['importance_qeeg'] = women_means

df=feat_imp.reindex([9,11,8,6,7,5,0,2,1,3,4,10])

feat_imp=df
#feat_imp.sort_values(by='features', inplace=True,ascending=False)


fig, ax = plt.subplots(figsize=(9,6))
rects1 = ax.barh(ind, feat_imp['importance_qeegr'], width, color='g')
rects2 = ax.barh(ind + width, feat_imp['importance_qeeg'], width, color='b')

# add some text for labels, title and axes ticks
ax.set_xlabel('Feature Importance')
ax.set_title('Ranking - Average Feature Importance')
ax.set_yticks(ind + width / 2)
ax.set_yticklabels(feat_imp['features'])

ax.legend((rects1[0], rects2[0]), ('qEEG-R', '24h EEG'))

plt.savefig(r"F:\feature_importance.pdf", bbox_inches='tight')




np.random.seed(10)
arr = np.arange(10)
np.random.shuffle(arr)
print(arr)



