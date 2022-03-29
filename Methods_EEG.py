

import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV                               
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import brier_score_loss,f1_score
import random as rand
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import time
from scipy.stats import randint as sp_randint
import scipy as sp

from scipy import interp 
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import RFECV
from sklearn.naive_bayes import GaussianNB
import xlwt
from sklearn.metrics import make_scorer
import xgboost as xgb

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. 
    """    
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()

    #sens=mat[0,0]/(mat[0,0]+mat[1,0])

    spec=tn/(tn+fp)
    
    # Return the score
    return(spec)
    
#Public variables    
scoring_fnc = make_scorer(performance_metric)

score = scoring_fnc

n_jobs=-2

random_state=1

class_weight=False

#scores=['roc_auc']
#scores=['f1']
scores=['precision']


class XGBoost_Pipeline: 
 
    def RandomGridSearchXGBoost(self,x_train,y_train,x_test,y_test,splits,path_results,xgb_m,itera):
        """
        This function looks for the best set o parameters for RFC method
        Input: 
            X: training set
            Y: labels of training set
            splits: cross validation splits, used to make sure the parameters are stable
        Output:
            clf.best_params_: dictionary with the parameters, to use: param_svm['kernel']
        """
        
        start_xgb = time.time()  
        
        tuned_parameters = {
        'learning_rate': ([0.1, 0.01, 0.001]),
        #'gamma': ([100,10,1, 0.1, 0.01, 0.001]),                  
        #'max_depth':    ([3,5,10,15]),
        #'subsample ':    ([0.5,1]),
        #'reg_lambda ':  [1,10,100],
        #'alpha ':   [1,10,100],
        
        'min_child_weight': [1, 5, 10],
        'gamma': [0, 0.5, 1, 1.5, 2, 5],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.3,0.4,0.5,0.6,0.7,0.8],
        'max_depth': [3, 5, 7, 9, 10]
        
        }
        
    

            
        clf_grid=xgb.XGBClassifier()
        print("XGB Grid Search")
        clf_grid =  RandomizedSearchCV(clf_grid, tuned_parameters, cv=splits,
                           scoring='%s' % scores[0],n_jobs=n_jobs,random_state=1)
        
        clf_grid=clf_grid.fit(x_train, y_train)
              
        #print("Score",clf.best_score_)
        end_xgb = time.time()
        print("Total time to process XGB: ",end_xgb - start_xgb)
        
        with open(path_results+"parameters_xgb.txt", "a") as file:
            for item in clf_grid.best_params_:
              file.write(" %s %s " %(item,clf_grid.best_params_[item] ))
            file.write("\n")
        clf=xgb.XGBClassifier(**clf_grid.best_params_,random_state=random_state)
          
        clf.fit(x_train, y_train)
        
        preds = clf.predict(x_test)
        
        probas = clf.predict_proba(x_test)[:, 1]
        
        probas_train = clf.predict_proba(x_train)[:, 1]

        xgb_m.clf_auc[itera]=roc_auc_score(y_test,probas)
        
        fpr_rf, tpr_rf, t = roc_curve(y_test, probas)  
        
        
        #Here we optimize the classifier for the threshold for f1_Score and specificity
        ts=np.linspace(0.1, 0.99, num=100)
        best_val=0
        best_t=0
        t_spec=0
        found=False
        for i in range(ts.shape[0]):
            p=probas_train>ts[i]
            c_f1=f1_score(y_train, p)
            tn, fp, fn, tp = confusion_matrix(y_train, p).ravel()
            c_spec=tn/(tn+fp)
            if c_f1>best_val:
                best_val=c_f1
                best_t=ts[i]
            if c_spec>=0.95 and not found:
                t_spec=ts[i]
                found=True 
                print(c_spec)
    
        print(t_spec,best_t)        
        xgb_m.clf_thresholds.append(t)
        
        xgb_m.clf_brier[itera] = brier_score_loss(y_test, probas)   
        
        xgb_m.feat_imp.append(clf.feature_importances_)
        
        xgb_m.clf_f1_score[itera]=f1_score(y_test, probas>=0.5)
        tn, fp, fn, tp = confusion_matrix(y_test, probas>=0.5).ravel()        
        xgb_m.clf_sens[itera]=tp/(tp+fn)
        xgb_m.clf_spec[itera]=tn/(tn+fp)
        
        xgb_m.f1_score_f1[itera]=f1_score(y_test, probas>best_t)        
        tn, fp, fn, tp = confusion_matrix(y_test, probas>best_t).ravel()        
        xgb_m.sens_f1[itera]=tp/(tp+fn)
        xgb_m.spec_f1[itera]=tn/(tn+fp)
        
        xgb_m.f1_score_spec[itera]=f1_score(y_test, probas>t_spec)
        tn, fp, fn, tp = confusion_matrix(y_test, probas>t_spec).ravel()        
        xgb_m.sens_spec[itera]=tp/(tp+fn)
        xgb_m.spec_spec[itera]=tn/(tn+fp)

        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\xgbsens_"+str(itera)+".npy",xgb_m.sens_spec[itera])
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\xgbspec_"+str(itera)+".npy",xgb_m.spec_spec[itera])
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\xgbauc_"+str(itera)+".npy",xgb_m.clf_auc[itera])
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\xgbf1_"+str(itera)+".npy",xgb_m.f1_score_spec[itera])
        #rfc_m.clf_ppv[itera]=tp/(tp+fp)
        
        #rfc_m.clf_npv[itera]=tn/(tn+fn)                        
            
        return(fpr_rf,tpr_rf,probas,clf,preds)
        
        
        
    def __init__(self,run,x_train,y_train,x_test,y_test,itera,cv,mean_tprr,xgb_m,path_results):
        if run:
            self.name='XGB'
            xgb_m.run=True
            fpr_rf,tpr_rf,probas_t,self.clf,preds=self.RandomGridSearchXGBoost(x_train,y_train,x_test,y_test,cv,path_results,xgb_m,itera)
            print("Done Grid Search")
            print("Done testing - XGB", xgb_m.clf_auc[itera])
            np.save(path_results+"XGB_probas_"+str(itera)+".npy",probas_t)
            #np.save(path_results+"rfc_probas_train.npy",train_p)
            #np.save(path_results+"rfc_pre.npy",preds)
            mean_fpr = np.linspace(0, 1, 100) 
            xgb_m.mean_tpr += interp(mean_fpr, fpr_rf,tpr_rf)
            xgb_m.mean_tpr[0] = 0.0
        else:
            self.name='NONE'
            self.clf=0



class RFC_Pipeline: 
 
    def RandomGridSearchRFC(self,x_train,y_train,x_test,y_test,splits,path_results,rfc_m,itera):
        """
        This function looks for the best set o parameters for RFC method
        Input: 
            X: training set
            Y: labels of training set
            splits: cross validation splits, used to make sure the parameters are stable
        Output:
            clf.best_params_: dictionary with the parameters, to use: param_svm['kernel']
        """    
        
        start_rfc = time.time()  
        
        tuned_parameters = {
        'n_estimators': ([200,400,500,600,800,1000,1200,1400]),
        'max_features': (['auto', 'sqrt', 'log2']),                   # precomputed,'poly', 'sigmoid'
        'max_depth':    ([10,20,30,40, 50, 60, 70, 80, 90, 100, None]),
        'criterion':    (['gini', 'entropy']),
        'min_samples_split':  [2,4,6,8],
        'min_samples_leaf':   [2,4,6,8,10]
        }
        if class_weight:
            rfc = RandomForestClassifier(n_estimators=25, oob_score = True,random_state=random_state,class_weight='balanced')   
        else:
            rfc = RandomForestClassifier(n_estimators=25, oob_score = True,random_state=random_state)   
        print("RFC Grid Search")
        #clf_grid =  RandomizedSearchCV(rfc, tuned_parameters, cv=splits,
        #                   scoring=score,n_jobs=n_jobs)
        clf_grid =  RandomizedSearchCV(rfc, tuned_parameters, cv=splits,
                           scoring='%s' % scores[0],n_jobs=n_jobs)
        
                                  
        clf_grid.fit(x_train, y_train)
        #print("Score",clf.best_score_)
        end_rfc = time.time()
        print("Total time to process: ",end_rfc - start_rfc)
        
        with open(path_results+"parameters_rfc.txt", "a") as file:
            for item in clf_grid.best_params_:
              file.write(" %s %s " %(item,clf_grid.best_params_[item] ))
            file.write("\n")
    
        if class_weight:            
            clf = RandomForestClassifier(**clf_grid.best_params_,random_state=random_state,class_weight='balanced')
        else:
            clf = RandomForestClassifier(**clf_grid.best_params_,random_state=random_state)
        
        clf.fit(x_train, y_train)
        
        preds = clf.predict(x_test)
        
        probas = clf.predict_proba(x_test)[:, 1]
        
        probas_train = clf.predict_proba(x_train)[:, 1]

        rfc_m.clf_auc[itera]=roc_auc_score(y_test,probas)
        
        fpr_rf, tpr_rf, t = roc_curve(y_test, probas)  
        
        
        #Here we optimize the classifier for the threshold for f1_Score and specificity
        ts=np.linspace(0.1, 0.99, num=100)
        best_val=0
        best_t=0
        t_spec=0
        found=False
        for i in range(ts.shape[0]):
            p=probas_train>ts[i]
            c_f1=f1_score(y_train, p)
            tn, fp, fn, tp = confusion_matrix(y_train, p).ravel()
            c_spec=tn/(tn+fp)
            if c_f1>best_val:
                best_val=c_f1
                best_t=ts[i]
            if c_spec>=0.95 and not found:
                t_spec=ts[i]
                found=True 
                print(c_spec)
    
        print(t_spec,best_t)        
        rfc_m.clf_thresholds.append(t)
        
        rfc_m.clf_brier[itera] = brier_score_loss(y_test, probas)   
        
        rfc_m.feat_imp.append(clf.feature_importances_)
        
        rfc_m.clf_f1_score[itera]=f1_score(y_test, probas>=0.5)
        tn, fp, fn, tp = confusion_matrix(y_test, probas>=0.5).ravel()        
        rfc_m.clf_sens[itera]=tp/(tp+fn)
        rfc_m.clf_spec[itera]=tn/(tn+fp)
        
        rfc_m.f1_score_f1[itera]=f1_score(y_test, probas>best_t)        
        tn, fp, fn, tp = confusion_matrix(y_test, probas>best_t).ravel()        
        rfc_m.sens_f1[itera]=tp/(tp+fn)
        rfc_m.spec_f1[itera]=tn/(tn+fp)
        
        rfc_m.f1_score_spec[itera]=f1_score(y_test, probas>t_spec)
        tn, fp, fn, tp = confusion_matrix(y_test, probas>t_spec).ravel()        
        rfc_m.sens_spec[itera]=tp/(tp+fn)
        rfc_m.spec_spec[itera]=tn/(tn+fp)

        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\rfcsens_"+str(itera)+".npy",rfc_m.sens_spec[itera])
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\rfcspec_"+str(itera)+".npy",rfc_m.spec_spec[itera])
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\rfcauc_"+str(itera)+".npy",rfc_m.clf_auc[itera])
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\rfcf1_"+str(itera)+".npy",rfc_m.f1_score_spec[itera])
        #rfc_m.clf_ppv[itera]=tp/(tp+fp)
        
        #rfc_m.clf_npv[itera]=tn/(tn+fn)                        
            
        return(fpr_rf,tpr_rf,probas,clf,preds)
        
        
        
    def __init__(self,run,x_train,y_train,x_test,y_test,itera,cv,mean_tprr,rfc_m,path_results):
        if run:
            self.name='RFC'
            rfc_m.run=True
            fpr_rf,tpr_rf,probas_t,self.clf,preds=self.RandomGridSearchRFC(x_train,y_train,x_test,y_test,cv,path_results,rfc_m,itera)
            print("Done Grid Search")
            print("Done testing - RFC", rfc_m.clf_auc[itera])
            np.save(path_results+"rfc_probas_"+str(itera)+".npy",probas_t)
            #np.save(path_results+"rfc_probas_train.npy",train_p)
            #np.save(path_results+"rfc_pre.npy",preds)
            mean_fpr = np.linspace(0, 1, 100) 
            rfc_m.mean_tpr += interp(mean_fpr, fpr_rf,tpr_rf)
            rfc_m.mean_tpr[0] = 0.0
        else:
            self.name='NONE'
            self.clf=0

class SVM_Pipeline: 

    def RandomGridSearchSVM(self,x_train,y_train,x_test,y_test,splits,path_results,svm_m,itera):
        """
        This function looks for the best set o parameters for RFC method
        Input: 
            X: training set
            Y: labels of training set
            splits: cross validation splits, used to make sure the parameters are stable
        Output:
            clf.best_params_: dictionary with the parameters, to use: param_svm['kernel']
        """    
        
        start_svm = time.time()  
    
        tuned_parameters = {
        'C':            ([0.1, 0.01, 0.001, 1, 10, 100]),
        'kernel':       ['linear', 'rbf','poly'],                
        'degree':       ([1,2,3,4,5,6]),
        'gamma':         [1, 0.1, 0.01, 0.001, 0.0001]
        #'tol':         [1, 0.1, 0.01, 0.001, 0.0001],
        }
        
  
        print("SVM Grid Search")
  
        if class_weight:
            clf_grid =  RandomizedSearchCV(SVC(class_weight='balanced'), tuned_parameters, cv=splits,scoring='%s' % scores[0],n_jobs=n_jobs)    
        else:
            clf_grid =  RandomizedSearchCV(SVC(), tuned_parameters, cv=splits,scoring='%s' % scores[0],n_jobs=n_jobs)    
        
       # clf_grid =  RandomizedSearchCV(SVC(class_weight=class_weight), tuned_parameters, cv=splits,
       #                scoring=score,n_jobs=n_jobs) 
        clf_grid.fit(x_train, y_train)
    
        end_svm = time.time()
        print("Total time to process: ",end_svm - start_svm)
        #print("Score",clf.best_score_)
        with open(path_results+"parameters_svm.txt", "a") as file:
            for item in clf_grid.best_params_:
              file.write(" %s %s " %(item,clf_grid.best_params_[item] ))
            file.write("\n")
            
            #,class_weight=class_weight
        if class_weight:
            clf = SVC(**clf_grid.best_params_,random_state=random_state,class_weight='balanced')
        else:
            clf = SVC(**clf_grid.best_params_,random_state=random_state)

        
        clf.fit(x_train, y_train)
            
        decisions = clf.decision_function(x_test)
        probas=\
        (decisions-decisions.min())/(decisions.max()-decisions.min())
        
        decisions = clf.decision_function(x_train)
        probas_train=\
        (decisions-decisions.min())/(decisions.max()-decisions.min())
        
                
        #Here we optimize the classifier for the threshold for f1_Score and specificity
        ts=np.linspace(0.1, 0.99, num=100)
        best_val=0
        best_t=0
        t_spec=0
        found=False
        for i in range(ts.shape[0]):
            p=probas_train>ts[i]
            c_f1=f1_score(y_train, p)
            tn, fp, fn, tp = confusion_matrix(y_train, p).ravel()
            c_spec=tn/(tn+fp)
            if c_f1>best_val:
                best_val=c_f1
                best_t=ts[i]
            if c_spec>=0.95 and not found:
                t_spec=ts[i]
                
        svm_m.clf_f1_score[itera]=f1_score(y_test, probas>=0.5)
        tn, fp, fn, tp = confusion_matrix(y_test, probas>=0.5).ravel()        
        svm_m.clf_sens[itera]=tp/(tp+fn)
        svm_m.clf_spec[itera]=tn/(tn+fp)
        
        svm_m.f1_score_f1[itera]=f1_score(y_test, probas>best_t)        
        tn, fp, fn, tp = confusion_matrix(y_test, probas>best_t).ravel()        
        svm_m.sens_f1[itera]=tp/(tp+fn)
        svm_m.spec_f1[itera]=tn/(tn+fp)
        
        svm_m.f1_score_spec[itera]=f1_score(y_test, probas>t_spec)
        tn, fp, fn, tp = confusion_matrix(y_test, probas>t_spec).ravel()        
        svm_m.sens_spec[itera]=tp/(tp+fn)
        svm_m.spec_spec[itera]=tn/(tn+fp)
        
        svm_m.clf_auc[itera]=roc_auc_score(y_test,probas)
    
        fpr_rf, tpr_rf, _ = roc_curve(y_test, probas)  
        
        svm_m.clf_brier[itera] = brier_score_loss(y_test, probas)   
                
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\svmsens_"+str(itera)+".npy",svm_m.sens_spec[itera])
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\svmspec_"+str(itera)+".npy",svm_m.spec_spec[itera])
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\svmauc_"+str(itera)+".npy",svm_m.clf_auc[itera])
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\svmf1_"+str(itera)+".npy",svm_m.f1_score_spec[itera])
                
        #svm_m.clf_ppv[itera]=tp/(tp+fp)
        
        #svm_m.clf_npv[itera]=tn/(tn+fn)  
            
        return(fpr_rf,tpr_rf,probas,clf)
        

        
    def __init__(self,run,x_train,y_train,x_test,y_test,itera,svm_m,cv,mean_tprr,path_results):
        if run:
            self.name='SVM'
            svm_m.run=True
            fpr_rf,tpr_rf,probas_t,self.clf=self.RandomGridSearchSVM(x_train,y_train,x_test,y_test,cv,path_results,svm_m,itera)            
            print("Done testing - SVM", svm_m.clf_auc[itera])
            np.save(path_results+"svm_probas_"+str(itera)+".npy",probas_t)
            mean_fpr = np.linspace(0, 1, 100) 
            svm_m.mean_tpr += interp(mean_fpr, fpr_rf,tpr_rf)
            svm_m.mean_tpr[0] = 0.0
        else:
            self.name='NONE'
            self.clf=0

class NN_Pipeline: 

    def RandomGridSearchNN(self,x_train,y_train,x_test,y_test,splits,path_results,nn_m,itera):
        """
        This function looks for the best set o parameters for RFC method
        Input: 
            X: training set
            Y: labels of training set
            splits: cross validation splits, used to make sure the parameters are stable
        Output:
            clf.best_params_: dictionary with the parameters, to use: param_svm['kernel']
        """    
        
        start_nn = time.time()  
    
        tuned_parameters = {
        'activation': (['relu','logistic']),
        'hidden_layer_sizes':([[12,16,12],[12,24,12],[12,12],[12,24],[12]]),
        #'hidden_layer_sizes':([[131,191,131],[131,231,131],[131,131,131]]),
        'alpha':     ([0.01, 0.001, 0.0001]),
        'batch_size':         [16,32],
        'learning_rate_init':    [0.01, 0.001],
        'solver': ["adam"]}
        
           
        print("NN Grid Search")
        mlp = MLPClassifier(max_iter=5000) 
        clf_grid = RandomizedSearchCV(mlp, tuned_parameters, cv= splits, scoring='%s' % scores[0],n_jobs=n_jobs)
            
        clf_grid.fit(x_train, y_train)
             
        end_nn = time.time()
        print("Total time to process NN: ",end_nn - start_nn)
        with open(path_results+"parameters_NN.txt", "a") as file:
            for item in clf_grid.best_params_:
              file.write(" %s %s " %(item,clf_grid.best_params_[item] ))
            file.write("\n")
            
        clf = MLPClassifier(**clf_grid.best_params_,random_state=random_state)
        
        clf.fit(x_train, y_train)
        
        preds = clf.predict(x_test)
        
        probas = clf.predict_proba(x_test)[:, 1]
        
        probas_train = clf.predict_proba(x_train)[:, 1]
        
        nn_m.clf_auc[itera]=roc_auc_score(y_test,probas)
        
        #Here we optimize the classifier for the threshold for f1_Score and specificity
        ts=np.linspace(0.1, 0.99, num=100)
        best_val=0
        best_t=0
        t_spec=0
        found=False
        for i in range(ts.shape[0]):
            p=probas_train>ts[i]
            c_f1=f1_score(y_train, p)
            tn, fp, fn, tp = confusion_matrix(y_train, p).ravel()
            c_spec=tn/(tn+fp)
            if c_f1>best_val:
                best_val=c_f1
                best_t=ts[i]
            if c_spec>=0.95 and not found:
                t_spec=ts[i]
                found=True 
    
    
        fpr_rf, tpr_rf, _ = roc_curve(y_test, probas)  
        
        nn_m.clf_brier[itera] = brier_score_loss(y_test, probas)   
                
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
          
        nn_m.clf_f1_score[itera]=f1_score(y_test, probas>=0.5)
        tn, fp, fn, tp = confusion_matrix(y_test, probas>=0.5).ravel()        
        nn_m.clf_sens[itera]=tp/(tp+fn)
        nn_m.clf_spec[itera]=tn/(tn+fp)
        
        nn_m.f1_score_f1[itera]=f1_score(y_test, probas>best_t)        
        tn, fp, fn, tp = confusion_matrix(y_test, probas>best_t).ravel()        
        nn_m.sens_f1[itera]=tp/(tp+fn)
        nn_m.spec_f1[itera]=tn/(tn+fp)
        
        nn_m.f1_score_spec[itera]=f1_score(y_test, probas>t_spec)
        tn, fp, fn, tp = confusion_matrix(y_test, probas>t_spec).ravel()        
        nn_m.sens_spec[itera]=tp/(tp+fn)
        nn_m.spec_spec[itera]=tn/(tn+fp)  
        
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\nnsens_"+str(itera)+".npy",nn_m.sens_spec[itera])
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\nnspec_"+str(itera)+".npy",nn_m.spec_spec[itera])
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\nnauc_"+str(itera)+".npy",nn_m.clf_auc[itera])
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\nnf1_"+str(itera)+".npy",nn_m.f1_score_spec[itera])
        
        #nn_m.clf_ppv[itera]=tp/(tp+fp)
        
        #nn_m.clf_npv[itera]=tn/(tn+fn)  
            
        return(fpr_rf,tpr_rf,probas,clf)
        
                

        
    def __init__(self,run,x_train,y_train,x_test,y_test,itera,nn_m,cv,mean_tprr,path_results):
        if run:
            self.name='NN'
            nn_m.run=True
            fpr_rf,tpr_rf,probas_t,self.clf=self.RandomGridSearchNN(x_train,y_train,x_test,y_test,cv,path_results,nn_m,itera)
            print("Done testing - NN", nn_m.clf_auc[itera])
            np.save(path_results+"nn_probas_"+str(itera)+".npy",probas_t)
            mean_fpr = np.linspace(0, 1, 100) 
            nn_m.mean_tpr += interp(mean_fpr, fpr_rf,tpr_rf)
            nn_m.mean_tpr[0] = 0.0
        else:
            self.name='NONE'
            self.clf=0
                

class LR_Pipeline: 
    
     def RandomGridSearchLR(self,x_train,y_train,x_test,y_test,splits,path_results,lr_m,itera,feats):
        """
        This function looks for the best set o parameters for RFC method
        Input: 
            X: training set
            Y: labels of training set
            splits: cross validation splits, used to make sure the parameters are stable
        Output:
            clf.best_params_: dictionary with the parameters, to use: param_svm['kernel']
        """    
        
        start_lr = time.time()  
    
        
        tuned_parameters = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
        
        #scores=['roc_auc']
           
        print("LR Grid Search")
        if class_weight:
            lr = LogisticRegression(random_state=random_state,class_weight='balanced',max_iter=10000) 
        else:
            lr = LogisticRegression(random_state=random_state,max_iter=10000) 
        #clf_grid = RandomizedSearchCV(lr, tuned_parameters, cv= splits,scoring='%s' % scores[0],verbose=5)
        clf_grid = RandomizedSearchCV(lr, tuned_parameters, cv= splits,scoring=score)
            
        clf_grid.fit(x_train, y_train)
             
        end_lr = time.time()
        print("Total time to process LR: ",end_lr - start_lr)
        with open(path_results+"parameters_LR.txt", "a") as file:
            for item in clf_grid.best_params_:
              file.write(" %s %s " %(item,clf_grid.best_params_[item] ))
            file.write("\n")
        if class_weight:   
            clf = LogisticRegression(**clf_grid.best_params_,random_state=random_state,class_weight='balanced') 
        else:
            clf = LogisticRegression(**clf_grid.best_params_,random_state=random_state) 
            
        
        clf.fit(x_train, y_train)
        
        preds = clf.predict(x_test)
        
        probas = clf.predict_proba(x_test)[:, 1]
        
        probas_train = clf.predict_proba(x_train)[:, 1]

        lr_m.clf_auc[itera]=roc_auc_score(y_test,probas)
        
        fpr_rf, tpr_rf, t = roc_curve(y_test, probas)  
        
        
        #Here we optimize the classifier for the threshold for f1_Score and specificity
        ts=np.linspace(0.1, 0.99, num=100)
        best_val=0
        best_t=0
        t_spec=0
        found=False
        for i in range(ts.shape[0]):
            p=probas_train>ts[i]
            c_f1=f1_score(y_train, p)
            tn, fp, fn, tp = confusion_matrix(y_train, p).ravel()
            c_spec=tn/(tn+fp)
            if c_f1>best_val:
                best_val=c_f1
                best_t=ts[i]
            if c_spec>=0.95 and not found:
                t_spec=ts[i]
                found=True 
      
        lr_m.clf_thresholds.append(t)
                    
        lr_m.clf_auc[itera]=roc_auc_score(y_test,probas)
        
        fpr_rf, tpr_rf, _ = roc_curve(y_test, probas)
        
        lr_m.clf_f1_score[itera]=f1_score(y_test, preds)
        
        lr_m.clf_brier[itera] = brier_score_loss(y_test, probas)   
                
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
            
        odds=np.exp(clf.coef_)
        feats=np.array(feats,dtype='float64')
        pos=0
        for i in range(0,feats.shape[0]):
            if feats[i]==1:
                feats[i]=odds[0,pos]                
                pos=pos+1

        lr_m.feat_imp.append(feats)   
        
        lr_m.clf_f1_score[itera]=f1_score(y_test, probas>=0.5)
        tn, fp, fn, tp = confusion_matrix(y_test, probas>=0.5).ravel()        
        lr_m.clf_sens[itera]=tp/(tp+fn)
        lr_m.clf_spec[itera]=tn/(tn+fp)
        
        lr_m.f1_score_f1[itera]=f1_score(y_test, probas>best_t)        
        tn, fp, fn, tp = confusion_matrix(y_test, probas>best_t).ravel()        
        lr_m.sens_f1[itera]=tp/(tp+fn)
        lr_m.spec_f1[itera]=tn/(tn+fp)
        
        lr_m.f1_score_spec[itera]=f1_score(y_test, probas>t_spec)
        tn, fp, fn, tp = confusion_matrix(y_test, probas>t_spec).ravel()        
        lr_m.sens_spec[itera]=tp/(tp+fn)
        lr_m.spec_spec[itera]=tn/(tn+fp)  
        
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\lrsens_"+str(itera)+".npy",lr_m.sens_spec[itera])
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\lrspec_"+str(itera)+".npy",lr_m.spec_spec[itera])
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\lrauc_"+str(itera)+".npy",lr_m.clf_auc[itera])
        np.save(r"E:\DeepEEG\final_Results5Stim00124_hours\lrf1_"+str(itera)+".npy",lr_m.f1_score_spec[itera])
        
        #lr_m.clf_ppv[itera]=tp/(tp+fp)
        
        #lr_m.clf_npv[itera]=tn/(tn+fn) 
            
        return(fpr_rf,tpr_rf,probas,clf)
        

        
     def __init__(self,run,x_train,y_train,x_test,y_test,itera,mean_tprr,lr_m,cv,path_results):
        feats=np.ones(x_train.shape[1])
        if run:
            lr_m.run=True
            self.name='LR'    
            fpr_lr,tpr_lr,probas_t,self.clf=self.RandomGridSearchLR(x_train,y_train,x_test,y_test,cv,path_results,lr_m,itera,feats)
            print("Done testing - LR", lr_m.clf_auc[itera])
            np.save(path_results+"lr_probas_"+str(itera)+".npy",probas_t)
            mean_fpr = np.linspace(0, 1, 100) 
            lr_m.mean_tpr += interp(mean_fpr, fpr_lr,tpr_lr)
            lr_m.mean_tpr[0] = 0.0
        else:
            self.name='NONE'
            self.clf=0        

def Mean_Confidence_Interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    m=np.round(m,decimals=2)
    m=np.round(m,decimals=2)
    m=np.round(m,decimals=2)
    return np.round(m,decimals=2), np.round(m-h,decimals=2), np.round(m+h,decimals=2)


def Print_Results(m,splits,names,path_results):    
    colors=['darkorange','blue','green','black','yellow']
    path_results_txt=path_results+"Results.txt"
    for i in range(0,len(names)):        
        with open(path_results_txt, "a") as file:
              file.write("Results %s \n" %(names[i])) 
              file.write("Average AUC %0.4f CI  %0.4f - %0.4f \n" %(Mean_Confidence_Interval(m[i].clf_auc)))              
              file.write("Average Sensitivity %0.4f CI  %0.4f - %0.4f \n" %(Mean_Confidence_Interval(m[i].clf_sens)))
              file.write("Average Specificity %0.4f CI  %0.4f - %0.4f \n" %(Mean_Confidence_Interval(m[i].clf_spec)))
              file.write("\n")
        np.save(file=path_results+'AUCs_'+names[i]+'.npy',arr=m[i].clf_auc)
        mean_tpr=m[i].mean_tpr
        mean_tpr /= splits
        mean_tpr[-1] = 1.0
        #frac_pos_rfc  /= skf.get_n_splits(X, Y)
        mean_fpr = np.linspace(0, 1, 100) 
        mean_auc_rfc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color=colors[i],lw=2, label=names[i]+' (area = %0.2f)' % mean_auc_rfc)
        plt.legend(loc="lower right")
        np.save(file=path_results+'tpr_'+names[i]+'.npy',arr=mean_tpr)
        np.save(file=path_results+'fpr_'+names[i]+'.npy',arr=mean_fpr)
            
    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
    #plt.show() 
    
 
    
    
#def Print_Results_Excel(m,splits,names,path_results):    
#    colors=['darkorange','blue','green','black','yellow']
#    book = xlwt.Workbook(encoding="utf-8")    
#    sheet1 = book.add_sheet("Sheet 1")
#    path_results_txt=path_results+"Results.xls"
#
#    sheet1.write(0, 0, "Methods")
#    sheet1.write(0, 1, "AUC 95% CI ")
#    sheet1.write(0, 2, "Sensitivity")
#    sheet1.write(0, 3, "Specificity")
#    #Spec and sensitivty are inverted because of the label
#    for i in range(0,len(names)):        
#        print(i,names[i])
#        sheet1.write(i+1,0,(names[i])) 
#        sheet1.write(i+1,1,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].clf_auc))))              
#        sheet1.write(i+1,2,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].clf_sens))))              
#        sheet1.write(i+1,3,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].clf_spec))))              
#
#        np.save(file=path_results+'AUCs_'+names[i]+'.npy',arr=m[i].clf_auc)
#        np.save(file=path_results+'Thresholds_'+names[i]+'.npy',arr=m[i].clf_thresholds)
#        mean_tpr=m[i].mean_tpr
#        mean_tpr /= splits
#        mean_tpr[-1] = 1.0
#        #frac_pos_rfc  /= skf.get_n_splits(X, Y)
#        mean_fpr = np.linspace(0, 1, 100) 
#        mean_auc_rfc = auc(mean_fpr, mean_tpr)
#        plt.plot(mean_fpr, mean_tpr, color=colors[i],lw=2, label=names[i]+' (area = %0.2f)' % mean_auc_rfc)
#        plt.legend(loc="lower right")
#        np.save(file=path_results+'tpr_'+names[i]+'.npy',arr=mean_tpr)
#        np.save(file=path_results+'fpr_'+names[i]+'.npy',arr=mean_fpr)
#        np.save(file=path_results+'probas_'+names[i]+'.npy',arr=m[i].probas)
#        if names[i]=='RFC':
#            np.save(file=path_results+'Feat_Importance'+names[i]+'.npy',arr=m[i].feat_imp)
#    book.save(path_results_txt)        
#    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
#    #plt.show() 
#
#    
    
def Print_Results_Excel(m,splits,names,path_results):    
    colors=['darkorange','blue','green','black','yellow']
    book = xlwt.Workbook(encoding="utf-8")    
    sheet1 = book.add_sheet("Sheet 1")
    path_results_txt=path_results+"Results.xls"

    sheet1.write(0, 0, "Methods")
    sheet1.write(0, 1, "AUC 95% CI ")
    sheet1.write(0, 2, "Brier ")
    sheet1.write(0, 3, "F1-Score")
    sheet1.write(0, 4, "Sensitivity")
    sheet1.write(0, 5, "Specificity")
    #sheet1.write(0, 6, "PPV")
    #sheet1.write(0, 7, "NPV")
    sheet1.write(0, 6, "F1-Score_f1")
    sheet1.write(0, 7, "Sensitivity_f1")
    sheet1.write(0, 8, "Specificity_f1")
    sheet1.write(0, 9, "F1-Score_spec")
    sheet1.write(0, 10, "Sensitivity_spec")
    sheet1.write(0, 11, "Specificity_spec")
    #Spec and sensitivty are inverted because of the label
    for i in range(0,len(names)):        
        print(i,names[i])
        sheet1.write(i+1,0,(names[i])) 
        sheet1.write(i+1,1,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].clf_auc))))              
        sheet1.write(i+1,2,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].clf_brier))))              
        sheet1.write(i+1,3,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].clf_f1_score))))              
        sheet1.write(i+1,4,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].clf_sens))))              
        sheet1.write(i+1,5,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].clf_spec))))              
        
        sheet1.write(i+1,6,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].f1_score_f1))))              
        sheet1.write(i+1,7,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].sens_f1))))              
        sheet1.write(i+1,8,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].spec_f1))))              
        sheet1.write(i+1,9,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].f1_score_spec))))              
        sheet1.write(i+1,10,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].sens_spec))))              
        sheet1.write(i+1,11,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].spec_spec))))              
        #sheet1.write(i+1,6,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].clf_ppv))))              
        #sheet1.write(i+1,7,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].clf_npv))))              
        
        

        np.save(file=path_results+'AUCs_'+names[i]+'.npy',arr=m[i].clf_auc)
        np.save(file=path_results+'Thresholds_'+names[i]+'.npy',arr=m[i].clf_thresholds)
        mean_tpr=m[i].mean_tpr
        mean_tpr /= splits
        mean_tpr[-1] = 1.0
        #frac_pos_rfc  /= skf.get_n_splits(X, Y)
        mean_fpr = np.linspace(0, 1, 100) 
        mean_auc_rfc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color=colors[i],lw=2, label=names[i]+' (area = %0.2f)' % mean_auc_rfc)
        plt.legend(loc="lower right")
        np.save(file=path_results+'tpr_'+names[i]+'.npy',arr=mean_tpr)
        np.save(file=path_results+'fpr_'+names[i]+'.npy',arr=mean_fpr)
        if names[i]=='RFC':
            np.save(file=path_results+'Feat_Importance'+names[i]+'.npy',arr=m[i].feat_imp)
    book.save(path_results_txt)        
    #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')    
    #plt.show() 
    