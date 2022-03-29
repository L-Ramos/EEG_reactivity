

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV                               
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import brier_score_loss
import random as rand
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
import time
from scipy.stats import randint as sp_randint
import scipy as sp
from sklearn.preprocessing import label_binarize
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.calibration import calibration_curve
from scipy import interp 
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from sklearn.metrics import auc
#import xgboost as xgb
import xlwt
import joblib
from sklearn.metrics import make_scorer

jobs=-2
scores=['roc_auc']

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between 
        true and predicted values based on the metric chosen. """
    
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_predict).ravel()

    #sens=mat[0,0]/(mat[0,0]+mat[1,0])

    spec=tn/(tn+fp)
    
    # Return the score
    return(spec)


"""
class XGBoost_Pipeline: 
 
    def RandomGridSearchXGBoost(self,X,Y,splits,path_results):
      
        This function looks for the best set o parameters for RFC method
        Input: 
            X: training set
            Y: labels of training set
            splits: cross validation splits, used to make sure the parameters are stable
        Output:
            clf.best_params_: dictionary with the parameters, to use: param_svm['kernel']
          
        
        start_rfc = time.time()  
        
        tuned_parameters = {
        'learning_rate': ([0.3,0.1, 0.01, 0.001]),
        #'gamma': ([100,10,1, 0.1, 0.01, 0.001]),                  
        #'max_depth':    ([3,5,10,15]),
        #'subsample ':    ([0.5,1]),
        #'reg_lambda ':  [1,10,100],
        #'alpha ':   [1,10,100],
        
        'min_child_weight': [1, 5, 10],
        'gamma': [0, 0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7, 9, 11, 13]
        
        }
        
        scores = ['roc_auc']

            
        clf=xgb.XGBClassifier()
        print("XGB Grid Search")
        clf =  RandomizedSearchCV(clf, tuned_parameters, cv=splits,
                           scoring='%s' % scores[0],n_jobs=jobs,random_state=1)
        
                                  
        clf.fit(X, Y)
        #print("Score",clf.best_score_)
        end_rfc = time.time()
        print("Total time to process: ",end_rfc - start_rfc)
        
        with open(path_results+"parameters_xgb.txt", "a") as file:
            for item in clf.best_params_:
              file.write(" %s %s " %(item,clf.best_params_[item] ))
            file.write("\n")
        return(clf.best_params_)
        
        
        
    def TestXGBoost(self,x_train,y_train,x_test,y_test,params,itera,xgb_m):
       
        This function trains and tests the SVM method on the dataset and returns training and testing AUC and the ROC values
        Input:
            X_train: training set
            Y_train: training set labels
            X_test: testing set
            Y_test: testing set
            n_estim:  number of trees
            max_feat: number of features when looking for best split
            crit: criterion for quality of split measure
            itera: iteration of cross validation, used to write down the models 
        Output: 
            result: array with training [0] and testing error[1]
            fpr_svm: false positive rate, ROc values
            tpr_svm: true positive rate, ROc values
            clf: trained classifier, can be used to combine with others, like superlearner
            probas: probabilities from predict function
            brier: brier score
        
         
        
        #clf=xgb.XGBClassifier(gamma=params['gamma'],learning_rate=params['learning_rate'],max_depth=params['max_depth'])
        clf=xgb.XGBClassifier(**params)
        clf=clf.fit(x_train, y_train)
        
        #print(clf.feature_importances_)
        preds = clf.predict(x_test)
        probas = clf.predict_proba(x_test)[:, 1]
        preds=probas>0.5   
        xgb_m.clf_auc[itera]=roc_auc_score(y_test,probas)
    
        fpr_rf, tpr_rf, _ = roc_curve(y_test, probas)  
        xgb_m.clf_brier[itera] = brier_score_loss(y_test, probas)   
        conf_m=confusion_matrix(y_test, preds)
        xgb_m.clf_sens[itera]=conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])
        xgb_m.clf_spec[itera]=conf_m[1,1]/(conf_m[1,1]+conf_m[0,1])
        
        #name=('Models/RFC'+str(itera)+'.pkl')
        #joblib.dump(clf,name)
        
        return(fpr_rf,tpr_rf,probas)
    def __init__(self,run,x_train,y_train,x_test,y_test,itera,cv,mean_tprr,xgb_m,path_results):
        if run:
            self.name='XGB'
            xgb_m.run=True
            params=self.RandomGridSearchXGBoost(x_train,y_train,cv,path_results)
            print("Done Grid Search")
            fpr_rf,tpr_rf,probas_t=self.TestXGBoost(x_train,y_train,x_test,y_test,params,itera,xgb_m)
            print("Done testing - XGB", xgb_m.clf_auc[itera])
            mean_fpr = np.linspace(0, 1, 100) 
            xgb_m.mean_tpr += interp(mean_fpr, fpr_rf,tpr_rf)
            xgb_m.mean_tpr[0] = 0.0
            xgb_m.clf_fpr.append(fpr_rf)
            xgb_m.clf_tpr.append(tpr_rf)
        else:
            self.name='NONE'
"""



class RFC_Pipeline: 
 
    def RandomGridSearchRFC(self,X,Y,splits,path_results):
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
        'n_estimators': ([100,200,400,500,600,800,1000]),
        'max_features': (['auto', 'sqrt', 'log2',1,2,4,7]),                   # precomputed,'poly', 'sigmoid'
        'max_depth':    ([10,20,30]),
        'criterion':    (['gini', 'entropy']),
        'min_samples_split':  [2,3,4,5],
        'min_samples_leaf':   [2,3,4,5,6,7],
        'class_weight': ['balanced'],
        }
        
        #scores = ['roc_auc']
        scoring_fnc = make_scorer(performance_metric)

            
        rfc = RandomForestClassifier(n_estimators=25, oob_score = True,random_state=1)   
        
        print("RFC Grid Search")
        #clf =  RandomizedSearchCV(rfc, tuned_parameters, cv=splits,
        #                   scoring='%s' % scores[0],n_jobs=jobs,random_state=1)
        
        clf =  RandomizedSearchCV(rfc, tuned_parameters, cv=splits,scoring=scoring_fnc,n_jobs=jobs,random_state=1)
        
                                  
        clf.fit(X, Y)
        #print("Score",clf.best_score_)
        end_rfc = time.time()
        print("Total time to process: ",end_rfc - start_rfc)
        
        with open(path_results+"parameters_rfc.txt", "a") as file:
            for item in clf.best_params_:
              file.write(" %s %s " %(item,clf.best_params_[item] ))
            file.write("\n")
            
        kf = KFold(n_splits=5)
        thresh=np.zeros(5)
        clf_t = RandomForestClassifier(**clf.best_params_,random_state=1) 
        for l,(train_index, test_index) in enumerate(kf.split(X)):
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index] 
            clf_t.fit(X_train,y_train)
            probas = clf_t.predict_proba(X_test)[:, 1]
            spec=0
            total_vals=1000
            tresh=np.linspace(0,0.99,num=total_vals)
            cont=0        
            #doing same for probas_base
            while spec<=0.90 and cont<total_vals:
            
                t=tresh[cont]
            
                p=probas[:]>t
            
                #mat=confusion_matrix(y_test,p)
                        
                tn, fp, fn, tp = confusion_matrix(y_test,p).ravel()

                sens=tp/(tp+fn)
                spec=tn/(tn+fp)
                
                cont+=1
            
            #print(spec,sens,t)
                
            thresh[l]=t
        
        return(clf.best_params_,np.mean(thresh))
        
        
        
    def TestRFC(self,X_train,Y_train,X_test,Y_test,n_estim,max_feat,crit,itera,rfc_m,k,t):
        """
        This function trains and tests the SVM method on the dataset and returns training and testing AUC and the ROC values
        Input:
            X_train: training set
            Y_train: training set labels
            X_test: testing set
            Y_test: testing set
            n_estim:  number of trees
            max_feat: number of features when looking for best split
            crit: criterion for quality of split measure
            itera: iteration of cross validation, used to write down the models 
            k: 
        Output: 
            result: array with training [0] and testing error[1]
            fpr_svm: false positive rate, ROc values
            tpr_svm: true positive rate, ROc values
            clf: trained classifier, can be used to combine with others, like superlearner
            probas: probabilities from predict function
            brier: brier score
        """    
         
        clf = RandomForestClassifier(max_features=max_feat,n_estimators=n_estim, oob_score = True,criterion=crit,random_state=1)
               
        clf.fit(X_train,Y_train)
        #print(clf.feature_importances_)
        preds = clf.predict(X_test)
        probas = clf.predict_proba(X_test)[:, 1]
        #preds=probas>0.5   
        preds_bin=probas>t
        
        rfc_m.clf_thresholds[itera]=t
        rfc_m.clf_auc[itera]=roc_auc_score(Y_test,probas)
    
        fpr_rf, tpr_rf, _ = roc_curve(Y_test, probas)  
        rfc_m.clf_brier[itera] = brier_score_loss(Y_test, probas)   
        
        conf_m=confusion_matrix(Y_test, preds_bin)
        tn, fp, fn, tp = confusion_matrix(Y_test, preds_bin).ravel()

        
        rfc_m.clf_sens[itera]=tp/(tp+fn)
        rfc_m.clf_spec[itera]=tn/(tn+fp)
        rfc_m.feat_imp[itera,:]=clf.feature_importances_
        print(rfc_m.clf_spec[itera],rfc_m.clf_sens[itera],rfc_m.clf_auc[itera],t)
        #name=('E:\\DeepEEG\\Models\\RFC'+str(itera)+"_Stim"+str(k)+'.pkl')
        #joblib.dump(clf,name)
        
        return(fpr_rf,tpr_rf,probas)
    """    
    def Find_Threshold(self,prob,y,itera,rfc_m):
       
        
        spec=0
        total_vals=1000
        tresh=np.linspace(0.99,0,num=total_vals)
        cont=0        
        #doing same for probas_base
        while spec<=0.95 and cont<total_vals:
            
            t=tresh[cont]
            
            p=prob[:]>t
            
            mat=confusion_matrix(y,p)
        
            sens=mat[0,0]/(mat[0,0]+mat[1,0])
        
            spec=mat[1,1]/(mat[1,1]+mat[0,1])
            cont+=1
            
            print(spec,sens,t)
        conf_m=confusion_matrix(y, p)
        rfc_m.clf_sens_opt[itera]=conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])
        rfc_m.clf_spec_opt[itera]=conf_m[1,1]/(conf_m[1,1]+conf_m[0,1])
        return(0)
    """    
    def __init__(self,run,x_train,y_train,x_test,y_test,itera,cv,mean_tprr,rfc_m,path_results,k):
        if run:
            self.name='RFC'
            rfc_m.run=True
            Paramsrfc,t=self.RandomGridSearchRFC(x_train,y_train,cv,path_results)
            print("Done Grid Search")
            fpr_rf,tpr_rf,probas_t=self.TestRFC(x_train,y_train,x_test,y_test,Paramsrfc['n_estimators'],Paramsrfc['max_features'],Paramsrfc['criterion'],itera,rfc_m,k,t)
            #self.Find_Threshold(probas_t,y_test,itera,rfc_m)
            print("Done testing - RFC", rfc_m.clf_auc[itera])
            mean_fpr = np.linspace(0, 1, 100) 
            rfc_m.mean_tpr += interp(mean_fpr, fpr_rf,tpr_rf)
            rfc_m.mean_tpr[0] = 0.0
            rfc_m.clf_fpr.append(fpr_rf)
            rfc_m.clf_tpr.append(tpr_rf)

            np.save((path_results+self.name+"_prob_stim_"+ str(k)+"_"+str(itera))+".npy",np.concatenate((probas_t.reshape(-1,1),y_test.reshape(-1,1)),axis=1))
        else:
            self.name='NONE'

class SVM_Pipeline: 

    def RandomGridSearchSVM(self,X,Y,splits,path_results):
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
        
        #scores = ['roc_auc']   
  
        print("SVM Grid Search")
        clf =  RandomizedSearchCV(SVC(), tuned_parameters, cv=splits,
                       scoring='%s' % scores[0],n_jobs=-1,random_state=1)    
        
        #clf =  GridSearchCV(SVC(random_state=1), tuned_parameters, cv=splits,
        #               scoring='%s' % scores[0],n_jobs=jobs)    
        clf.fit(X, Y)
    
        end_svm = time.time()
        print("Total time to process: ",end_svm - start_svm)
        #print("Score",clf.best_score_)
        with open(path_results+"parameters_svm.txt", "a") as file:
            for item in clf.best_params_:
              file.write(" %s %s " %(item,clf.best_params_[item] ))
            file.write("\n")
            
            
        kf = KFold(n_splits=5)
        thresh=np.zeros(5)
        clf_t = SVC(**clf.best_params_,random_state=1) 
        for l,(train_index, test_index) in enumerate(kf.split(X)):
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index] 
            clf_t.fit(X_train,y_train)
            decisions = clf.decision_function(X_test)
            probas=\
            (decisions-decisions.min())/(decisions.max()-decisions.min())
            spec=0
            total_vals=1000
            tresh=np.linspace(0.99,0,num=total_vals)
            cont=0        
            #doing same for probas_base
            while spec<=0.90 and cont<total_vals:
            
                t=tresh[cont]
            
                p=probas[:]>t
            
                tn, fp, fn, tp = confusion_matrix(y_test,p).ravel()

                sens=tp/(tp+fn)
                
                spec=tn/(tn+fp)
                
                cont+=1
            
            print(spec,sens,t)
            thresh[l]=t
            
        return(clf.best_params_,np.mean(thresh))
        
        
        
    def TestSVM(self,X_train,Y_train,X_test,Y_test,kernel,C,gamma,deg,itera,svm_m,t):
        """
        This function trains and tests the SVM method on the dataset and returns training and testing AUC and the ROC values
        Input:
            X_train: training set
            Y_train: training set labels
            X_test: testing set
            Y_test: testing set
            n_estim:  number of trees
            max_feat: number of features when looking for best split
            crit: criterion for quality of split measure
            itera: iteration of cross validation, used to write down the models 
        Output: 
            result: array with training [0] and testing error[1]
            fpr_svm: false positive rate, ROc values
            tpr_svm: true positive rate, ROc values
            clf: trained classifier, can be used to combine with others, like superlearner
            probas: probabilities from predict function
            brier: brier score
        """    

         
        clf = svm.SVC(C=C,kernel=kernel,gamma=gamma,degree=deg,probability=True,random_state=1)
               
        clf.fit(X_train,Y_train)
        #preds = clf.predict(X_test)
        decisions = clf.decision_function(X_test)
        probas=\
        (decisions-decisions.min())/(decisions.max()-decisions.min())
#        preds=probas>0.5
        preds_bin=probas>t
        svm_m.clf_thresholds[itera]=t
        
        #probas=clf.predict_proba(X_test)[:, 1]
        svm_m.clf_auc[itera]=roc_auc_score(Y_test,probas)
    
        fpr_rf, tpr_rf, _ = roc_curve(Y_test, probas)  
        
        svm_m.clf_brier[itera] = brier_score_loss(Y_test, probas)  
        
        tn, fp, fn, tp = confusion_matrix(Y_test, preds_bin).ravel()
        svm_m.clf_sens[itera]=tp/(tp+fn)
        svm_m.clf_spec[itera]=tn/(tn+fp)
        
        #conf_m=confusion_matrix(Y_test, preds_bin)
        #svm_m.clf_sens[itera]=conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])
        #svm_m.clf_spec[itera]=conf_m[1,1]/(conf_m[1,1]+conf_m[0,1])
        print(svm_m.clf_spec[itera],svm_m.clf_sens[itera],svm_m.clf_auc[itera],t)
        #name=('Models/RFC'+str(itera)+'.pkl')
        #joblib.dump(clf,name)
        
    
        return(fpr_rf,tpr_rf,probas)

        
    def __init__(self,run,x_train,y_train,x_test,y_test,itera,svm_m,cv,mean_tprr,path_results,k):
        if run:
            svm_m.run=True
            self.name='SVM'
            Paramssvm,t=self.RandomGridSearchSVM(x_train,y_train,cv,path_results)
            fpr_rf,tpr_rf,probas_t=self.TestSVM(x_train,y_train,x_test,y_test,Paramssvm.get('kernel'),Paramssvm.get('C'),Paramssvm.get('gamma'),Paramssvm.get('degree'),itera,svm_m,t)
            print("Done testing - SVM", svm_m.clf_auc[itera])
            mean_fpr = np.linspace(0, 1, 100) 
            svm_m.mean_tpr += interp(mean_fpr, fpr_rf,tpr_rf)
            svm_m.mean_tpr[0] = 0.0
            svm_m.clf_fpr.append(fpr_rf)
            svm_m.clf_tpr.append(tpr_rf)
            np.save((path_results+self.name+"_prob_stim_"+ str(k)+"_"+str(itera))+".npy",np.concatenate((probas_t.reshape(-1,1),y_test.reshape(-1,1)),axis=1))
        else:
            self.name='NONE'

class LR_Pipeline: 
    
    def TestLogistic(self,X_train,Y_train,X_test,Y_test,itera,lr_m):
        
        
        kf = KFold(n_splits=5)
        thresh=np.zeros(5)
        clf_t = clf = LogisticRegression(C=1,solver="liblinear") 
        for l,(train_index, test_index) in enumerate(kf.split(X_train)):
            
            x_train, x_test = X_train[train_index], X_train[test_index]
            y_train, y_test = Y_train[train_index], Y_train[test_index] 
            clf_t.fit(x_train,y_train)
            decisions = clf.decision_function(x_test)
            probas=\
            (decisions-decisions.min())/(decisions.max()-decisions.min())
            spec=0
            total_vals=1000
            tresh=np.linspace(0,0.99,num=total_vals)
            cont=0        
            #doing same for probas_base
            while spec<=0.90 and cont<total_vals:
            
                t=tresh[cont]
            
                p=probas[:]>t
                
                tn, fp, fn, tp = confusion_matrix(y_test,p).ravel()
        
                sens=tp/(tp+fn)
                
                spec=tn/(tn+fp)
            
                cont+=1
            
             #print(spec,sens,t)
                
            thresh[l]=t
        t=np.mean(thresh)
    
        clf = LogisticRegression(C=1,solver="liblinear")
    
         
        clf.fit(X_train,Y_train)
    
        preds = clf.predict(X_test)
        probas = clf.predict_proba(X_test)[:, 1]
        preds=probas>t
        
        #plt.hist(probas,bins=10)
        #plt.show()
      
        lr_m.clf_auc[itera]=roc_auc_score(Y_test,probas)
    
        fpr_rf, tpr_rf, _ = roc_curve(Y_test, probas)  
        lr_m.clf_brier[itera] = brier_score_loss(Y_test, probas)   
        tn, fp, fn, tp = confusion_matrix(Y_test, preds).ravel()
        lr_m.clf_sens[itera]=tp/(tp+fn)
        lr_m.clf_spec[itera]=tn/(tn+fp)
        print(lr_m.clf_spec[itera],lr_m.clf_sens[itera],lr_m.clf_auc[itera],t)
        #name=('Models/RFC'+str(itera)+'.pkl')
        #joblib.dump(clf,name)
        
        return(fpr_rf,tpr_rf,probas)
        
    def __init__(self,run,x_train,y_train,x_test,y_test,itera,mean_tprr,lr_m,path_results,k):
        if run:
            lr_m.run=True
            self.name='LR'
            fpr_lr,tpr_lr,probas_t=self.TestLogistic(x_train,y_train,x_test,y_test,itera,lr_m)
            print("Done testing - LR", lr_m.clf_auc[itera])
            mean_fpr = np.linspace(0, 1, 100) 
            lr_m.mean_tpr += interp(mean_fpr, fpr_lr,tpr_lr)
            lr_m.mean_tpr[0] = 0.0
            lr_m.clf_fpr.append(fpr_lr)
            lr_m.clf_tpr.append(tpr_lr)
            np.save((path_results+self.name+"_prob_stim_"+ str(k)+"_"+str(itera))+".npy",np.concatenate((probas_t.reshape(-1,1),y_test.reshape(-1,1)),axis=1))
        else:
            self.name='NONE'

class NN_Pipeline: 

    def RandomGridSearchNN(self,X,Y,splits,path_results,layers_nn):
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
    
          
        #scores = ['roc_auc']
        tuned_parameters = {
        'activation': (['relu','logistic','tanh']),
        'hidden_layer_sizes':(layers_nn),
        'alpha':     ([1,0.1, 0.01, 0.001]),
        'batch_size':         [32,64],
        'learning_rate_init':    [0.01, 0.001,0.0001],
        'solver': ["adam"]}
        
        
        print("NN Grid Search")
        mlp = MLPClassifier(max_iter=5000,random_state=1) 
        clf = RandomizedSearchCV(mlp, tuned_parameters, cv= splits, scoring='%s' % scores[0],n_jobs=jobs,random_state=1)
            
        clf.fit(X, Y)
       
        #clf.fit(x_train, y_train)
        #probas=clf.predict_proba(x_test)[:,1]
        #auc=roc_auc_score(y_test,probas)
        
        end_nn = time.time()
        print("Total time to process NN: ",end_nn - start_nn)
        with open(path_results+"parameters_NN.txt", "a") as file:
            for item in clf.best_params_:
              file.write(" %s %s " %(item,clf.best_params_[item] ))
            file.write("\n")
        #print(clf.best_params_)
        
        kf = KFold(n_splits=5)
        thresh=np.zeros(5)
        clf_t = MLPClassifier(max_iter=5000,random_state=1,**clf.best_params_) 
        for l,(train_index, test_index) in enumerate(kf.split(X)):
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index] 
            clf_t.fit(X_train,y_train)
            probas = clf_t.predict_proba(X_test)[:, 1]
            spec=0
            total_vals=1000
            tresh=np.linspace(0,0.99,num=total_vals)
            cont=0        
            #doing same for probas_base
            while spec<=0.90 and cont<total_vals:
            
                t=tresh[cont]
            
                p=probas[:]>t
            
                tn, fp, fn, tp = confusion_matrix(y_test,p).ravel()
        
                sens=tp/(tp+fn)
                
                spec=tn/(tn+fp)
                
                cont+=1
            
            #print(spec,sens,t)
                
            thresh[l]=t
        
        return(clf.best_params_,np.mean(thresh))
        
        
        
        
                
    def TestNN(self,X_train,Y_train,X_test,Y_test,act,hid,alpha,batch,learn,solver,itera,nn_m,t):
        """
        This function trains and tests the SVM method on the dataset and returns training and testing AUC and the ROC values
        Input:
            X_train: training set
            Y_train: training set labels
            X_test: testing set
            Y_test: testing set
            act:  activation function for the hidden layers
            hid: size of hidden layers, array like [10,10]
            alpha: regularization parameters
            batch: minibatch size
            learn: learning rate
            solver: Adam or SGD
            itera: iteration of cross validation, used to write down the models 
        Output: 
            result: array with training [0] and testing error[1]
            fpr_svm: false positive rate, ROc values
            tpr_svm: true positive rate, ROc values
            clf: trained classifier, can be used to combine with others, like superlearner
            probas: probabilities from predict function
            brier: brier score
        """

        print("NN")
          
        clf_nn=MLPClassifier(solver=solver,activation=act,hidden_layer_sizes=hid,alpha=alpha,
                             batch_size=batch,learning_rate_init=learn,max_iter=5000,random_state=1)
                    
        clf_nn = clf_nn.fit(X_train, Y_train)
        #preds = clf_nn.predict(X_test)
        probas = clf_nn.predict_proba(X_test)[:, 1]
        preds=probas>t
        nn_m.clf_auc[itera]=roc_auc_score(Y_test,probas)
    
        fpr_rf, tpr_rf, _ = roc_curve(Y_test, probas)  
        nn_m.clf_brier[itera] = brier_score_loss(Y_test, probas)  
        tn, fp, fn, tp = confusion_matrix(Y_test, preds).ravel()
        
        nn_m.clf_sens[itera]=tp/(tp+fn)
                
        nn_m.clf_spec[itera]=tn/(tn+fp)
        #conf_m=confusion_matrix(Y_test, preds)
        #nn_m.clf_sens[itera]=conf_m[0,0]/(conf_m[0,0]+conf_m[1,0])
        #nn_m.clf_spec[itera]=conf_m[1,1]/(conf_m[1,1]+conf_m[0,1])
        print(nn_m.clf_spec[itera],nn_m.clf_sens[itera],nn_m.clf_auc[itera],t)
        
        #name=('Models/RFC'+str(itera)+'.pkl')
        #joblib.dump(clf,name)
        
        
        return(fpr_rf,tpr_rf,probas,clf_nn)

        
    def __init__(self,run,x_train,y_train,x_test,y_test,itera,nn_m,cv,mean_tprr,layers,path_results,k):
        if run:
            self.name='NN'
            nn_m.run=True
            Paramsnn,t=self.RandomGridSearchNN(x_train,y_train,cv,path_results,layers)
            fpr_rf,tpr_rf,probas_t,self.clf=self.TestNN(x_train,y_train,x_test,y_test,Paramsnn.get('activation'),Paramsnn.get('hidden_layer_sizes'),Paramsnn.get('alpha'),Paramsnn.get('batch_size'),
                                               Paramsnn.get('learning_rate_init'),Paramsnn.get('solver'),itera,nn_m,t)
            print("Done testing - NN", nn_m.clf_auc[itera])
            print("Size:",probas_t.shape)
            mean_fpr = np.linspace(0, 1, 100) 
            nn_m.mean_tpr += interp(mean_fpr, fpr_rf,tpr_rf)
            nn_m.mean_tpr[0] = 0.0
            np.save((path_results+self.name+"_prob_stim_"+ str(k)+"_"+str(itera)+".npy"),np.concatenate((probas_t.reshape(-1,1),y_test.reshape(-1,1)),axis=1))
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
    
 
    
    
def Print_Results_Excel(m,splits,names,path_results):    
    colors=['darkorange','blue','green','black','yellow']
    book = xlwt.Workbook(encoding="utf-8")    
    sheet1 = book.add_sheet("Sheet 1")
    path_results_txt=path_results+"Results.xls"

    sheet1.write(0, 0, "Methods")
    sheet1.write(0, 1, "AUC 95% CI ")
    sheet1.write(0, 2, "Sensitivity")
    sheet1.write(0, 3, "Specificity")
    #Spec and sensitivty are inverted because of the label
    for i in range(0,len(names)):        
        print(i,names[i])
        sheet1.write(i+1,0,(names[i])) 
        sheet1.write(i+1,1,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].clf_auc))))              
        sheet1.write(i+1,2,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].clf_sens))))              
        sheet1.write(i+1,3,str("%0.2f CI  %0.2f - %0.2f"%(Mean_Confidence_Interval(m[i].clf_spec))))              

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

    