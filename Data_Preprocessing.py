# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 12:20:34 2018

@author: laramos
"""

"""
This file contains several pre-processing fuctions for the EEG data project

@author: laramos
"""
import pandas as pd
import numpy as np
import glob
from scipy import signal
from scipy.signal import butter, lfilter
import pyeeg


#Created my own bin_power function to extract this feature
def Bin_Power(X,Band,Fs):    
    C = np.fft.fft(X)
    C = abs(C)
    Power =np.zeros(len(Band)-1);
    for Freq_Index in range(0,len(Band)-1):
    		Freq = float(Band[Freq_Index])										## Xin Liu
    		Next_Freq = float(Band[Freq_Index+1])
    		Power[Freq_Index] = sum(C[int(np.floor(Freq/Fs*len(X))):int(np.floor(Next_Freq/Fs*len(X)))])
    Power_Ratio = np.array(Power/sum(Power),'float32')
    return Power, Power_Ratio
    
def Spectral_Entropy(X,Band,fs,Power_Ratio):    
        Spectral_Entropy = 0
        for i in range(0, len(Power_Ratio) - 1):
            Spectral_Entropy = Spectral_Entropy+ Power_Ratio[i] * np.log(Power_Ratio[i])
        Spectral_Entropy /= np.log(len(Power_Ratio))	# to save time, minus one is omitted
        return -1 * Spectral_Entropy

def Denoise_Data(data):
    """
    This function denoise the data according to Marjolein matlab file
    input:
        data: (samples,signal,channels) array to be denoised
    Output:
        filtered_data: (samples,signal,14), denoised data
    """
    #channels to be subtract from each other
    locFp1 = 0 
    locFp2 = 1
    locT3 = 2
    locC3 = 3
    locCz = 4
    locC4 = 5
    locT4 = 6
    locO1 = 7
    locO2 = 8

    filtered_data=np.zeros((data.shape[0],data.shape[1],data.shape[2]+5))  
     
    #new subtracted channels
    filtered_data[:,:,0]=data[:,:,locFp1]-data[:,:,locT3]
    filtered_data[:,:,1]=data[:,:,locT3]-data[:,:,locO1]
    filtered_data[:,:,2]=data[:,:,locFp2]-data[:,:,locT4]
    filtered_data[:,:,3]=data[:,:,locT4]-data[:,:,locO2]
    filtered_data[:,:,4]=data[:,:,locFp1]-data[:,:,locC3]
    filtered_data[:,:,5]=data[:,:,locC3]-data[:,:,locO1]
    filtered_data[:,:,6]=data[:,:,locFp2]-data[:,:,locC4]
    filtered_data[:,:,7]=data[:,:,locC4]-data[:,:,locO2]
    filtered_data[:,:,8]=data[:,:,locFp1]-data[:,:,locFp2]
    filtered_data[:,:,9]=data[:,:,locT3]-data[:,:,locC3]
    filtered_data[:,:,10]=data[:,:,locC3]-data[:,:,locCz]
    filtered_data[:,:,11]=data[:,:,locCz]-data[:,:,locC4]
    filtered_data[:,:,12]=data[:,:,locC4]-data[:,:,locT4]
    filtered_data[:,:,13]=data[:,:,locO1]-data[:,:,locO2]

    #Filter
    freq=np.array([0.5,30])/(0.5*256)
    b, a = signal.butter(3,freq, btype='bandpass')
    #filtered_data = lfilter(b, a, filtered_data)
    for k in range(0,filtered_data.shape[0]):
        for i in range(0,filtered_data.shape[2]):
            filtered_data[k,:,i]=signal.filtfilt(b, a,filtered_data[k,:,2],padtype = 'odd')    
    return(filtered_data)

def Pre_Process_Data(path,path_outcome,channels,n_samples,denoise):
    """
    This fucntion reads the outcome .csv and connects with the outcomes with the data of each file
    It will skip files with missing labels
    Input:
        path: path of the files (EEG)
        path_outcome: path for the outcome file .csv
    Output:
        data: 3d array (number of samples, size of signal, channels)
        labels: 1d array with 0 for poor and 1 for good outcome
    """            
    
    outcome=pd.read_csv(path_outcome,sep=",")
    names_subj=(outcome['patName']) #names of the patients
    outcome=np.array(outcome['Outcome_binary']) #labels
    
    #reads all files
    files=(glob.glob(path))
    samples=len(files)
    
    
    #storing data and labels
    data=np.zeros((samples,n_samples,channels))
    labels=np.zeros(samples)
    
    #this is used to skip the patients without outcome, I can't use i
    cont_pos=0
    for i in range(0,samples):
        #eeg data file
        file_input=open(files[i])
        #first assume outcome is missing
        found=False    
        for k in range(0,len(names_subj)):            
            if (str(names_subj[k][3:6]) in files[i]):#from 3:6 because the patient name has AMC in the outcome file, I compare only the numbers
                l=outcome[k]#read outcome
                found=True #found the outcome
               #print(k,names_subj[k])
        if found==False:
            print(files[i]) #print missing outcome file
        else:    
            #else reads the file data and add the label to thelabel var
            labels[cont_pos]=l    
            lines_input=file_input.readlines()    
            for j in range(0,n_samples):
                data[cont_pos,j,:]=np.fromstring(lines_input[j], dtype='float32', sep=";").reshape(1,channels)   
            #return(data[cont_pos,:,:],labels)
            cont_pos=cont_pos+1
    #Denoises data according to Marjolein 
    if denoise:           
        data=Denoise_Data(data)
    #returns only the positions for which a label was found
    labels=labels[0:cont_pos]             
    data=data[0:cont_pos,:,:]
    return(data,labels)  
    
def Load_Features(path_features,path_outcome,n_feats):
    """
    This functions loads all the features created by Marjolein
    It will also skip the files without an outcome label
    Input: 
        path_features: path for the feature files
        path_outcome: path for the outcome file .csv
    Output: 
        no output, data and label file are written to local folder
    """
    
    #gets all files
    files=(glob.glob(path_features))
   
    samples=len(files)
    #8 features for 14 signals
    channels=14
    #pre and pos data files
    data_pre=np.zeros((int(samples/2),n_feats,channels))   
    data_pos=np.zeros((int(samples/2),n_feats,channels))       
    labels_pre=np.zeros(int(samples/2))
    labels_pos=np.zeros(int(samples/2))
    

    outcome=pd.read_csv(path_outcome,sep=",")
    names_subj=(outcome['patName'])
    outcome=np.array(outcome['Outcome_binary'])
    
    #positions for the data variables, half the files are from pre and half pos signal
    cont_pre=0
    cont_pos=0
    pats=list()
    for i in range(0,samples):
            file_input=open(files[i])
            lines_input=file_input.readlines()  
            #assume file is not there
            found=False    
            for k in range(0,len(names_subj)):            
                num_subj=(names_subj['names'].iloc[k])
                if (num_subj in files[i]):
                    l=outcome[k]#copy outcome
                    found=True#found patient label

            if found==False: 
                print("Missing file: ", files[i])#print file with missing label
            else:    
                    if 'pre' in files[i]:
                        for j in range(0,n_feats):
                            data_pre[cont_pre,j,:]=np.fromstring(lines_input[j], dtype='float32', sep=";").reshape(14)  
                        label_pre[cont_pre]=l
                        cont_pre=cont_pre+1
                        pats.append(files[i])  
                    else:
                        for j in range(0,n_feats):
                            data_pos[cont_pos,j,:]=np.fromstring(lines_input[j], dtype='float32', sep=";").reshape(14)    
                        label_pos[cont_pos]=l
                        cont_pos=cont_pos+1
    #computing average over the 14 channels
   
    data_pre_avg=np.mean(data_pre,axis=2)        
    data_pos_avg=np.mean(data_pos,axis=2)
    data_pre_avg=data_pre_avg[0:cont_pre,:]
    data_pos_avg=data_pos_avg[0:cont_pos,:]
    
    return(data_pre_avg,data_pos_avg,label,pats)

def Extract_Features(data,name_file):

    pre_data=data[:,0:1280,:]
    pos_data=data[:,1280:2560,:]
    
    samples=pre_data.shape[0]
    channels=pre_data.shape[2]
      
    pre_hurst=np.zeros((samples,channels))
    pos_hurst=np.zeros((samples,channels))    
    
    pre_dfa=np.zeros((samples,channels))
    pos_dfa=np.zeros((samples,channels))
        
    pre_svd=np.zeros((samples,channels))
    pos_svd=np.zeros((samples,channels))
    
    pre_apentropy=np.zeros((samples,channels))
    pos_apentropy=np.zeros((samples,channels))
    
    pre_fisher_info=np.zeros((samples,channels))
    pos_fisher_info=np.zeros((samples,channels))
    
    pre_sample_entropy=np.zeros((samples,channels))
    pos_sample_entropy=np.zeros((samples,channels))
    
    pre_h_mob=np.zeros((samples,channels))
    pos_h_mob=np.zeros((samples,channels))
    
    pre_h_com=np.zeros((samples,channels))
    pos_h_com=np.zeros((samples,channels))
        
    #std of each channel as feature
        
    
    print("Computing Features!")
    for i in range(0,samples):
        print("Processing sample: ",i)
        for j in range(0,channels):
            
            #Hurst Exponent
            pre_hurst[i,j]=pyeeg.hurst(pre_data[i,:,j])
            pos_hurst[i,j]=pyeeg.hurst(pos_data[i,:,j])
            
            #Detrended Fluctuation Analysis
            pre_dfa[i,j]=pyeeg.dfa(pre_data[i,:,j])
            pos_dfa[i,j]=pyeeg.dfa(pos_data[i,:,j])
            
            #Tau and DE (based on documentation)
            tau=5
            DE=4
            #SVD Entropy
            pre_svd[i,j]=pyeeg.svd_entropy(pre_data[i,:,j],tau,DE)
            pos_svd[i,j]=pyeeg.svd_entropy(pos_data[i,:,j],tau,DE)
            
            #Fisher Information
            pre_fisher_info[i,j]=pyeeg.fisher_info(pre_data[i,:,j],tau,DE)
            pos_fisher_info[i,j]=pyeeg.fisher_info(pos_data[i,:,j],tau,DE)
                    
            m=4 # embedding dimension
            r_pre=np.std(pre_data[i,:,j])*0.25 #25% of std of X
            r_pos=np.std(pos_data[i,:,j])*0.25
            
            #Sample Entropy                    
            pre_sample_entropy[i,j]=pyeeg.samp_entropy(pre_data[i,:,j],m,r_pre)
            pos_sample_entropy[i,j]=pyeeg.samp_entropy(pos_data[i,:,j],m,r_pos)
            
            #Hjorth mobility and complexity
            mob,com=pyeeg.hjorth(pre_data[i,:,j])
            pre_h_mob[i,j]=mob
            pre_h_com[i,j]=com
    
            mob,com=pyeeg.hjorth(pos_data[i,:,j])
            pos_h_mob[i,j]=mob
            pos_h_com[i,j]=com  
    
            #Band=[1,4,8,13,25]
            #Fs=256
            #pre_bin_power[:,i,j],pre_bin_power_ratio[:,i,j]=Bin_Power(pre_data[i,:,j],Band,Fs)
            #pos_bin_power[:,i,j],pos_bin_power_ratio[:,i,j]=Bin_Power(pos_data[i,:,j],Band,Fs)
            
            #pre_spectral[i,j]=Spectral_Entropy(pre_data[i,:,j],Band,Fs,pre_bin_power_ratio[:,i,j])
            #pos_spectral[i,j]=Spectral_Entropy(pos_data[i,:,j],Band,Fs,pos_bin_power_ratio[:,i,j])
            
    print("Done Extracting features! ")
    samples=pre_data.shape[0]
    #Average Features over channels
    pre_husrt_avg=np.mean(pre_hurst,axis=1).reshape(samples,-1)
    pos_husrt_avg=np.mean(pos_hurst,axis=1).reshape(samples,-1)
    
    pre_dfa_avg=np.mean(pre_dfa,axis=1).reshape(samples,-1)
    pos_dfa_avg=np.mean(pos_dfa,axis=1).reshape(samples,-1)
    
    pre_svd_avg=np.mean(pre_svd,axis=1).reshape(samples,-1)
    pos_svd_avg=np.mean(pos_svd,axis=1).reshape(samples,-1)
    
    pre_fisher_info_avg=np.mean(pre_fisher_info,axis=1).reshape(samples,-1)
    pos_fisher_info_avg=np.mean(pos_fisher_info,axis=1).reshape(samples,-1)
    
    pre_sample_entropy_avg=np.mean(pre_sample_entropy,axis=1).reshape(samples,-1)
    pos_sample_entropy_avg=np.mean(pos_sample_entropy,axis=1).reshape(samples,-1)
    
    pre_h_mob_avg=np.mean(pre_h_mob,axis=1).reshape(samples,-1)
    pre_h_com_avg=np.mean(pre_h_com,axis=1).reshape(samples,-1)
    
    pos_h_mob_avg=np.mean(pos_h_mob,axis=1).reshape(samples,-1)
    pos_h_com_avg=np.mean(pos_h_com,axis=1).reshape(samples,-1)
            
    pre_X=np.concatenate((pre_husrt_avg,pre_dfa_avg,pre_svd_avg,pre_fisher_info_avg,pre_sample_entropy_avg,pre_h_mob_avg,pre_h_com_avg),axis=1)
    pos_X=np.concatenate((pos_husrt_avg,pos_dfa_avg,pos_svd_avg,pos_fisher_info_avg,pos_sample_entropy_avg,pos_h_mob_avg,pos_h_com_avg),axis=1)
    
    np.save("pre_"+name_file+".npy",pre_X)
    np.save("pos_"+name_file+".npy",pos_X)
    print("Saved file pre features" )
    print("Saved file pos features")
    return()