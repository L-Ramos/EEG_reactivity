# -*- coding: utf-8 -*-
"""
This code is for fixing outliers

@author: laramos
"""

X_complete = np.concatenate((X_train,X_test),axis=0)
X_complete = data
val_outliers = list()
list_outliers = list()
for i in range(0,X_complete.shape[0]):    
    max_v= np.max(X_complete[:,0,:])
    w = np.where(X_complete[:,0,:]==max_v)    
    #X_complete[w[0][0],0] = 0
    list_outliers.append(w[0][0])
    val_outliers.append(max_v)
    print(max_v,w[0][0])
    break
    
        
    
val=40
data_s=np.zeros((60,n_feats,channels))
l=0
#file = path_data+"\\"+x[20]+"_24h_epoch"+str(val)+".txt"    
#file = r'L:\\basic\\divd\\knf\\Onderzoek_projecten\\EEG_IC\\DeepEEG\\final_DeepEEGdata\\5Sec\\frequencyFeatures_fft\\Pat060_24h_epoch40.txt'
for p in range(1,61):
    file = r'L:\\basic\\divd\\knf\\Onderzoek_projecten\\EEG_IC\\DeepEEG\\final_DeepEEGdata\\5Sec\\frequencyFeatures_fft\\Pat090_24h_epoch'+str(p)+'.txt'
    file_input=open(file)
    lines_input=file_input.readlines()
    for j in range(0,n_feats):
        data_s[p-1,j,:]=np.fromstring(lines_input[j], dtype='float32', sep=";").reshape(channels) 
    
max_v = np.zeros(60)    
for p in range(0,60):
    max_v[p] = np.max(data_s[p,:,:])
    
    
    
    
    
    
missing = list(missing)    
missing.append('Pat090')
np.save('E:\missing_ecg.npy',missing)
    
    
    
    
    
n_feats=12
channels=14
files=list()
labels=list()
missing_list=list()
for i in range(0,x.shape[0]):    
    if x[i] not in missing:
        for var in range(1,60):
        #val=random.randint(1,60)
    #read the files of one of the patients
        #f=(glob.glob(path+"\\"+x[i]+"_"+stimulus[k]+"*"+pre_post+"Stim.txt"))        
        #f=(glob.glob(r"E:\DeepEEG\data\\"+secs+"sec\\frequencyFeatures_fft\\"+x[i]+"_baseline_epoch"+str(val)+".txt")) 
            #print(x[i],val)
            f=(glob.glob(path_data+"\\"+x[i]+"_baseline_epoch"+str(val)+".txt"))        
            #append the file names and the same label for each file available
            if len(f)>0:
                labels.append(y[i])
                files.append(f[0])
            else:
                print(x[i],f)
                missing_list.append(x[i])
#print(files)            
#create the data read to read all the values and convertthe albels
data=np.zeros((len(files),n_feats,channels))
labels=np.array(labels)


for l,file in enumerate(files):
    file_input=open(file)
    lines_input=file_input.readlines()
    for j in range(0,n_feats):
        data[l,j,:]=np.fromstring(lines_input[j], dtype='float32', sep=";").reshape(channels)     