# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:55:08 2019

@author: laramos
"""
import pandas as pd

which='pre' #'subtract' # #'subtract'
path=r"F:\DeepEEG"
folders="\\final_Results5Stim001"+which

path_results=path+folders+"\\Results.xls"
    
xls = pd.ExcelFile(path_results)    

df1 = xls.parse('Sheet 1')

df1.insert(0,'Stimulus', 1)

for i in range(2,6):
    path_results=path+"\\final_Results5Stim00"+str(i)+which+"\\Results.xls"   
    xls2 = pd.ExcelFile(path_results)    
    df2 = xls2.parse('Sheet 1')
    df2.insert(0,'Stimulus', i)
    
    df1=pd.concat((df1,df2),axis=0,ignore_index=True)

writer = pd.ExcelWriter(path+folders+".xlsx", engine='xlsxwriter')
#writer = pd.ExcelWriter("E:\\DeepEEG\\Results\\5sec_Baseline.xlsx", engine='xlsxwriter')
df1.to_excel(writer, 'Sheet1')
writer.save()    








import pandas as pd
path=r"E:\DeepEEG\New_Results"
folders="\\2_Results5Stim001pre"

path_results=path+folders+"\\Results.xls"
    
xls = pd.ExcelFile(path_results)    

df1 = xls.parse('Sheet 1')

df1.insert(0,'Stimulus', 1)

for i in range(2,6):
    path_results=path+"\\2_Results5Stim00"+str(i)+"pre\\"+"Results.xls"   
    xls2 = pd.ExcelFile(path_results)    
    df2 = xls2.parse('Sheet 1')
    df2.insert(0,'Stimulus', i)
    
    df1=pd.concat((df1,df2),axis=0,ignore_index=True)

writer = pd.ExcelWriter(path+folders+".xlsx", engine='xlsxwriter')
#writer = pd.ExcelWriter("E:\\DeepEEG\\Results\\5sec_Baseline.xlsx", engine='xlsxwriter')
df1.to_excel(writer, 'Sheet1')
writer.save()   