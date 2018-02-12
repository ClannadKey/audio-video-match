# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 14:54:32 2017

@author: Administrator
"""
from LSTMFunction import create_V2A_network,rowrank,columnrank
from pre_treat_lrt import pre_treat_new
#from load_data import loaddata 
import numpy as np
from tqdm import tqdm

def LSTM_lzy():
    model_DATAPATH='./LSTM_model_data'
    txt_DATAPATH='./filelists'
    A_dim=(120,128)
    V_dim=(120,1024)
    
    #loaddata('C:/Users/Administrator/Desktop/xuexiziliao5/stddzy/FeatOnly_Dataset/txt')
    cal_V,cal_A=pre_treat_new(txt_DATAPATH+'/Video_Name_List.txt',txt_DATAPATH+'/Audio_Name_List.txt')
    #cal_V=np.load('all_V_fortest.npy')
    #cal_A=np.load('all_A_fortest.npy')
    cal_number=cal_V.shape[0]
    
    var_A=np.load(model_DATAPATH+'/A0.npy')
    var_V=np.load(model_DATAPATH+'/V0.npy')
    pye=np.load(model_DATAPATH+'/pye.npy')
    cal_V=cal_V/var_V
    cal_A=cal_A/var_A-pye
    
    LSTM_lzy=create_V2A_network(A_dim,V_dim)
    LSTM_lzy.summary()
    
    LSTM_lzy.load_weights(model_DATAPATH+'/LSTM_lzy.best.hdf5')
    
    answer_var=np.zeros([cal_number,cal_number])
    numberV_var=np.zeros([cal_number])
    V_var=np.load(model_DATAPATH+'/V_var.npy')
    
    with tqdm(total=100) as pbar:
        for i in range(0,cal_number): 
            for j in range(0,cal_number):
                a=cal_A[i,:,:]
                a=a[np.newaxis,:,:]
                v=cal_V[j,:,:]
                v=v[np.newaxis,:,:]
        
                pre=LSTM_lzy.predict([a,v])
                answer_var[i,j]=np.dot(np.mean(np.square(pre),axis=1),V_var)
                pbar.update(100/(cal_number*cal_number))
            
    
    row_var=rowrank(answer_var)+1    
    #print(answer)
    
    endV_var=0;
    for i in range(0,cal_number):
        numberV_var[i]=row_var[i,i]
        if row_var[i,i]<=5.5:
            endV_var=endV_var+1
    '''
    print(numberV)
    print('var:'+str(endV_var)+'  origin:'+str(endV))
    print('var:'+str(np.mean(numberV_var))+'  origin:'+str(np.mean(numberV)))
    
    print('')
    '''
    for i in range(0,2):
        col_mes=columnrank(row_var)
        row_var=row_var+col_mes
        row_var=rowrank(row_var)+1
        endV_var=0
        for j in range(0,cal_number):
            numberV_var[j]=row_var[j,j]
            if row_var[j,j]<=5.5:
                endV_var=endV_var+1
        #print('var:'+str(endV_var)+' mean:'+str(np.mean(numberV_var)))
    
    np.save('LSTM_lzy.npy',row_var)
    return(row_var)
    
if __name__ == '__main__':
    row_var=LSTM_lzy()
    print(row_var)