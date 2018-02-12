import numpy as np
from functools import reduce
import os
def tpr_sum(tmp1,t_dis):
    all_arr=[]
    for i in range(120-t_dis+1):
        tmp2=reduce(lambda a,b:a+b,tmp1[i:i+t_dis])
        all_arr.append(list(tmp2))
    return all_arr

def pre_treat():
    print('loading data...')
    time_dis=3
    cnt=0
    for its in os.listdir(r'./Test'):
        if cnt>0:
            tmp=np.load(r'./Test/'+its+'/'+r'afeat.npy').astype('float32')
            tp=tpr_sum(tmp,time_dis)
            all_arr=np.array(tp)
            x=np.concatenate((x,all_arr),axis=0)
    
            tmp=np.load(r'./Test/'+its+'/'+r'vfeat.npy').astype('float32')
            tp=tpr_sum(tmp,time_dis)
            all_arr=np.array(tp)
            y=np.concatenate((y,all_arr),axis=0)
        else:            
            tmp=np.load(r'./Test/'+its+'/'+r'afeat.npy').astype('float32')
            tp=tpr_sum(tmp,time_dis)
            x=np.array(tp)
            
            tmp=np.load(r'./Test/'+its+'/'+r'vfeat.npy').astype('float32')
            tp=tpr_sum(tmp,time_dis)
            y=np.array(tp)
        cnt+=1
#        print(cnt)

    x_mean=np.array([tp for tp in map(lambda a:np.mean(x[:,a]),range(128))])
    x_std=np.array([tp for tp in map(lambda a:np.std(x[:,a]),range(128))])

    y_mean=np.array([tp for tp in map(lambda a:np.mean(y[:,a]),range(1024))])
    y_std=np.array([tp for tp in map(lambda a:np.std(y[:,a]),range(1024))])

    for i in range(len(x)):
        x[i]=(x[i]-x_mean)/x_std
        y[i]=(y[i]-y_mean)/y_std    
    print('loading data over...')
    return y,x

def pre_treat_path(txt1,txt2,path='./Train/',list_path='./filelists/'):#v,a
    print('loading data...')
    txt_list1,txt_list2=[],[]
    f1=open(txt1,'rt')
    f2=open(txt2,'rt')
    for line in f1:        txt_list1.append(line[:-1])
    for line in f2:        txt_list2.append(line[:-1])
 #   print(txt_list1)
    time_dis=3
    cnt=0

    for its in txt_list2:
        if cnt>0:
            tmp=np.load(path+its+'/'+r'afeat.npy').astype('float32')
            tp=tpr_sum(tmp,time_dis)
            all_arr=np.array(tp)
            x=np.concatenate((x,all_arr),axis=0)
        else:            
            tmp=np.load(path+its+'/'+r'afeat.npy').astype('float32')
            tp=tpr_sum(tmp,time_dis)
            x=np.array(tp)
        cnt+=1
        
    cnt=0        
    for its in txt_list1:
        if cnt>0:
            tmp=np.load(path+its+'/'+r'vfeat.npy').astype('float32')
            tp=tpr_sum(tmp,time_dis)
            all_arr=np.array(tp)
            y=np.concatenate((y,all_arr),axis=0)
        else:            
            tmp=np.load(path+its+'/'+r'vfeat.npy').astype('float32')
            tp=tpr_sum(tmp,time_dis)
            y=np.array(tp)
        cnt+=1


    x_mean=np.array([tp for tp in map(lambda a:np.mean(x[:,a]),range(128))])
    x_std=np.array([tp for tp in map(lambda a:np.std(x[:,a]),range(128))])

    y_mean=np.array([tp for tp in map(lambda a:np.mean(y[:,a]),range(1024))])
    y_std=np.array([tp for tp in map(lambda a:np.std(y[:,a]),range(1024))])

    for i in range(len(x)):
        x[i]=(x[i]-x_mean)/x_std
        y[i]=(y[i]-y_mean)/y_std    

    f1.close()
    f2.close()
    print('data load over...')
    return y,x    

def pre_treat_new(txt1,txt2,path='./Train/',list_path='./filelists/'):
    print('loading data...')
    txt_list1,txt_list2=[],[]
    f1=open(txt1,'rt')
    f2=open(txt2,'rt')
    for line in f1:        txt_list1.append(line[:11])
    for line in f2:        txt_list2.append(line[:11])
 #   print(txt_list1)
    x=[]
    y=[]
    for its in txt_list2:
        tmp=np.load(path+its+'/'+r'afeat.npy').astype('float32')
        x.append(tmp)
        
    for its in txt_list1:
        tmp=np.load(path+its+'/'+r'vfeat.npy').astype('float32')
        y.append(tmp)

    f1.close()
    f2.close()
    print('data load over...')
    return np.array(y),np.array(x)

if __name__ == '__main__':
    tp1,tp2=pre_treat_new('./filelists/train_filelist.txt','./filelists/train_filelist.txt')