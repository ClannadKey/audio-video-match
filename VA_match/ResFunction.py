from keras.layers import Add,Input,Activation,Reshape,Conv1D,AveragePooling1D,Lambda,LSTM,Dense,TimeDistributed,SimpleRNN,GRU
from keras.layers import BatchNormalization,noise,Dropout,PReLU
from keras.models import Model
from keras import initializers
from keras import backend as K
import numpy as np

pool=2
stride=2
L=int(120/pool)
pad='valid'
noi=0.01

def QQ(vect):
    v1,v2=vect
    return v1-v2

def create_V2A_network(A_dim,V_dim,unit1,dropV,drop):
    
    A_input=Input(shape=A_dim)
    AP=AveragePooling1D(pool_size=pool,strides=stride,padding='valid')(A_input)
    
    V_input=Input(shape=V_dim)
    VP=AveragePooling1D(pool_size=pool,strides=stride,padding='valid')(V_input)
    
#VD=PReLU(shared_axes=1)(VP)
    VD=TimeDistributed(BatchNormalization())(VP)
    VD=PReLU(shared_axes=1)(VD)
    VD=noise.GaussianNoise(noi)(Dropout(dropV)(VD))
    VD=TimeDistributed(Dense(units=unit1,kernel_initializer=initializers.lecun_normal()))(VD)
    
#res_1=PReLU(shared_axes=1)(VP)
    res_1=TimeDistributed(BatchNormalization())(VP)
    res_1=PReLU(shared_axes=1)(res_1)
    res_1=noise.GaussianNoise(noi)(Dropout(drop)(res_1))
    res_1=TimeDistributed(Dense(units=unit1,kernel_initializer=initializers.lecun_normal()))(res_1)
     
#res_2=PReLU(shared_axes=1)(res_1)
    res_2=TimeDistributed(BatchNormalization())(res_1)
    res_2=PReLU(shared_axes=1)(res_2)
    res_2=noise.GaussianNoise(noi)(Dropout(drop)(res_2))
    res_2=TimeDistributed(Dense(units=unit1,kernel_initializer=initializers.lecun_normal()))(res_2)
    
    res_out=Add()([res_2,VD])
    
    distance=Lambda(QQ,output_shape=[L,128])([res_out,AP])
    
    res_model=Model(inputs=[A_input,V_input],outputs=distance)
    
    return res_model   
    
def rowrank(x):
    a,b=x.shape
    xrank=np.zeros([a,b])
    for i in range(a):
        xi=x[i,:].copy()
        xisort=np.sort(xi)
        for j in range(b):
            index=np.where(xi==xisort[j])
            xrank[i,index]=j
    return xrank
   
def columnrank(x):
    a,b=x.shape
    xrank=np.zeros([a,b])
    for j in range(b):
        xi=x[:,j].copy()
        xisort=np.sort(xi)
        for i in range(a):
            index=np.where(xi==xisort[i])
            xrank[index,j]=i
    return xrank
    
def counttop5(x,number):
    top=0
    mean=0
    for i in range(number):
        mean=mean+x[i,i]
        if x[i,i]<4.5:
            top=top+1
    mean=1.0*mean/number
    return top,mean

def sum_rank(x,number,k):
    print('iterate_sum_rank')
    row=rowrank(x)
    column=columnrank(row)
    #print(counttop5(row,number),counttop5(column,number))
    for i in range(k):
        Sum=row+column
        row=rowrank(Sum)
        column=columnrank(Sum)
        #print(counttop5(row,number),counttop5(column,number))       
    return row