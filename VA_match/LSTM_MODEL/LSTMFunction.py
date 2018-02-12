from keras.layers import Add,Input,Activation,Reshape,Conv1D,AveragePooling1D,Lambda,LSTM,Dense,TimeDistributed,SimpleRNN,GRU
from keras.layers import BatchNormalization,noise,Dropout,PReLU,Average
from keras.models import Model
from keras import initializers,regularizers
from keras import backend as K
import numpy as np

kernel=1
cut=0
pool=2
stride=2
index1=int(0+cut)
index2=int(120/pool-cut)
L=index2-index1
pad='valid'
noi=0.01
act='tanh'
def QQ(vect):
    v1,v2=vect
    return v1-v2
def slice_cut(x,start,end):
    return x[:,start:end,:]
        
def create_V2A_network(A_dim,V_dim):
    
    A_input=Input(shape=A_dim)
    AP=AveragePooling1D(pool_size=pool,strides=stride,padding='valid')(A_input)
    
    V_input=Input(shape=V_dim)
    VP=AveragePooling1D(pool_size=pool,strides=stride,padding='valid')(V_input)
    
    VL=LSTM(units=1024,return_sequences=True,stateful=False,dropout=0.2,recurrent_dropout=0.2,kernel_initializer=initializers.lecun_normal(),recurrent_initializer=initializers.lecun_uniform())(VP)
    VL=TimeDistributed(Dense(units=128,kernel_initializer=initializers.lecun_normal(),activation='tanh'))(VL)
    
    VT=TimeDistributed(Dense(units=128,kernel_initializer=initializers.lecun_normal(),activation='tanh'))(VP)
    VL=Average()([VL,VT])

    distance=Lambda(QQ,output_shape=[L,128])([VL,AP])
    
    res_model=Model(inputs=[A_input,V_input],outputs=distance)
    
    #my_model.summary()
    
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
        i=0
        while(i<a):
            index=np.where(xi==xisort[i])
            siz=np.shape(xi[index])
            ctt=i*np.ones(siz)
            xrank[index,j]=ctt
            i=i+siz[0]
    return xrank

