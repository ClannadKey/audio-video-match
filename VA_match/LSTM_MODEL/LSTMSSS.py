#minus
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import tensorflow as tf  #from V1707
import setproctitle  #from V1707
from keras import backend as K
K.set_image_data_format('channels_first')

config=tf.ConfigProto()  #from V1707
#config.gpu_options.allow_growth=True  #from V1707
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess=tf.Session(config=config)  #from V1707
#import keras.backend.tensorflow_backend as KTF
#KTF._set_session(tf.Session(config=config))
setproctitle.setproctitle('std@luziyang')  #from V1707
'''
from keras import initializers
#from keras.utils import plot_model
from keras.models import Model,Input
from keras.layers import Concatenate,AveragePooling1D,noise,BatchNormalization,Conv1D,Activation,Dropout
from keras.layers import LSTM,Dense,TimeDistributed,Lambda
from keras.optimizers import Adam,Adamax
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import backend as K
#from pdb import set_trace
import numpy as np
from LSTMFunction import create_V2A_network,rowrank,columnrank

add=0.4

Vu1=128
Vu2=512
#Vu3=1024
dropV=0.7
drop1=0.4

Au1=1024
Au2=1024
#Au3=1408
dropA=0.7
drop2=0.4
L=60

np.random.seed(1337)#599,797,497,397* not good
lr1 = 0.002 #0.001 learning rate
lr2 = 0.0005
pre1=300
pre2=60
Vepoch=300
patience1=40
Aepoch=300
patience2=20

cal_number=60
test_number=60
train_number=1300-cal_number-test_number

V2A_train_Y=np.zeros([train_number,L,128])
V2A_test_Y=np.zeros([test_number,L,128])
A2V_train_Y=np.zeros([train_number,L,1024])
A2V_test_Y=np.zeros([test_number,L,1024])
#load data
#from keras.utils import np_utils

all_A=np.load('all_A.npy')
all_V=np.load('all_V.npy')
index_all=np.arange(1300)
np.random.shuffle(index_all)
A_var=np.var(all_A)
A_var=np.sqrt(A_var)
V_var=np.var(all_V)
V_var=np.sqrt(V_var)

all_A=all_A[index_all,:,:]/A_var
pye=(np.max(all_A)+np.min(all_A))/2
all_A=all_A-(np.max(all_A)+np.min(all_A))/2
all_V=all_V[index_all,:,:]/V_var
#all_V=all_V-(np.max(all_V)+np.min(all_V))/2
np.save('pye.npy',pye)
np.save('A0.npy',A_var)
np.save('V0.npy',V_var)
#all_A=all_A/255*2-1
#all_V=all_V/255*2-1
print('load and shuffle finish',A_var,V_var)
AA=all_A.reshape([-1,128])
VV=all_V.reshape([-1,1024])
AA_var=np.array([tp for tp in map(lambda a:np.var(a),AA.T)])
AA_var=AA_var.reshape([1,128])
AAA_var=np.repeat(AA_var,L,axis=0)
VV_var=np.array([tp for tp in map(lambda a:np.var(a),VV.T)])
VV_var=VV_var.reshape([1,1024])
VVV_var=np.repeat(VV_var,L,axis=0)

#split train test and cal
train_A1=all_A[0:train_number,:,:].copy()
train_V1=all_V[0:train_number,:,:].copy()
test_A1=all_A[train_number:train_number+test_number,:,:].copy()
test_V1=all_V[train_number:train_number+test_number,:,:].copy()
cal_A=all_A[1300-cal_number:1300,:,:].copy()
cal_V=all_V[1300-cal_number:1300,:,:].copy()
print('split finished')

A_dim=(120,128)
V_dim=(120,1024)
#Interaction-based Match
A_input=Input(shape=A_dim)
V_input=Input(shape=V_dim)



#V2A
LSTM_lzy=create_V2A_network(A_dim,V_dim)
dis=LSTM_lzy([A_input,V_input])

LSTM_lzy.summary();
#plot_model(LSTM_lzy,to_file='LSTM_lzy.png',show_shapes=True)
print('V2A')
'''
model_checkpoint=ModelCheckpoint(
    filepath='V2A.best.hdf5',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    period=1
)
'''

LSTM_lzy.compile(loss='mse',optimizer=Adam(lr1))
early_stopping =EarlyStopping(monitor='val_loss', patience=30) 
LSTM_lzy.fit([train_A1,train_V1],V2A_train_Y,
                validation_data=([test_A1,test_V1],V2A_test_Y),
                callbacks=[early_stopping],
                batch_size=int(train_number/10),verbose=1,epochs=pre1)

oldSSS=10000.
P=patience1
LSTM_lzy.compile(loss='mse',optimizer=Adam(lr2))
V_var=np.zeros([Vepoch*L,128],float)
#mean=1/(np.load('mean.npy')+0.3)
Vmean=np.load('SVmean.npy') 
Vmean=np.mean(Vmean,axis=0)
Vmean=Vmean.reshape([-1,1])
Vmean=1/(Vmean+add)
for EP in range(Vepoch):
    LSTM_lzy.fit([train_A1,train_V1],V2A_train_Y,
                    validation_data=([test_A1,test_V1],V2A_test_Y),
                    batch_size=int(train_number/10),verbose=1,epochs=1)
    SSS=0.
    SSS_origin=0.
    for i in range(0,test_number): 
        a=test_A1[i:i+1].repeat(test_number,axis=0)
        SSSV2A=LSTM_lzy.predict([a,test_V1])
        #set_trace()
        for_save=np.mean(SSSV2A[i,:,:]**2,axis=0)
        V_var[EP*60+i,:]=for_save

        SSSV2A_origin=np.mean(np.mean(SSSV2A**2,axis=1),axis=1)    
        SSSV2A=np.dot(np.mean(SSSV2A**2,axis=1),Vmean)
        
        SSS=SSS+float(np.sum(SSSV2A<=SSSV2A[i]))
        SSS_origin=SSS_origin+float(np.sum(SSSV2A_origin<=SSSV2A_origin[i]))
    if SSS_origin<oldSSS:
        oldSSS=SSS_origin
        P=patience1
        LSTM_lzy.save_weights('LSTM_lzy.best.hdf5')
    else:
        P=P-1      
    print('EP='+str(EP)+'  patience='+str(P)+'  rank='+str(SSS/test_number)+'  rank='+str(SSS_origin/test_number))
    if P==0:
        V_var=V_var[(EP-20)*L:EP*L,:]
        np.save('SVmean.npy',V_var) 
        print('Final Rank='+str(oldSSS/test_number))
        break

V_var=np.mean(V_var,axis=0)
V_var=V_var.reshape([-1,1])
V_var=1/(V_var+add)
np.save('V_var.npy',V_var)
LSTM_lzy.load_weights('LSTM_lzy.best.hdf5')
answer=np.zeros([cal_number,cal_number])
answer_var=np.zeros([cal_number,cal_number])
numberV=np.zeros([cal_number])
numberV_var=np.zeros([cal_number])

#set_trace()
for i in range(0,cal_number): 
    for j in range(0,cal_number):
        a=cal_A[i,:,:]
        a=a[np.newaxis,:,:]
        v=cal_V[j,:,:]
        v=v[np.newaxis,:,:]

        pre=LSTM_lzy.predict([a,v])
        answer[i,j]=np.mean(np.square(pre))
        answer_var[i,j]=np.dot(np.mean(np.square(pre),axis=1),V_var)
        
row=rowrank(answer)+1
row_var=rowrank(answer_var)+1    

endV=0;
endV_var=0;
for i in range(0,cal_number):
    numberV[i]=row[i,i]
    numberV_var[i]=row_var[i,i]
    if row[i,i]<=5.5:
        endV=endV+1
    if row_var[i,i]<=5.5:
        endV_var=endV_var+1
print(numberV)
print('var:'+str(endV_var)+'  origin:'+str(endV))
print('var:'+str(np.mean(numberV_var))+'  origin:'+str(np.mean(numberV)))

print('')
for i in range(0,3):
    col_mes=columnrank(row_var)
    row_var=row_var+col_mes
    row_var=rowrank(row_var)+1
    endV_var=0
    for j in range(0,cal_number):
        numberV_var[j]=row_var[j,j]
        if row_var[j,j]<=5.5:
            endV_var=endV_var+1
    print('var:'+str(endV_var)+' mean:'+str(np.mean(numberV_var)))

#s0

