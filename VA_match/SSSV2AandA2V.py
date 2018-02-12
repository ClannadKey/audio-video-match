#minus
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
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
setproctitle.setproctitle('try@linziqian')  #from V1707
'''

from keras.models import Input
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping
from pdb import set_trace
import numpy as np
import pre_treat_lrt
from ResFunction import create_V2A_network,rowrank,columnrank,sum_rank

V,A=pre_treat_lrt.pre_treat_new('./filelists/train_filelist.txt','./filelists/train_filelist.txt')
Vstd=np.load('Vstd.npy')
Astd=np.load('Astd.npy')
Vm=np.load('Vm.npy')
Am=np.load('Am.npy')
for i in range(128):
    A[:,:,i]=(A[:,:,i]-Am[i])/Astd[i]
    
for i in range(1024):
    V[:,:,i]=(V[:,:,i]-Vm[i])/Vstd[i]

add=0.4

Vu1=128
dropV=0.25
drop1=0.125

L=60

lr1 = 0.002 #0.001 learning rate
lr2 = 0.0002
EP1=40
EP2=60

cal_number=60
test_number=120
train_number=1240-cal_number-test_number

V2A_train_Y=np.zeros([train_number,L,128])
V2A_test_Y=np.zeros([test_number,L,128])
A2V_train_Y=np.zeros([train_number,L,1024])
A2V_test_Y=np.zeros([test_number,L,1024])

all_A=A
all_V=V

np.random.seed(1999)
index_all=np.arange(1240)
np.random.shuffle(index_all)
all_A=all_A[index_all].copy()
all_V=all_V[index_all].copy()

cal_index,test_index=np.zeros(1240).astype('bool'),np.zeros(1240).astype('bool')

tp1=np.arange(60)+0
tp2=np.arange(120)+650
cal_index[tp1]=True
test_index[tp2]=True
train_index=~(cal_index|test_index)
print(tp1)

#split train test and cal
train_A1=all_A[train_index,:,:].copy()
train_V1=all_V[train_index,:,:].copy()
test_A1=all_A[test_index,:,:].copy()
test_V1=all_V[test_index,:,:].copy()
cal_A=all_A[cal_index,:,:].copy()
cal_V=all_V[cal_index,:,:].copy()
print('split finished')

A_dim=(120,128)
V_dim=(120,1024)
#Interaction-based Match
A_input=Input(shape=A_dim)
V_input=Input(shape=V_dim)


#V2A
V2A_network=create_V2A_network(A_dim,V_dim,Vu1,dropV,drop1)
dis=V2A_network([A_input,V_input])

V2A_network.summary();
print('V2A')
'''
model_checkpoint=ModelCheckpoint(
    filepath='./V2A.best.hdf5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    period=1
)
'''

V2A_network.compile(loss='mse',optimizer=Adam(lr1))
early_stopping =EarlyStopping(monitor='val_loss', patience=20) 
V2A_network.fit([train_A1,train_V1],V2A_train_Y,
                validation_data=([test_A1,test_V1],V2A_test_Y),
                callbacks=[early_stopping],
                batch_size=int(train_number/10),verbose=1,epochs=EP1)

#V2A_network.load_weights('V2A.best.hdf5')
V2A_network.compile(loss='mse',optimizer=Adam(lr2))
early_stopping =EarlyStopping(monitor='val_loss', patience=20) 
V2A_network.fit([train_A1,train_V1],V2A_train_Y,
                validation_data=([test_A1,test_V1],V2A_test_Y),
                callbacks=[early_stopping],
                batch_size=int(train_number/10),verbose=1,epochs=EP2)

#V2A_network.load_weights('V2A.best.hdf5')


V2A_network.save_weights('V2A.best.hdf5')

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

        pre=V2A_network.predict([a,v])
        answer[i,j]=np.mean(np.square(pre))
        
row=rowrank(answer)+1

endV=0
endV_var=0
for i in range(0,cal_number):
    numberV[i]=row[i,i]
    if row[i,i]<=5.5:
        endV=endV+1
        
print(numberV)
print('  origin:',endV)
print('  origin:',np.mean(numberV))
sum_rank(answer,cal_number,20)


