# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 15:56:46 2017

@author: liruntong
"""
from keras.optimizers import Adamax
from keras.callbacks import EarlyStopping
from random import sample,seed
from keras import backend as K
from dis_conv import res_model1,model_3
import numpy as np
import dis_cal_lrt
import pre_treat_lrt



x_train,y_train=pre_treat_lrt.pre_treat_path('./filelists/train_filelist.txt','./filelists/train_filelist.txt')
x_train,y_train=x_train.astype('float32'),y_train.astype('float32')

vad_set_num=120
sec=3
time_len=121-sec
all_cnt=(time_len)*int(x_train.shape[0]/118)
#seed(1337)
tp=np.array(sample(range(int(x_train.shape[0]/118)),vad_set_num))
ind1=np.zeros(all_cnt).astype('bool')

for tmp_num in range(time_len):
    ind1[tmp_num+time_len*tp]=True

x_val=x_train[ind1]
y_val=y_train[ind1]
x_train,y_train=x_train[~(ind1)],y_train[~(ind1)] 

#---------------------------------
model_1=res_model1(1024,128)
model_1.compile(loss='mse',optimizer=Adamax())
early_stopping =EarlyStopping(monitor='val_loss', patience=5) 
history_callback = model_1.fit(x_train,y_train,batch_size=1024,callbacks=[early_stopping],epochs=60,validation_data=(x_val,y_val))
#loss_history = history_callback.history["val_loss"]
y_t_pre=model_1.predict(x_train)
model3=model_3(128)
model3.compile(loss='mse',optimizer=Adamax())
early_stopping =EarlyStopping(monitor='loss', patience=2)

model_1.save('./def_model/lrt_model_test.h5')


con_pro1=[]
con_pro2=[]
neg_num=30*10**4
for i in range(neg_num):
    tp=sample(range(int(len(y_train)/time_len)),2)
    tp1,tp2=sample(range(time_len),2)
    con_pro1.append(y_t_pre[tp[0]*time_len+tp1])
    con_pro2.append(y_train[tp[1]*time_len+tp2])
con_pro1=np.array(con_pro1)
con_pro2=np.array(con_pro2)
con_pro1=np.concatenate((con_pro1,y_t_pre))
con_pro2=np.concatenate((con_pro2,y_train))
tgt=np.append(20*np.ones((neg_num,1)),np.zeros((len(y_train),1)))

history_callback = model3.fit([con_pro1,con_pro2],tgt,batch_size=1024,callbacks=[early_stopping],epochs=60)

m3_wgt=model3.layers[-2].get_weights()[0]
m3_wgt=m3_wgt.reshape(128)

np.save('./def_model/lrt_model_para.npy',m3_wgt)
