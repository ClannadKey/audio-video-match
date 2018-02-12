from ResFunction import create_V2A_network,rowrank,columnrank,sum_rank
from keras import Input
import pre_treat_lrt
import numpy as np

cal_number=60

V,A=pre_treat_lrt.pre_treat_new('./filelists/Video_Name_List.txt','./filelists/Audio_Name_List.txt')

Vstd=np.load('Vstd.npy')
Astd=np.load('Astd.npy')
Vm=np.load('Vm.npy')
Am=np.load('Am.npy')

for i in range(128):
    A[:,:,i]=(A[:,:,i]-np.mean(A[:,:,i]))/Astd[i] 
for i in range(1024):
    V[:,:,i]=(V[:,:,i]-np.mean(V[:,:,i]))/Vstd[i]

Vu1=128
dropV=0.25
drop1=0.125

A_dim=(120,128)
V_dim=(120,1024)

A_input=Input(shape=A_dim)
V_input=Input(shape=V_dim)

V2A_network=create_V2A_network(A_dim,V_dim,Vu1,dropV,drop1)
dis=V2A_network([A_input,V_input])

V2A_network.summary()
V2A_network.load_weights('1999V2A.best.hdf5')

answer=np.zeros([cal_number,cal_number])
answer_var=np.zeros([cal_number,cal_number])
numberV=np.zeros([cal_number])
numberV_var=np.zeros([cal_number])

for i in range(0,cal_number): 
    for j in range(0,cal_number):
        a=A[i,:,:]
        a=a[np.newaxis,:,:]
        v=V[j,:,:]
        v=v[np.newaxis,:,:]

        pre=V2A_network.predict([a,v])
        answer[i,j]=np.mean(np.square(pre))
        
row=rowrank(answer)

sum_rank(row,cal_number,20)
np.save('lzq.npy',row)

