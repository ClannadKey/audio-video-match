import numpy as np 
from ResFunction import rowrank,columnrank,counttop5,sum_rank

cal=60

lzy=np.load('LSTM_lzy.npy')
lzq=np.load('lzq.npy')
lrt=np.load('lrt_mtx.npy')

lzy=rowrank(lzy)
lzq=rowrank(lzq)
lrt=rowrank(lrt)

#print('origin：')
lzy_sum=sum_rank(lzy,cal,15)
lzq_sum=sum_rank(lzq,cal,15)
lrt_sum=sum_rank(lrt,cal,15)

#print('sum_sum:')
answer=sum_rank(lzy_sum+lzq_sum+lrt_sum,cal,15)
print(answer)
np.save('answer.npy',answer)