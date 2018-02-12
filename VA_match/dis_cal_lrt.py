import numpy as np
def dis_cal(dis,tpk):
    cnt=0
    for i in range(len(dis)):
       if sum(dis[i][i]>=dis[i])<tpk+1:
             cnt+=1
    print(cnt*1.0/len(dis))   
def mtx_cal(result,test_value,sample_for_check=60,time_len=118):
    dis_matrix=[]
    for test_num in range(0,int(len(result)/time_len)):
        # to evaluate the accuracy of each y_test
        sum_of_tp=np.zeros(sample_for_check)
        for mtx_cnt in range(time_len):
            # to calculate the distance
            test=np.array([i%time_len==mtx_cnt for i in range(sample_for_check*time_len)])
            tp=np.array([ts for ts in map(lambda a:sum(((a-test_value[test_num*time_len+mtx_cnt])**2)),result[test])])
            sum_of_tp+=tp 
        tp_dis=np.array([qaz for qaz in map(lambda a:sum(a>sum_of_tp),sum_of_tp)])
        dis_matrix.append(tp_dis)    
    return dis_matrix
def mtx_cal1(result,test_value,m3_wgt,sample_for_check=60,time_len=118):
    dis_matrix=[]
    for test_num in range(0,int(len(result)/time_len)):
        # to evaluate the accuracy of each y_test
        sum_of_tp=np.zeros(sample_for_check)
        for mtx_cnt in range(time_len):
            # to calculate the distance
            test=np.array([i%time_len==mtx_cnt for i in range(sample_for_check*time_len)])
            tp=np.array([ts for ts in map(lambda a:sum(((a-test_value[test_num*time_len+mtx_cnt])**2)*m3_wgt),result[test])])
            sum_of_tp+=tp 
        tp_dis=np.array([qaz for qaz in map(lambda a:sum(a>sum_of_tp),sum_of_tp)])
        dis_matrix.append(tp_dis)    
    return dis_matrix
def mean_k(mtx):
    cnt=0
    for i in range(len(mtx)):
        cnt+=mtx[i][i]
    return cnt/len(mtx)
def matrix_T(matrix):
    mtx_t=matrix.T
    new_mtx=[]
    for i in range(len(matrix)):
        tp=[asdf for asdf in map(lambda a:sum(a>=mtx_t[i])-1,mtx_t[i])]
        new_mtx.append(tp)
    new_mtx=np.array(new_mtx)
    return new_mtx.T