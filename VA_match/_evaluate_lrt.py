import numpy as np
import dis_cal_lrt
import pre_treat_lrt
from keras.models import load_model  
from keras import backend as K
from optparse import OptionParser
from tools.config_tools import Config

parser = OptionParser()
parser.add_option('--config',
                  type=str,
                  help="evaluation configuration",
                  default="./configs/test_config.yaml")

(opts, args) = parser.parse_args()
assert isinstance(opts, object)
opt = Config(opts.config)
#print(opt)

test_model=load_model(opt.init_model_lrt)
para_model=np.load(opt.para_model_lrt)

x_test,y_test=pre_treat_lrt.pre_treat_path(opt.video_flist,opt.audio_flist,opt.data_dir+'/')

y_pre=test_model.predict(x_test)
print('You use',int(len(y_pre)/118), 'samples for check')
dis_matrix1=np.array(dis_cal_lrt.mtx_cal1(y_pre,y_test,para_model,sample_for_check=int(len(y_pre)/118)))
print('The rank matrix is: \n',dis_matrix1)
print('And you get the acc:',end=' ')
dis_cal_lrt.dis_cal(np.array(dis_matrix1),opt.topk)

np.save('lrt_mtx.npy',dis_matrix1)