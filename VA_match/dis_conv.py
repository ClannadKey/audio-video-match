from keras.models import Model
from keras.layers import Dense,Dropout,Input,Add,Multiply,Flatten,noise,Lambda,Subtract,Conv1D#,LSTM,Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras import regularizers,initializers
from keras.layers.core import Reshape
from keras.constraints import min_max_norm,non_neg,unit_norm
import numpy as np
from keras import backend as K

class Constraint(object):
    def __call__(self, w):
        return w
    def get_config(self):
        return {}
class Unit_Abs(Constraint):
    def __init__(self, axis=0):
        self.axis = axis
    def __call__(self, w):
        return K.abs(w) / (K.epsilon() + (K.sum(K.abs(w),
                                               axis=self.axis,
                                               keepdims=True)))
    def get_config(self):
        return {'axis': self.axis}

def res_unit(input_layer,dense_num):
	tp1=PReLU()(input_layer)
	tp2=Dropout(0.2)(BatchNormalization()(tp1))
	tp3=Dense(dense_num,kernel_initializer=initializers.lecun_normal())(tp2)
	union_layer=Add()([input_layer,tp3])

	return union_layer

def res_model1(input_shape1,input_shape2):

	x_input=Input(shape=(input_shape1,),name='x_input')
	output_1=Dropout(0.3)(PReLU()(x_input))
	output0=Dense(input_shape2,kernel_initializer=initializers.lecun_normal())(output_1)
	output1=res_unit(output0,input_shape2)

#	output7=Dense(output_shape,kernel_initializer=initializers.lecun_normal())(Dropout(0.3)(output6))
	
	model=Model(inputs=x_input,outputs=output1)
	return model
def res_model2(input_shape1,input_shape2):

	x_input=Input(shape=(input_shape1,),name='x_input')
	
	output_1=Dropout(0.3)(x_input)
	output0=Dense(input_shape2,kernel_initializer=initializers.lecun_normal())(output_1)
	output1=res_unit(output0,input_shape2)
	output2=res_unit(output1,input_shape2)
#	output7=Dense(output_shape,kernel_initializer=initializers.lecun_normal())(Dropout(0.3)(output6))
	
	model=Model(inputs=x_input,outputs=output2)
	return model

def model_3(input_shape):
	x_input1=Input(shape=(input_shape,),name='x_input1')
	x_input2=Input(shape=(input_shape,),name='x_input2')
	check_layer1=Subtract()([x_input1,x_input2])
	check_layer2=Lambda(lambda x: K.square(x))(check_layer1)
	check_layer3=Reshape((input_shape,1))(check_layer2)
	check_layer4=Conv1D(filters=1,kernel_size=input_shape,use_bias=False,kernel_constraint=Unit_Abs(),kernel_initializer=initializers.lecun_normal())(check_layer3)
#   output7=Dense(output_shape,kernel_initializer=initializers.lecun_normal())(Dropout(0.3)(output6))
	check_layer5=Flatten()(check_layer4)
	model=Model(inputs=[x_input1,x_input2],outputs=check_layer5)
	return model

if __name__=="__main__":
	tp=res_model1(1024,128)
#	plot_model(tp, to_file='model.png')


