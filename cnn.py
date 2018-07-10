# -*- coding: utf-8 -*-
from generator10 import SEWA_juan_generator
from generator10 import my_callback
import pandas as pd
import glob
import time
import numpy as np 
import scipy as sc
import csv   as cs
import os 
from shutil import copyfile
import fnmatch
import os.path
from skimage import io
from keras.models import Sequential
from keras.optimizers import Adam,SGD
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from PIL import Image
#K.set_image_dim_ordering('th')
K.set_image_data_format('channels_last')
from keras.callbacks import TensorBoard,CSVLogger, TensorBoard
import time
CUDA_VISIBLE_DEVICES=0
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import keras.backend.tensorflow_backend as tf_bkend
tf_bkend.set_session(sess)

from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import TensorBoard
from skimage import color, exposure, transform
import h5py



###########################PARAMETERS#############################
dimen  ='V'
dat_aug='Y'
pre_tra='Y'#,(Y,N,X)
lay_blo=0
ler=1e-6
batch=30
delay=50 #frames
norm='Y'
last='linear'
tri='_2806_Valance2.4'
##################################################################
file=dimen+dat_aug+pre_tra+str(lay_blo)+str(ler)+str(batch)+str(delay)+norm+last+tri
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
import sys

model_name = sys.argv[0][:-3]

if dimen=='A':
	path_image_train='/home/AN35190/Recola/video_frames/48x48/prepro/train/'
	path_image_dev='/home/AN35190/Recola/video_frames/48x48/prepro/dev/'
	path_labels_train='/home/AN35190/Recola/video_frames/48x48/prepro/labels/train/'
	path_labels_dev='/home/AN35190/Recola/video_frames/48x48/prepro/labels/dev/'

if dimen=='V':
	path_image_train='/home/AN35190/Recola/video_frames/48x48/prepro/train/'
	path_image_dev='/home/AN35190/Recola/video_frames/48x48/prepro/dev/'
	path_labels_train='/home/AN35190/Recola/video_frames/48x48/prepro/labels/valance_50/train/'
	path_labels_dev='/home/AN35190/Recola/video_frames/48x48/prepro/labels/valance_50/dev/'
print '===========CONFIG============'
print path_labels_train
print file
print 'Dimen=',dimen
print 'Data augmentation=',dat_aug
print 'Pre trained =',pre_tra
print 'NUmber of layer blocked= ',lay_blo
print 'Learning Rate =',ler
print 'Batch Size= ',batch
print 'Delay in frames= ',delay
print 'Input Normalized=',norm
print 'Activation func Last neuron= ',last
print 'Notes= ',tri
print '============================='

batch_generator_train=SEWA_juan_generator(batch)
batch_generator_dev  =SEWA_juan_generator(batch)

train=np.train=[True,True,True,True,True]
if lay_blo==1:
	train[0]=False
if lay_blo==2:
	train[1]=False
	train[0]=False
if lay_blo==3:
	train[2]=False
	train[1]=False
	train[0]=False
if lay_blo==4:
	train[3]=False
	train[2]=False
	train[1]=False
	train[0]=False
if lay_blo==5:
	train[4]=False
	train[3]=False
	train[2]=False
	train[1]=False
	train[0]=False

# learning rate schedule
def step_decay(epoch):
	if epoch<=60:
		lrate=1e-5
	if (epoch<=100) & (epoch>60):
		lrate=1e-6
	if epoch>100:
		lrate=1e-7
	return lrate

def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(64, (3, 3), input_shape=(48, 48,1),trainable=train[0]))#64,100
	model.add(Activation('relu'))
	BatchNormalization(axis=-1) # https://yashk2810.github.io/Applying-Convolutional-Neural-Network-on-the-MNIST-dataset/
	model.add(Conv2D(64, (3, 3),trainable=train[1]))#64,100
	model.add(Activation('tanh'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	BatchNormalization(axis=-1)
	model.add(Conv2D(128, (2, 2),trainable=train[2]))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	BatchNormalization()
	model.add(Dense(100,trainable=train[3]))
	model.add(Activation('tanh'))
	BatchNormalization()
	model.add(Dropout(0.25))
	model.add(Dense(50,trainable=train[4]))
	model.add(Activation('relu'))
	model.add(Dense(10))
	model.add(Activation('tanh'))
	model.add(Dense(7))
	model.add(Activation('softmax'))
	if pre_tra=='Y':
		model.load_weights('/home/AN35190/fer+/fer+_data/data_48x48/best/best.hdf5')
	model.layers.pop()
	model.outputs = [model.layers[-1].output]
	model.layers[-1].outbound_nodes = []
	model.add(Dense(1, activation=last))
	# Compile model
	optimizer=Adam(lr=ler)
	model.compile(loss='mse', optimizer=optimizer, metrics=['mse'] )
	return model




model = larger_model()
model.summary()
call=my_callback(file)
lrate = LearningRateScheduler(step_decay)

csv_logger=CSVLogger('log_'+file+'.csv',append=True,separator=';')

model.fit_generator(batch_generator_train.my_generator_train(),epochs=500,validation_data=batch_generator_dev.my_generator_dev(),steps_per_epoch=int(52352*10/batch),validation_steps=int(59175/batch),callbacks=[call,csv_logger,TensorBoard(log_dir='./logs/'+str(file))], verbose = 2)

#247500
#train 297000/batch   dev 49800/batch
#checkpointer=ModelCheckpoint(filepath='/home/AN35190/'+"weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5",monitor='val_loss',verbose=1,save_best_only=True,mode='min')
#model.fit_generator(batch_generator_train.my_generator_train(),epochs=200,validation_data=batch_generator_dev.my_generator_dev(),steps_per_epoch=100,validation_steps=100,callbacks=[call,csv_logger], verbose = 1)
#model.fit_generator(batch_generator_train.my_generator_train(),epochs=200,steps_per_epoch=100,callbacks=[call,csv_logger], verbose = 1)
