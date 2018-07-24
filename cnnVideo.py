from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  
filepath = "/home/AP85900/Multimodal Learning/training_Cnn/2018-07-23-0/"

import arff #liac_arff
import numpy as np
import pandas as pd
import scipy
import pdb

from keras.layers import Input, Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import concatenate, merge
from keras.models import Model
from keras import regularizers
from keras.utils import plot_model
from keras import initializers
from keras.models import Sequential
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.callbacks import TensorBoard
import time

# Code for avoiding keras + tensorflow from using all memory:
# Similar to the solution above, but also need to manually setup the session on Keras back-end:
import tensorflow as tf
#config = tf.ConfigProto(device_count = {'GPU': 0})
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
import keras.backend.tensorflow_backend as tf_bkend
tf_bkend.set_session(sess)

import utils_cnn
import matplotlib.pyplot as plt
plt.switch_backend('agg')

####################
print("Inside Convolutional Autoencoders")

##################LOAD VIDEO DATA###############################
x_train_video, x_valid_video = utils_cnn.LoadRecolaVideo()

print("x_train_video:", x_train_video.shape)
print("x_valid_video:", x_valid_video.shape)

################ Build AutoEncoder ##########################


############################### Input Layer ##########################
input_img = Input(shape=(48, 48, 1), name ='input_video')  # black and white images 48x48x1=2304

################## Build Encoder  ###########################
encoded = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', data_format="channels_last", name ='encoder_conv2D_1')(input_img) #48x48x32
encoded = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name ='encoder_maxPooling2D_1')(encoded)  #24x24x16
encoded = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', data_format="channels_last", name ='encoder_conv2D_2')(encoded)  #24x24x64
encoded = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name ='encoder_maxPooling2D_2')(encoded)  #12x12x8
encoded = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', data_format="channels_last", name ='encoder_conv2D_3')(encoded)  #12x12x128
encoded = MaxPooling2D(pool_size=(2, 2), padding='same', data_format="channels_last", name ='encoder_maxPooling2D_3')(encoded)  #6x6x8

# at this point the representation is (6, 6, 128) i.e. 4608-dimensional

################## Build Decoder  ###########################
decoded = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last", name ='decoder_conv2D_1')(encoded)  #6x6x128
decoded = UpSampling2D((2, 2), data_format="channels_last", name ='decoder_upSampling2D_1')(decoded)  #12x12x8
decoded = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last", name ='decoder_conv2D_2')(decoded)  #12x12x64
decoded = UpSampling2D((2, 2), data_format="channels_last", name ='decoder_upSampling2D_2')(decoded)  #24x24x8
decoded = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last", name ='decoder_conv2D_3')(decoded)   #24x24x32
decoded = UpSampling2D((2, 2), data_format="channels_last", name ='decoder_upSampling2D_3')(decoded)  #48x48x16
decoded_cnn = Conv2D(1, (3, 3), activation='sigmoid', padding='same', data_format="channels_last", name ='output_video')(decoded)  #48x48x1

############################### Build Model #############################
autoencoder = Model(input_img, decoded_cnn)
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(6, 6, 128,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-7](encoded_input)
decoder_layer = autoencoder.layers[-6](decoder_layer)
decoder_layer = autoencoder.layers[-5](decoder_layer)
decoder_layer = autoencoder.layers[-4](decoder_layer)
decoder_layer = autoencoder.layers[-3](decoder_layer)
decoder_layer = autoencoder.layers[-2](decoder_layer)
decoder_layer = autoencoder.layers[-1](decoder_layer)
# create the decoder model
decoder = Model(inputs = encoded_input, outputs = decoder_layer)

print("Autoencoder Summary")
autoencoder.summary()
plot_model(autoencoder,to_file=filepath + 'modelautoencoder_Cnn_T4.png', show_shapes =True, show_layer_names=True )
print("\n")
print("Encoder Summary")
encoder.summary()
print("\n")
print("Decoder Summary")
decoder.summary()
print("\n")

############################### Build Model and Train ##########################
autoencoder.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])

now = time.strftime("%c")
tbcallback = TensorBoard(log_dir='./tmp_cnn/'+now, histogram_freq=0, write_graph=True, write_images=True )

autoencoderCnn = autoencoder.fit(x_train_video, x_train_video,
                 epochs =100,
                 verbose =2,
                 batch_size = 128,
                 shuffle = True,
                 validation_data =(x_valid_video, x_valid_video),
                 callbacks=[tbcallback]
                 )

############################### Save Results ##########################
autoencoder.save(filepath + 'autoencoder_Cnn_T4.h5')

#loading the best weights
#autoencoder.load_weights(filepath + 'autoencoder_Cnn_T4.h5')

# list all data in history
#print(autoencoderCnn.history.keys())


#summarize history for accuracy video
plt.clf()
plt.plot(autoencoderCnn.history['acc'])
plt.plot(autoencoderCnn.history['val_acc'])
plt.title('Model Accuracy Video')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(filepath + 'modelaccuracyVideo_Cnn_T4.png')

# summarize history for loss video
plt.clf()
plt.plot(autoencoderCnn.history['loss'])
plt.plot(autoencoderCnn.history['val_loss'])
plt.title('Model Loss Video')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(filepath + 'modellossVideo_Cnn_T4.png')

############################### Encode and Decode some Input Vectors ##########################
encoded_feature_train = encoder.predict(x_train_video)
encoded_feature_dev = encoder.predict(x_valid_video)
#reshape
encoded_feature_train = encoded_feature_train.reshape(encoded_feature_train.shape[0],np.prod(encoded_feature_train.shape[1:]))
encoded_feature_dev = encoded_feature_dev.reshape(encoded_feature_dev.shape[0],np.prod(encoded_feature_dev.shape[1:]))

np.savetxt(filepath + 'encoded_Cnn_train_100.csv', encoded_feature_train, delimiter=",")
np.savetxt(filepath + 'encoded_Cnn_dev_100.csv', encoded_feature_dev, delimiter=",")