from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  

import arff
import numpy as np
import pandas as pd
import scipy
import pdb

from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler
from keras.layers import concatenate, merge
from keras.utils import plot_model
from keras import initializers
from keras.models import Sequential
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.callbacks import TensorBoard
from keras.layers import Dropout
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

import utils
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
####################

print("Inside Deep autoencoders Multi Modal Shared Features")
##################LOAD VIDEO DATA###############################
x_train_video, x_test_video, x_valid_video = utils.LoadRecolaVideo()

print("x_train_video:", x_train_video.shape)
print("x_test_video:", x_test_video.shape)
print("x_valid_video:", x_valid_video.shape)

######################## LOAD AUDIO DATA ######################
x_train_audio, x_test_audio, x_valid_audio = utils.LoadRecolaAudio()
 
print("train_audio.shape:", x_train_audio.shape)
print("test_audio.shape:", x_test_audio.shape)
print("valid_audio.shape:", x_valid_audio.shape)

############################### Input Layers ##########################

input_train_Video = Input(shape=(316,), name='input_video')
input_train_Audio = Input(shape=(102,), name = 'input_audio')

############################### Separated Layers ##########################

video_branch = Dense(316, input_dim = 316, name ='encoded_video_branch', kernel_initializer=initializers.glorot_uniform(seed=None), bias_initializer='zeros', activation='relu')(input_train_Video)
video_branch.Dropout(0.5)
audio_branch = Dense(102, input_dim = 102, name = 'encoded_audio_branch',kernel_initializer=initializers.glorot_uniform(seed=None), bias_initializer='zeros')(input_train_Audio)
audio_branch.Dropout(0.5)

#model.add(Dropout(0.2, input_shape=(60,)))

#video_branch = Dense(316, activation='relu', name ='encoded_video_branch1')(video_branch)
#audio_branch = Dense(102, activation='relu', name = 'encoded_audio_branch1')(audio_branch)

#video_branch = Dense(316, activation='relu', name ='encoded_video_branch2')(video_branch)
#audio_branch = Dense(102, activation='relu', name = 'encoded_audio_branch2')(audio_branch)

############################### Shared Layers ##########################
sharedFeature = concatenate([video_branch,audio_branch])
print("sharedFeature.shape", sharedFeature.shape)

################ Build AutoEncoder ##########################

encoded_shared = Dense(512,kernel_initializer=initializers.glorot_uniform(seed=None) , bias_initializer='zeros',activation='relu', name ='encoder_512')(sharedFeature)
encoded_shared.Dropout(0.5)
encoded_shared = Dense(256,kernel_initializer=initializers.glorot_uniform(seed=None),bias_initializer='zeros', activation='relu', name ='encoder_256', )(encoded_shared)
encoded_shared.Dropout(0.5)
encoded_shared = Dense(128,kernel_initializer=initializers.glorot_uniform(seed=None),bias_initializer='zeros', activation='relu', name ='encoder_128')(encoded_shared)
encoded_shared.Dropout(0.5)
#encoded_shared = Dense(64, activation='relu', name ='encoder_64')(encoded_shared)

####Encoded#########
encoded  = Dense(128,kernel_initializer=initializers.glorot_uniform(seed=None),bias_initializer='zeros' ,activation='sigmoid', name ='autoencoder_128')(encoded_shared)

################## Build Decoder  ###########################
#decoded_shared = Dense(64, activation='relu', name ='decoder_64')(encoded)
decoded_shared = Dense(128,kernel_initializer=initializers.glorot_uniform(seed=None),bias_initializer='zeros',activation='relu', name ='decoder_128')(encoded)
decoded_shared.Dropout(0.5)
decoded_shared = Dense(256,kernel_initializer=initializers.glorot_uniform(seed=None),bias_initializer='zeros',activation='relu', name ='decoder_256')(decoded_shared)
decoded_shared.Dropout(0.5)
decoded_shared = Dense(512,kernel_initializer=initializers.glorot_uniform(seed=None),bias_initializer='zeros',activation= 'relu', name ='decoder_512')(decoded_shared)
decoded_shared.Dropout(0.5)
############################### Separated Layers ##########################
decoded_video_branch = Dense(316,kernel_initializer=initializers.glorot_uniform(seed=None),bias_initializer='zeros',activation='relu', name ='decoded_video_branch')(decoded_shared)
decoded_video_branch.Dropout(0.5)
decoded_audio_branch = Dense(102,kernel_initializer=initializers.glorot_uniform(seed=None),bias_initializer='zeros',activation='relu', name ='decoded_audio_branch')(decoded_shared)
decoded_audio_branch.Dropout(0.5)
############################### Output Layers ##########################
decoded_video_output = Dense(316,kernel_initializer=initializers.glorot_uniform(seed=None),bias_initializer='zeros',activation='relu', name ='output_video')(decoded_video_branch)
decoded_video_output.Dropout(0.5)
decoded_audio_output = Dense(102,kernel_initializer=initializers.glorot_uniform(seed=None),bias_initializer='zeros',activation='relu', name ='output_audio')(decoded_audio_branch)
decoded_audio_output.Dropout(0.5)
############################### Build Model ###############################

##########################################################################
# Maps an input to its reconstruction  
autoencoder = Model(inputs=[input_train_Video, input_train_Audio], outputs = [decoded_video_output, decoded_audio_output])
##########################################################################

##########################################################################
# Separate encoder model : Maps an input to its encoded representation
encoder = Model(inputs=[input_train_Video, input_train_Audio], outputs = encoded)
##########################################################################

##########################################################################
# Separate decoder model : Maps an encoded representation to its output
##########################################################################
encoded_input = Input(shape=(128,))

decoder_audio = autoencoder.layers[-7](encoded_input)
decoder_audio = autoencoder.layers[-6](decoder_audio)
decoder_audio = autoencoder.layers[-5](decoder_audio)
decoder_audio = autoencoder.layers[-3](decoder_audio)
decoder_audio = autoencoder.layers[-1](decoder_audio)

decoder_video = autoencoder.layers[-7](encoded_input)
decoder_video = autoencoder.layers[-6](decoder_video)
decoder_video = autoencoder.layers[-5](decoder_video)
decoder_video = autoencoder.layers[-4](decoder_video)
decoder_video = autoencoder.layers[-2](decoder_video)


decoder1 = Model(inputs = encoded_input, outputs = decoder_audio)
decoder2 = Model(inputs = encoded_input, outputs = decoder_video)
#########################################################################

autoencoder.summary()
encoder.summary()
decoder1.summary()
decoder2.summary()
plot_model(autoencoder,to_file='/home/AN84020/training/featuresLearning/Imagemodels/modelautoencoder_ShF_T4.png', show_shapes =True, show_layer_names=True )

#autoencoder.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['accuracy'])
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics = ['accuracy'])
autoencoder.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])

############################### Build Model and Train ##########################
now = time.strftime("%c")
tbcallback = TensorBoard(log_dir='./tmp/'+now, histogram_freq=0, write_graph=True, write_images=True )

autoencoderSharedFeatures = autoencoder.fit([x_train_video, x_train_audio], [x_train_video, x_train_audio],
                 epochs =15,
                 verbose =2,
                 batch_size = 50,
                 shuffle = True,
                 validation_data =( [x_test_video, x_test_audio], [x_test_video,x_test_audio] ),
                 callbacks=[tbcallback]
                 )

autoencoder.save('/home/AN84020/training/featuresLearning/autoencoder_ShF_T4.h5') 

# list all data in history
print(autoencoderSharedFeatures.history.keys())
#summarize history for accuracy video
plt.plot(autoencoderSharedFeatures.history['output_video_acc'])
plt.plot(autoencoderSharedFeatures.history['val_output_video_acc'])
plt.title('Model Accuracy Video')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/AN84020/training/featuresLearning/Imagemodels/modelaccuracyVideo_ShF_T4.jpeg')

# summarize history for accuracy audio
plt.plot(autoencoderSharedFeatures.history['output_audio_acc'])
plt.plot(autoencoderSharedFeatures.history['val_output_audio_acc'])
plt.title('Model Accuracy Audio')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/AN84020/training/featuresLearning/Imagemodels/modelaccuracyAudio_ShF_T4.jpeg')

# summarize history for loss video
plt.plot(autoencoderSharedFeatures.history['output_video_loss'])
plt.plot(autoencoderSharedFeatures.history['val_output_video_loss'])
plt.title('Model Loss Video')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/AN84020/training/featuresLearning/Imagemodels/modellossVideo_ShF_T4.jpeg')

# summarize history for loss
plt.plot(autoencoderSharedFeatures.history['output_audio_loss'])
plt.plot(autoencoderSharedFeatures.history['val_output_audio_loss'])
plt.title('Model Loss Audio')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/AN84020/training/featuresLearning/Imagemodels/modellossAudio_ShF_T4.jpeg')

# summarize history for loss
plt.plot(autoencoderSharedFeatures.history['loss'])
plt.plot(autoencoderSharedFeatures.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('/home/AN84020/training/featuresLearning/Imagemodels/modelloss_ShF_T4.jpeg')


############################### Encode and Decode some Input Vectors ##########################

encoded_feature_train = encoder.predict( [x_train_video, x_train_audio] )
encoded_feature_dev = encoder.predict( [x_valid_video, x_valid_audio] )

#decoded_video_features = decoder2.predict( encoded_video_features )
#decoded_audio_features = decoder1.predict( encoded_audio_features )

np.savetxt('/home/AN84020/training/featuresLearning/encoded_ShF_train_1.csv', encoded_feature_train, delimiter=",")

np.savetxt("/home/AN84020/training/featuresLearning/encoded_ShF_dev_1.csv", encoded_feature_dev, delimiter=",")
# Show the decoded vectors

#n = 10  # how many input vectors we will display

#print("Decoded Video Features\n")
#for i in range(n):
#	print("%f", decoded_video_features[i])

#print("\nDecoded Audio Features\n")
#for i in range(n):
#	print("%f", decoded_audio_features[i])


