from __future__ import print_function
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.model_selection import train_test_split
from keras import regularizers


#############################load MNIST#################################
num_classes =10
batch_size = 64
epochs = 100

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
 
print('Training data shape : ', train_images.shape, train_labels.shape)
print('Testing data shape : ', test_images.shape, test_labels.shape)

#Split train into train and valid
train_images, valid_images, train_labels, valid_labels = train_test_split(train_images, train_labels, test_size=0.16666, random_state=1)
print('Training data shape : ', train_images.shape, train_labels.shape)
print('Validating data shape : ', valid_images.shape, valid_labels.shape)


train_images = train_images.reshape(50000, 784)
test_images = test_images.reshape(10000, 784)
valid_images = valid_images.reshape(10000, 784)

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
valid_images = valid_images.astype('float32')

train_images /= 255
test_images /= 255
valid_images /=255

print(train_images.shape[0], 'train samples')
print(test_images.shape[0], 'test samples')
print(valid_images.shape[0], 'valid samples')

# convert class vectors to binary class matrices
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)
valid_labels = keras.utils.to_categorical(valid_labels, num_classes)

print('Shape of label tensor y_test:', train_labels.shape)

###############Buil Model No regularisation####################################
sgd = keras.optimizers.SGD(lr=0.02, decay=0, momentum=0, nesterov=False)
##Initialize weights with Glorot Uniform and Biases to zeros
#model = Sequential()
#model.add(Dense(800, 
#                activation='relu', 
#                input_shape=(784,),
#                kernel_initializer='glorot_normal'))

#model.add(Dense(800, 
#                activation='relu',
#                kernel_initializer='glorot_normal'))

#model.add(Dense(num_classes, 
#                activation='softmax',
#                kernel_initializer='glorot_normal'))

#model.summary()

#model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


#trainig_noregula = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, verbose=1, 
#                                  validation_data=(valid_images, valid_labels))

#[test_loss, test_acc] = model.evaluate(test_images, test_labels)
#print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))


#plt.plot(trainig_noregula.history['acc'],'r',linewidth=3.0)
#plt.plot(trainig_noregula.history['val_acc'],'b',linewidth=3.0)
#plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
#plt.xlabel('Epochs ',fontsize=16)
#plt.ylabel('Accuracy',fontsize=16)
#plt.title('Accuracy Curves No Regularisation',fontsize=16)

#plt.savefig('plotNoRegu.png')
################################# L2################
model_l2 = Sequential()

model_l2.add(Dense(800, 
                activation='relu', 
                input_shape=(784,),
                kernel_regularizer=regularizers.l2(0.01),
                kernel_initializer='glorot_normal'))

model_l2.add(Dense(800, 
                activation='relu',
                kernel_regularizer=regularizers.l2(0.01),
                kernel_initializer='glorot_normal'))

model_l2.add(Dense(num_classes, 
                activation='softmax',
                kernel_regularizer=regularizers.l2(0.01),
                kernel_initializer='glorot_normal'))

model_l2.summary()

model_l2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


trainig_l2 = model_l2.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, verbose=2, 
                                  validation_data=(valid_images, valid_labels))

[test_loss, test_acc] = model_l2.evaluate(test_images, test_labels)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

plt.plot(trainig_l2.history['acc'],'r',linewidth=3.0)
plt.plot(trainig_l2.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves L2',fontsize=16)

plt.savefig('plotl2.png')





print("toto")
