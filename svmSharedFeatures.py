from __future__ import print_function
import os
#os.environ['CUDA_VISIBLE_DEVICES'] ='1'
import numpy as np
import scipy
import time
import keras
import tensorflow
from keras.callbacks import TensorBoard
from sklearn.svm import SVR
import pdb
import arff
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.svm import LinearSVR
from sklearn.datasets import make_regression
import utils 
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import csv
from sklearn.model_selection import GridSearchCV

####################### Load Shared Feature Files####################
#encoded_train = pd.read_csv('encoded_train1.csv', header = None)
encoded_train = pd.read_csv('/home/AN84020/training/featuresLearning/encoded_ShF_train_3.csv',header =None)
print("encoded_train:", encoded_train.shape)

#encoded_dev = pd.read_csv('encoded_valid1.csv', header = None)
encoded_dev = pd.read_csv('/home/AN84020/training/featuresLearning/encoded_ShF_dev_3.csv', header = None)
print("encoded_dev:", encoded_dev.shape)

#pdb.set_trace()
###########################Load Labels##############################
################

#############For Arousal###########################
train_arousal = arff.load(open('/data/databases/Recola/ratings_gold_standard/arousal/train_all.arff'))
train_arousal =[row[2:] for row in train_arousal['data']]
train_arousal = np.array(train_arousal)

print("train_arousal.shpe:", train_arousal.shape)

dev_arousal = arff.load(open('/data/databases/Recola/ratings_gold_standard/arousal/dev_all.arff'))
dev_arousal =[row[2:] for row in dev_arousal['data']]
dev_arousal = np.array(dev_arousal)

print("dev_arousalshape:", dev_arousal.shape)


###################################For Valence################################
train_valence = arff.load(open('/data/databases/Recola/ratings_gold_standard/valence/train_all.arff'))

train_valence = [row[2:] for row in train_valence['data']]
train_valence = np.array(train_valence)

print("train_valence.shape:", train_valence.shape)

dev_valence = arff.load(open('/data/databases/Recola/ratings_gold_standard/valence/dev_all.arff'))
dev_valence = [row[2:] for row in dev_valence['data']]
dev_valence = np.array(dev_valence)
#print("dev_valence.shape:",dev_valence.shape)


#pdb.set_trace()

#############################Scale Data##########################
encoded_train = preprocessing.scale(encoded_train)
print("encoded_train_scale:", encoded_train.shape)
#train_arousal = y_scaling.fit_transform(train_arousal)
print("train_arousal:", train_arousal.shape)
encoded_dev = preprocessing.scale(encoded_dev)
print("encoded_test_scale:", encoded_dev.shape)
#pdb.set_trace()

############################Start SVM Training####################
print("Start SVM")

######################### Linear SVR ##########################
#regr = make_regression(n_features=128,random_state=0)
#regr = LinearSVR(random_state=0)
#regr.fit(encoded_train,train_arousal)
#regr.fit(encoded_train, train_valence)
#print("regr.coef:",regr.coef_)

#prediction = regr.predict(encoded_dev)
#print("predict_regr:", prediction)
#print("predict_regr_shape:", prediction.shape)

#regr_score = regr.score(encoded_dev, dev_arousal)
#regr_score = regr.score(encoded_dev, dev_valence)
#print("regr_score:", regr_score)
#print("regr_score.shape:", regr_score.shape)

################## GridSearchCV And Linear SVR#####################

tunned_parameters = {'C':'C'}
tunned_parameters["C"] = [0.00001,0.0001,0.001,0.1,1]

regr= LinearSVR(random_state=None)

gridsearch = GridSearchCV(regr, tunned_parameters,cv=20, verbose=20)

#gridsearch.fit(encoded_train, train_arousal.ravel())
gridsearch.fit(encoded_train, train_valence.ravel())

hyper_best_params= gridsearch.best_params_['C']
print("best params:", hyper_best_params)

best_gridsearch_score= gridsearch.best_score_
print("best_gridsearch_score:",best_gridsearch_score)
 
best_gridsearch_estimator = gridsearch.best_estimator_
print("best_gridsearch_estimator:", best_gridsearch_estimator)

best_coef = best_gridsearch_estimator.coef_
print("best_coef:", best_coef)
print("best_coef_shape:", best_coef.shape)

prediction = best_gridsearch_estimator.predict(encoded_dev)
print("prediction:", prediction)
print("prediction.shape:", prediction.shape)

#score = best_gridsearch_estimator.score(encoded_dev, dev_arousal)
score = best_gridsearch_estimator.score(encoded_dev, dev_valence)
print("score:",score)

####################Calculate CCC#############

#val_ccc = utils.CCC(dev_arousal,prediction)
val_ccc = utils.CCC(dev_valence, prediction)

print("val_ccc:", val_ccc)

#print("Start Plotting")
#plt.plot(encoded_train,predict_regr, color ='c', lw =2, label ='Linear model')
#plt.xlabel('features encoded')
#plt.ylabel('predict SVR')
#plt.title('Support Vector Regression')
#plt.legend()
#plt.savefig('svm.jpeg')


#######################Write file to save val_CCC###############
df = pd.DataFrame(data={'Val CCC':[val_ccc], 'score':[score]})
#df.to_csv('/home/AN84020/training/featuresLearning/valCCC_3_arousal.csv', sep=',', index = False)
df.to_csv('/home/AN84020/training/featuresLearning/valCCC_3_valence.csv', sep=',', index=False)
