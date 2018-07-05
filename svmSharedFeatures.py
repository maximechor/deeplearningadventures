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
tunned_parameters["C"] = [0.0000001,0.000001, 0.00001,0.0001,0.001,0.01,0.1]

regr_valence = LinearSVR(random_state=None)
regr_arousal = LinearSVR(random_state=None)

#gridsearch_valence = GridSearchCV(regr_valence, tunned_parameters,cv=20, verbose=140)
#gridsearch_arousal = GridSearchCV(regr_arousal, tunned_parameters,cv=20, verbose=140)

gridsearch_valence = GridSearchCV(regr_valence, tunned_parameters, scoring=utils.CCC_scorer, cv=20, verbose=140)
gridsearch_arousal = GridSearchCV(regr_arousal, tunned_parameters, scoring=utils.CCC_scorer, cv=20, verbose=140)

gridsearch_valence.fit(encoded_train, train_valence.ravel())
gridsearch_arousal.fit(encoded_train, train_arousal.ravel())

hyper_best_params_valence = gridsearch_valence.best_params_['C']
print("valence best params:", hyper_best_params_valence)
hyper_best_params_arousal = gridsearch_arousal.best_params_['C']
print("arousal best params:", hyper_best_params_arousal)

best_gridsearch_score_valence= gridsearch_valence.best_score_
print("valence best_gridsearch_score:",best_gridsearch_score_valence)
best_gridsearch_score_arousal= gridsearch_arousal.best_score_
print("arousal best_gridsearch_score:",best_gridsearch_score_arousal)
 
best_gridsearch_estimator_valence = gridsearch_valence.best_estimator_
print("valence best_gridsearch_estimator:", best_gridsearch_estimator_valence)
best_gridsearch_estimator_arousal = gridsearch_arousal.best_estimator_
print("arousal best_gridsearch_estimator:", best_gridsearch_estimator_arousal)

best_coef_valence = best_gridsearch_estimator_valence.coef_
print("best_coef_valence:", best_coef_valence)
#print("best_coef_shape:", best_coef.shape)
best_coef_arousal = best_gridsearch_estimator_arousal.coef_
print("best_coef_arousal:", best_coef_arousal)

prediction_valence = best_gridsearch_estimator_valence.predict(encoded_dev)
print("valence prediction:", prediction_valence)
print("prediction_valence.shape:", prediction_valence.shape)
prediction_arousal = best_gridsearch_estimator_arousal.predict(encoded_dev)
print("arousal prediction:", prediction_arousal)
print("prediction_arousal.shape:", prediction_arousal.shape)

score_valence = best_gridsearch_estimator_valence.score(encoded_dev, dev_valence)
print("score_valence:",score_valence)
score_arousal = best_gridsearch_estimator_arousal.score(encoded_dev, dev_arousal)
print("score_arousal:",score_arousal)

####################Calculate CCC#############

val_ccc_valence = utils.CCC(dev_valence, prediction_valence)
print("val_ccc_valence:", val_ccc_valence)
val_ccc_arousal = utils.CCC(dev_arousal, prediction_arousal)
print("val_ccc_arousal:", val_ccc_arousal)

#print("Start Plotting")
#plt.plot(encoded_train,predict_regr, color ='c', lw =2, label ='Linear model')
#plt.xlabel('features encoded')
#plt.ylabel('predict SVR')
#plt.title('Support Vector Regression')
#plt.legend()
#plt.savefig('svm.jpeg')


#######################Write file to save val_CCC###############
df = pd.DataFrame(data={'Val CCC':[val_ccc_valence], 'score':[score_valence]})
df.to_csv('/home/AN84020/training/featuresLearning/valCCC_1_valence.csv', sep=',', index = False)
df = pd.DataFrame(data={'Val CCC':[val_ccc_arousal], 'score':[score_arousal]})
df.to_csv('/home/AN84020/training/featuresLearning/valCCC_1_arousal.csv', sep=',', index=False)

print("CCC")