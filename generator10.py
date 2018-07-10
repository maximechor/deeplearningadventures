# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
from shutil import copyfile,copy2
import os, errno
import scipy
from scipy import misc,ndimage
from scipy.misc import imsave
from PIL import Image
import keras
from scipy.cluster.vq import whiten
import Augmentor
import random
from  random import randint
from scipy.ndimage import filters
import sys

dimen='V'

if dimen=='A':
	dim='arousal'
	path_image_train='/home/AN35190/Recola/video_frames/48x48/prepro/train/'
	path_image_dev='/home/AN35190/Recola/video_frames/48x48/prepro/dev/'
	path_labels_train='/home/AN35190/Recola/video_frames/48x48/prepro/labels/train/'
	path_labels_dev='/home/AN35190/Recola/video_frames/48x48/prepro/labels/dev/'

if dimen=='V':
	dim='valance'
	path_image_train='/home/AN35190/Recola/video_frames/48x48/prepro/train/'
	path_image_dev='/home/AN35190/Recola/video_frames/48x48/prepro/dev/'
	path_labels_train='/home/AN35190/Recola/video_frames/48x48/prepro/labels/valance_50/train/'
	path_labels_dev='/home/AN35190/Recola/video_frames/48x48/prepro/labels/valance_50/dev/'


def calc_scores ( x, y ):
    # Computes the metrics CCC, PCC, and RMSE between the sequences x and y
    #  CCC:  Concordance correlation coeffient
    #  PCC:  Pearson's correlation coeffient
    #  RMSE: Root mean squared error
    # Input:  x,y: numpy arrays (one-dimensional)
    # Output: CCC,PCC,RMSE
	
	x_mean = np.nanmean(x)
	y_mean = np.nanmean(y)

	covariance = np.nanmean((x-x_mean)*(y-y_mean))

	x_var = 1.0 / (len(x)-1) * np.nansum((x-x_mean)**2) # Make it consistent with Matlab's nanvar (division by len(x)-1, not len($
	y_var = 1.0 / (len(y)-1) * np.nansum((y-y_mean)**2)

	CCC = (2*covariance) / (x_var + y_var + (x_mean-y_mean)**2)

	x_std = np.sqrt(x_var)
	y_std = np.sqrt(y_var)

    #PCC = covariance / (x_std * y_std)

    #RMSE = np.sqrt(np.nanmean((x - y)**2))

    #scores = np.array([ CCC, PCC, RMSE ])
	scores = np.array(CCC)
	return scores


class SEWA_juan_generator(object):
	#Constructor:
	def __init__(self, batch):
		#self.my_generator_train()
		#self.my_generator_dev()
		self.batch = batch
		self.dev_ccc  = []
		self.pair=[]
		self.mem_i=0
		self.mem_j=0
		self.mem_i_dev=0
		self.mem_j_dev=0
		self.bandera=0
		self.bandera_dev=0

	def my_generator_train(self):
		while True:
			i=self.mem_i
			j=self.mem_j
			file_list =os.listdir(path_image_train)#List of speakers to train the model (train1 to train 9)
			file_list.sort()#Organize the speakers
			label_list=os.listdir(path_labels_train)#List of labels train set()
			label_list.sort()
			#print len(file_list)
			self.pair=[]
			list=[]
			mi_tensor_x=[]
			mi_tensor_y=[]
			paresx=[]
			paresy=[]
			cuantos=0
			#pair=numpy.array(['',0]) 
			p = Augmentor.Pipeline()
			for i in range(0,len(file_list)):
				#mem_i=i
				frames=os.listdir(path_image_train+file_list[i])
				frames.sort()
				label=pd.read_csv(path_labels_train+'/'+file_list[i]+'/'+'dropeada_50_'+dim+'_'+file_list[i]+'.csv')
				self.number_train=label.size
				frame_list=range(0,len(frames)-1)
				random.shuffle(frame_list)
				for j in range(0,len(frames)-1):					
					#mem_j=j
					k=frame_list[j]
					X=Image.open(path_image_train+'/'+file_list[i]+'/'+frames[k])
					Y=(label.iloc[k,0])
					if len(mi_tensor_y)<self.batch and self.bandera==0:
						if len(frames)-1<=j or self.bandera==0:
							paresx.append(file_list[i]+'/'+frames[k])
							paresy.append(Y)
							mem_i=i
							mem_j=j
							#Data Augmentation 
							x1 = np.fliplr(X)						 	#Flip
							x1 = np.asarray(x1,dtype=np.float32).reshape(48,48,1)
							x1 = x1/255
							mi_tensor_x.append(x1)
							mi_tensor_y.append(Y)
							x2 = scipy.ndimage.rotate(X,random.randint(5,45),reshape=False)  #Random Rotate
							x2 = np.asarray(x2,dtype=np.float32).reshape(48,48,1)
							x2 = x2/255
							mi_tensor_x.append(x2)
							mi_tensor_y.append(Y)
							x3 = scipy.ndimage.rotate(X,random.randint(5,50),reshape=False)	#Random Rotate2	
							x3 = np.asarray(x3,dtype=np.float32).reshape(48,48,1)
							x3 = x3/255
							mi_tensor_x.append(x3)
							mi_tensor_y.append(Y) 
							x4 = scipy.ndimage.rotate(X,random.randint(5,50),reshape=False)	#Random Rotate + Flip	
							x4 = np.fliplr(x4)							
							x4 = np.asarray(x4,dtype=np.float32).reshape(48,48,1)
							x4 = x4/255
							mi_tensor_x.append(x4)
							mi_tensor_y.append(Y)
							x5 = filters.gaussian_filter(X,randint(2,4))			#Random Gaussian 		
							x5 = np.asarray(x5,dtype=np.float32).reshape(48,48,1)
							x5 = x5/255
							mi_tensor_x.append(x5)
							mi_tensor_y.append(Y)
							X  = np.asarray(X,dtype=np.float32).reshape(48,48,1)		#Normal Image
							X  = X/255
							mi_tensor_x.append(X)
							mi_tensor_y.append(Y)
							#print file_list[i]+'/'+frames[j]
							#cuantos=cuantos+6
						else:
							self.bandera=1
							mem_i=i
							mem_j=j
					else:
						
						mem_i=i
						mem_j=j						
						mi_tensor_x=np.array(mi_tensor_x)
						mi_tensor_y=np.array(mi_tensor_y)
						#mi_tensor_x=(mi_tensor_x-np.mean(mi_tensor_x))/(np.std(mi_tensor_x))
						#print 'TRAIN TENSOR SHAPE',mi_tensor_x.shape,mi_tensor_y.shape
						#print 'cuantos ',cuantos
						#for x in range(len(paresx)):
    						#	print paresx[x],paresy[x]
						#	print ('')
						yield mi_tensor_x,mi_tensor_y
						mi_tensor_x=[]
						mi_tensor_y=[]
						self.bandera=0
	def my_generator_dev(self):
		while True:
			i=self.mem_i_dev
			j=self.mem_j_dev
			file_list =os.listdir(path_image_dev)
			file_list.sort()
			label_list=os.listdir(path_labels_dev)
			label_list.sort()
			#print len(file_list)
			self.pair=[]
			list=[]
			mi_tensor_x=[]
			mi_tensor_y=[]
			cuantosdev=0
			mi_batch=[]
			for i in range(0,len(file_list)):
				frames=os.listdir(path_image_dev+file_list[i])
				frames.sort()
				label=pd.read_csv(path_labels_dev+'/'+file_list[i]+'/'+'dropeada_50_'+dim+'_'+file_list[i]+'.csv')
				self.number_train=label.size
				for j in range(0,len(frames)-1):
					X=Image.open(path_image_dev+'/'+file_list[i]+'/'+frames[j])
					Y=(label.iloc[j,0])
					if len(mi_tensor_y)<self.batch and self.bandera_dev==0:
						if  len(frames)-1<=j or self.bandera_dev==0:
							mem_i_dev=i
							mem_j_dev=j
							X=np.asarray(X,dtype=np.float32).reshape(48,48,1)
							X=X/255
							#X=(X-np.mean(X))/(np.std(X))
							mi_tensor_x.append(X)
							mi_tensor_y.append(Y)	
							#mi_batch.append(file_list[i]+'/'+frames[j])
							#cuantosdev=cuantosdev+1
						else:
							j=j-1
							mem_i_dev=i
							mem_j_dev=j
							self.bandera_dev=1
					else:
						mi_tensor_x=np.array(mi_tensor_x)
						mi_tensor_y=np.array(mi_tensor_y)				
						yield mi_tensor_x,mi_tensor_y
						mi_tensor_x=[]
						mi_tensor_y=[]
						#mi_batch=[]
						self.bandera_dev=0

				
class my_callback(keras.callbacks.Callback):
	def __init__(self,file):
		self.file=file
		self.pair=[]
		self.train_ccc = []
		self.train_ccc2= []
		self.dev_ccc   = []
		self.dev_ccc2  = []
		self.predictions=[]
		self.predictions_dev=[]
		self.ccc_prev=-1.000
		#self.batch_generator_dev=batch_generator_dev
		#self.batch_generator_dev=batch_generator_dev
		
	def on_epoch_end(self, epoch, logs={}):
		file_list =os.listdir(path_image_dev)
		file_list.sort()
		label_list=os.listdir(path_labels_dev)
		label_list.sort()
		self.pair=[]
		list=[]
		mi_tensor_x=[]
		mi_tensor_y=[]
		for i in range(0,len(file_list)):
			frames=os.listdir(path_image_dev+file_list[i])
			frames.sort()
			label=pd.read_csv(path_labels_dev+'/'+file_list[i]+'/'+'dropeada_50_'+dim+'_'+file_list[i]+'.csv')
			self.number_dev=label.size
			for j in range(0,len(frames)-1):
				X=Image.open(path_image_dev+'/'+file_list[i]+'/'+frames[j])
				Y=(label.iloc[j,0])
				X=np.asarray(X,dtype=np.float32).reshape(48,48,1)
				X=X/255
				mi_tensor_x.append(X)
				mi_tensor_y.append(Y)
		#val_pred=np.asarray(self.model.predict(np.asarray(mi_tensor_x)))


		val_pred=np.reshape(self.model.predict(np.asarray(mi_tensor_x)),-1)
		val_gt  =np.asarray(mi_tensor_y)
		print 'MI TENSOR CCC DEV',val_pred.shape,val_gt.shape
		
		np.save('PRED_DEV_'+self.file,val_pred)
		np.save('LABE_DEV_'+self.file,val_gt)
		

		ccc_dev=calc_scores(val_gt,val_pred)
		self.dev_ccc.append(ccc_dev)
		thefile = open('dev_ccc_'+self.file+'.txt', 'w')
		thefile.write("%s\n" % self.dev_ccc)
		thefile.close
		print 'ccc_dev=',ccc_dev
		if self.ccc_prev<ccc_dev:
			print 'CCC improved=',self.ccc_prev,' to ',ccc_dev
			self.model.save_weights('/home/AN35190/checkpoint/'+self.file+'.hdf5')
			self.ccc_prev=ccc_dev
		
		file_list_train =os.listdir(path_image_train)
		file_list_train.sort()
		label_list_train=os.listdir(path_labels_train)
		label_list_train.sort()
		self.pair_train=[]
		list_train=[]
		mi_tensor_x_train=[]
		mi_tensor_y_train=[]
		for i in range(0,len(file_list_train)):
			frames_train=os.listdir(path_image_train+file_list_train[i])
			frames_train.sort()
			label_train=pd.read_csv(path_labels_train+'/'+file_list_train[i]+'/'+'dropeada_50_'+dim+'_'+file_list_train[i]+'.csv')
			self.number_train=label.size
			for j in range(0,len(frames_train)-1):
				X_train=Image.open(path_image_train+'/'+file_list_train[i]+'/'+frames_train[j])
				Y_train=(label_train.iloc[j,0])
				X_train=np.asarray(X_train,dtype=np.float32).reshape(48,48,1)
				X_train=X_train/255
				#X_train=(X_train-np.mean(X_train))/(np.std(X_train))
				mi_tensor_x_train.append(X_train)
				mi_tensor_y_train.append(Y_train)


		val_pred_train=np.reshape(self.model.predict(np.asarray(mi_tensor_x_train)),-1)
		val_gt_train  =np.asarray(mi_tensor_y_train)
		print 'MI TENSOR SDHAPE CCC train',val_pred_train.shape,val_gt_train.shape
		#print type (val_pred_train)
		#print type (val_gt_train)
		np.save('PRED_TRA_'+self.file,val_pred_train)
		np.save('LABE_TRA_'+self.file,val_gt_train)


		ccc_train=calc_scores(val_gt_train,val_pred_train)
		self.train_ccc.append(ccc_train)
		thefile = open('train_ccc_'+self.file+'.txt', 'w')
		thefile.write("%s\n" % self.train_ccc)
		thefile.close

