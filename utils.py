import numpy as np
import pickle
import os
import gzip
import csv
import pandas as pd
import arff
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(os.path.split(__file__)[0], "..", "data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        #print('Downloading data from %s', % origin)
        print("Downloading data from", origin)
        urllib.urlretrieve(origin, dataset)

    print("... loading data")

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    return train_set, valid_set, test_set


def concatenate_list_data(data):
    print("Inside Concatenate_list_data")
    result =[]
    for key, value in data.iteritems():
        result.append([key,value])
    return result


def CCC(y_true_a, y_pred_a,sample_weight=None,multioutput='uniform_average'):
    print("Start CCC")
    y_true = np.squeeze(y_true_a).tolist()
    y_pred = np.squeeze(y_pred_a).tolist()
    print("y_true:",len(y_true))
    print("y_pred:", len(y_pred))
    cor=np.corrcoef(y_true,y_pred)[0][1]
    mean_true=np.mean(y_true)
    mean_pred=np.mean(y_pred)
    var_true=np.var(y_true)
    var_pred=np.var(y_pred)
    sd_true=np.std(y_true)
    sd_pred=np.std(y_pred)
    numerator=2*cor*sd_true*sd_pred
    denominator=var_true+var_pred+(mean_true-mean_pred)**2
    return numerator/denominator


def CreateFileCSV(fileName,content):
    print("Start Creating CSV File")
    with open(fileName,'w') as csvfile:
         filewriter = csv.writer(csvfile, delimiter=',', quoting = csv.QUOTE_ALL)
         filewriter.writerow(content)


test = "score:" + "334"

#CreateFileCSV('testCreation.csv',test)


def CreateFileCSV_Pandas(fileName, col1, val1, col2, val2):
    print("Start Creating CSV File")
    df = pd.DataFrame(data={col1:[val1] })
    df.to_csv(fileName, sep =',', index =True)

#CreateFileCSV_Pandas('testcsvPanda.csv','score', '2223', '', '')


def LoadRecolaVideo():
    '''
     Load Recola Video
    '''
    ##################LOAD VIDEO DATA###############################
    x_train_video= arff.load(open('/data/scratch/Odilon/concatenated/features_video_geometric_features/train_123456789.arff', 'r'))
    x_test_video = arff.load(open('/data/scratch/Odilon/concatenated/features_video_geometric_features/test_123456789.arff', 'r'))
    x_valid_video = arff.load(open('/data/scratch/Odilon/concatenated/features_video_geometric_features/dev_123456789.arff','r'))
    
    x_train_video_keys = list(x_train_video.keys())
    
    #print("x_train_video_keys:\n",x_train_video_keys)
    x_train_video = [row[2:] for row in x_train_video['data']]
    x_train_video = np.array(x_train_video)
    x_test_video = [row[2:] for row in x_test_video['data']]
    x_test_video = np.array(x_test_video)
    x_valid_video = [row[2:] for row in x_valid_video['data']]
    x_valid_video = np.array(x_valid_video)
    
    ##############Normalize Data##################
    x_train_video = np.asarray(x_train_video,'float32')
    x_train_video = x_train_video.astype('float32')/255.
    x_test_video = np.asarray(x_test_video,'float32')
    x_test_video = x_test_video.astype('float32')/255.
    x_valid_video = np.asarray(x_valid_video,'float32')
    x_valid_video = x_valid_video.astype('float32')/255.

    # To be tested???
    #x_train_video = preprocessing.normalize(x_train_video, norm ='l2')
    #x_test_video = preprocessing.normalize(x_test_video, norm ='l2')
    #x_valid_video = preprocessing.normalize(x_valid_video, norm ='l2')

    ##################Reshape Data#################
    x_train_video = x_train_video.reshape((len(x_train_video), np.prod(x_train_video.shape[1:])))
    x_test_video = x_test_video.reshape((len(x_test_video), np.prod(x_test_video.shape[1:])))
    x_valid_video = x_valid_video.reshape((len(x_valid_video), np.prod(x_valid_video.shape[1:])))
    #print("train_video.shape:", x_train_video.shape)
    #print("test_video.shape:", x_test_video.shape)
    #print("valid_video.shape:", x_valid_video.shape)
    
    return x_train_video, x_test_video,x_valid_video

train_video, test_video, valid_video = LoadRecolaVideo()
#print("train_video_test_shape:",train_video.shape)
#print("test_video_test_shape:", test_video.shape)
#print("valid_video_test_shape:", valid_video.shape)


def LoadRecolaVideo_WithoutPreprocessing():
    '''
     Load Recola Video
    '''
    ##################LOAD VIDEO DATA###############################
    x_train_video= arff.load(open('/data/scratch/Odilon/concatenated/features_video_geometric_features/train_123456789.arff', 'r'))
    x_test_video = arff.load(open('/data/scratch/Odilon/concatenated/features_video_geometric_features/test_123456789.arff', 'r'))
    x_valid_video = arff.load(open('/data/scratch/Odilon/concatenated/features_video_geometric_features/dev_123456789.arff','r'))
    
    x_train_video_keys = list(x_train_video.keys())
    
    #print("x_train_video_keys:\n",x_train_video_keys)
    x_train_video = [row[2:] for row in x_train_video['data']]
    x_train_video = np.array(x_train_video)
    x_test_video = [row[2:] for row in x_test_video['data']]
    x_test_video = np.array(x_test_video)
    x_valid_video = [row[2:] for row in x_valid_video['data']]
    x_valid_video = np.array(x_valid_video)
    
    return x_train_video, x_test_video,x_valid_video
    

def LoadRecolaAudio_WithoutPreprocessing():
    ######################## LOAD AUDIO DATA ######################
 
    x_train_audio =arff.load(open('/data/scratch/Odilon/concatenated/features_audio/train_123456789.arff', 'r'))
    x_test_audio =arff.load(open('/data/scratch/Odilon/concatenated/features_audio/test_123456789.arff','r'))
    x_valid_audio=arff.load(open('/data/scratch/Odilon/concatenated/features_audio/dev_123456789.arff','r'))
    
    x_train_audio =[row[2:] for row in x_train_audio['data']]
    x_train_audio = np.array(x_train_audio)
    x_test_audio =[row[2:] for row in x_test_audio['data']]
    x_test_audio = np.array(x_test_audio)
    x_valid_audio = [row[2:] for row in x_valid_audio['data']]
    x_valid_audio = np.array(x_valid_audio) 
   

    return x_train_audio, x_test_audio, x_valid_audio

train_audio, test_audio, valid_audio =  LoadRecolaAudio_WithoutPreprocessing()

#print("train_audio_test.shape:", train_audio.shape)
#print("test_audio_test.shape:", test_audio.shape)
#print("valid_audio_test.shape:",valid_audio.shape)


def LoadRecolaAudio():
    ######################## LOAD AUDIO DATA ######################
 
    x_train_audio =arff.load(open('/data/scratch/Odilon/concatenated/features_audio/train_123456789.arff', 'r'))
    x_test_audio =arff.load(open('/data/scratch/Odilon/concatenated/features_audio/test_123456789.arff','r'))
    x_valid_audio=arff.load(open('/data/scratch/Odilon/concatenated/features_audio/dev_123456789.arff','r'))
    
    x_train_audio =[row[2:] for row in x_train_audio['data']]
    x_train_audio = np.array(x_train_audio)
    x_test_audio =[row[2:] for row in x_test_audio['data']]
    x_test_audio = np.array(x_test_audio)
    x_valid_audio = [row[2:] for row in x_valid_audio['data']]
    x_valid_audio = np.array(x_valid_audio) 
   
    ###Normalize Data#####
    x_train_audio = np.asarray(x_train_audio, 'float32')
    x_train_audio = x_train_audio.astype('float32')/255.
    x_test_audio = np.asarray(x_test_audio, 'float32')
    x_test_audio = x_test_audio.astype('float32')/255.
    x_valid_audio = np.asarray(x_valid_audio, 'float32')
    x_valid_audio = x_valid_audio.astype('float32')/255.
    
    x_train_audio = x_train_audio.reshape((len(x_train_audio), np.prod(x_train_audio.shape[1:])))
    x_test_audio = x_test_audio.reshape((len(x_test_audio), np.prod(x_test_audio.shape[1:])))
    x_valid_audio = x_valid_audio.reshape((len(x_valid_audio), np.prod(x_valid_audio.shape[1:])))
    
    #print("train_audio.shape:", x_train_audio.shape)
    #print("test_audio.shape:", x_test_audio.shape)
    #print("valid_audio.shape:", x_valid_audio.shape)

    return x_train_audio, x_test_audio, x_valid_audio

#train_audio, test_audio, valid_audio =  LoadRecolaAudio()
#print("train_audio_test.shape:", train_audio.shape)
#print("test_audio_test.shape:", test_audio.shape)
#print("valid_audio_test.shape:",valid_audio.shape)



#def autoencoder(dims, act='relu', init='glorot_uniform'):
        
