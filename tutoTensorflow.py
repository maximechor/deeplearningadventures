import tensorflow as tf
import os
import sys
import pdb
from skimage import data
import numpy as np
import matplotlib.pyplot as plt
#pyplot.switch_backend('agg')

#Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

#Multiply
result = tf.multiply(x1,x2)

# Initialze the Session
##### Option 1############
#sess = tf.Session()

print(result)

#print("result:", sess.run(result))

#Close the session
#sess.close()

### Option 2############
with tf.Session() as sess:
  output = sess.run(result)
  print("result:",output)


#/home/AN84020/training/dataset
def load_data(data_directory):
    print("Inside Load_Data")
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory,d))]
    print("directories\n:", directories)
    labels =[]
    images =[]
    pdb.set_trace()
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        #print("label_directory:", label_directory)
        file_names =[os.path.join(label_directory, f)
                     for f in os.listdir(label_directory)
                     if f.endswith(".ppm")]
        #pdb.set_trace()

        for f in file_names:
            #images.append(skimage.data.imread(f))
            images.append(data.imread(f))
            labels.append(int(d))
            #pdb.set_trace()
   
    return images, labels

ROOT_PATH ="/home/AN84020/training/dataset/"
train_data_directory = os.path.join(ROOT_PATH,"BelgiumTSC_Training")
test_data_directory = os.path.join(ROOT_PATH,"BelgiumTSC_Testing")

print("train_data_directory:", train_data_directory)

images, labels = load_data(train_data_directory)

#Print the images dimension
images = np.array(images)
labels = np.array(labels)
print("images.ndim:", images.ndim)
print("number of imgaes elements:", images.size)

print("print first instance of images:", images[0])

print("labels dimen:", labels.ndim)
print("labels size:", labels.size)

print("count the number of labels:", len(set(labels)))

#print("Memory layout of the array:", images.flags)
#print("Length of one array element in bytes.:", images.itemsize)
#print("Total bytes consumed by the elements of the array.:", images.nbytes)

#plt.hist(labels, 62)
#plt.savefig('labelsBelgiumTSC.png')
#plt.show()
plt.plot(range(10))
#plt.savefig('testplot.png')
