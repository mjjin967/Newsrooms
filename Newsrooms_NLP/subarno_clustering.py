#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
import pandas as pd


# In[17]:


#!pip install nlp

get_ipython().system('git clone https://github.com/fastai/courses.git')
#import nlp


# In[24]:



import os
import sqlite3


scalar = MinMaxScaler()
      
kmeans = KMeans(n_clusters=64, random_state=0)
#x = Scalar.fit_transform(x)
pca = SparsePCA(max_iter=10)
#pca = KernelPCA(kernel='sigmoid')#n_components=5,
#labels = np.linspace(labels)

kmeans.fit_predict(data)
a = SpectralClustering(n_clusters=64,assign_labels="discretize",random_state=0).fit_predict(data)


  
  
#n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_clusters_ = len(set(a)) - (1 if -1 in a else 0)
some_labels = kmeans.labels_



def classify_documents(topics, labels):


	pass


def cluster_documents(topic, labels,data_dir):

	pass



def main(data_dir):
	all_articles = load_articles(data_dir) # list of tuples [title, article]
	labels = load_labels(data_dir) # hashmap of key: unique doc id, val: title



# Run using 'python nlp.py' or 'python nlp.py <PATH_TO_BBC_DIRECTORY>'
# to manually specify the path to the data.
# This may take a little bit of time (~30-60 seconds) to run.
if __name__ == '__main__':
	# data_dir = '/course/cs1951a/pub/nlp/bbc/data' if len(sys.argv) == 1 else sys.argv[1]
	data_dir = os.path.join(os.getcwd(), 'articles_db.db')
	main(data_dir)


# In[ ]:


import keras
from keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
import numpy as np
#from numpy import array
from keras.models import load_model, Model, model_from_json, Sequential
from keras.callbacks import EarlyStopping
#next steps to implement this deep learning layer if I cannot implement it in time:


#1. assign the index of each of the clusters as the y_train and make sure it corresponds to each vector in x_train
#2. use x_test which will be the the new incoming articles to x_test
#3. create a schedule function which will have the clustering fire every 6 hours or day or whatever time interval you want 
#and use the 2nd schedule function to have what is in this cell fire every half hour or hour. Then you are done! The prediction variable in this cell will tell you which of the existing clusters it should go to! 

#labels = np.array(labels)
#x = tf.cast(x, tf.float64)
#labels = tf.cast(labels, tf.float64)
#x = np.squeeze(x)
#labnels = np.squeeze(labels)
#x = pd.DataFrame(x)
#X_train = data
xShape = labels_.shape()
xShape
#theoretically this should return the indexes of the clusters 
y_train = labels_[1][0]
#what are x_train y_train x_test going to be
#x_train are the news articles
#y_train are the clusters
#x_test are the new news articles
inputs = Input(shape=(xShape,))
first = LSTM(8, activation='relu')(inputs)
A = Dense(250, activation='relu')(first)
x = Dense(500, activation='relu')(A)
#this batch normalization is to make sure that overfitting does not occur
x = BatchNormalization()(x)
x = Dense(10,activation='relu')(x)

#sigmoid is the best optimizer to use for classification tasks supposedly
outputs = Dense(1, activation='sigmoid')(x)

model.compile(loss= 'categorical_crossentropy',optimizer='adam')

model.fit(X_train,Y,epochs=1200, verbose=1,validation_split=0.2)

Prediction = model.predict(x_test)


