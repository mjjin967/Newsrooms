#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import numpy as np
import operator
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction import text 
from sklearn.utils import shuffle


# In[3]:


from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

# import pandas, numpy, string
# from keras.preprocessing import text, sequence
# from keras import layers, models, optimizers


# In[4]:


my_additional_stop_words = []
# my_additional_stop_words = ['el', 'san', 'th', 'los', 'ins', 'll', 'st', 'mr', 'ms', 've', 'don', 'dr']

stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)


# In[4]:


data_dir = "/path_to_your_data"


# In[5]:


df = pd.read_csv("%s/your_data_file.tsv" % (data_dir), sep="\t")


# In[6]:


df.shape


# In[7]:


df['title'][0]


# In[8]:


df.section.value_counts()


# In[9]:


## What to do with imbalanced data? 


# In[10]:


sentences = df['title'].values
y = df['section'].values


# In[11]:


sentences_train, sentences_test, y_train_str, y_test_str = train_test_split(sentences, y, test_size=0.20, random_state=1000)


# In[12]:


tmp = pd.DataFrame({'y':y_train_str})
tmp['y'].value_counts()


# In[13]:


tmp = pd.DataFrame({'y':y_test_str})
tmp['y'].value_counts()


# In[14]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)
X_train


# In[15]:


# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(sentences_train)
X_train_tfidf =  tfidf_vect.transform(sentences_train)
X_test_tfidf =  tfidf_vect.transform(sentences_test)
X_train_tfidf

# # ngram level tf-idf 
# tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
# tfidf_vect_ngram.fit(sentences_train)
# X_train_tfidf_ngram =  tfidf_vect_ngram.transform(sentences_train)
# X_test_tfidf_ngram =  tfidf_vect_ngram.transform(sentences_test)

# # characters level tf-idf
# tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
# tfidf_vect_ngram_chars.fit(sentences_train)
# X_train_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(sentences_train) 
# X_test_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(sentences_test)


# In[16]:


# label encode the target variable 
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train_str)
y_test = encoder.fit_transform(y_test_str)


# In[17]:


tmp = pd.DataFrame({'y':y_test})
tmp['y'].value_counts()


# In[18]:


y_test.shape


# In[19]:


def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax( axis=-1)
    
    return classifier, metrics.accuracy_score(predictions, valid_y)


# ## For balanced dataset 

# In[22]:


# Naive Bayes on Count Vectors
nb, accuracy = train_model(naive_bayes.MultinomialNB(), X_train, y_train, X_test, y_test)
print("NB, Count Vectors: ", accuracy)

# Naive Bayes on TF-IDF Vectors
nb_tfidf, accuracy = train_model(naive_bayes.MultinomialNB(), X_train_tfidf, y_train, X_test_tfidf, y_test)
print("NB, TF-IDF Vectors: ", accuracy)

# Linear Classifier on Count Vectors
lr, accuracy = train_model(linear_model.LogisticRegression(), X_train, y_train, X_test, y_test)
print ("LR, Count Vectors: ", accuracy)

# Linear Classifier on TF-IDF Vectors
lr_tfidf, accuracy = train_model(linear_model.LogisticRegression(), X_train_tfidf, y_train, X_test_tfidf, y_test)
print ("LR, TF-IDF Vectors: ", accuracy)

# RF on Count Vectors
rf, accuracy = train_model(ensemble.RandomForestClassifier(), X_train, y_train, X_test, y_test)
print("RF, Count Vectors: ", accuracy)

# RF on TF-IDF Vectors
rf_tfidf, accuracy = train_model(ensemble.RandomForestClassifier(), X_train_tfidf, y_train, X_test_tfidf, y_test)
print("RF, TF-IDF Vectors: ", accuracy)


# ## For unbalanced dataset

# In[20]:


# Naive Bayes on Count Vectors
nb, accuracy = train_model(naive_bayes.ComplementNB(), X_train, y_train, X_test, y_test)
print("NB, Count Vectors: ", accuracy)

# Naive Bayes on TF-IDF Vectors
nb_tfidf, accuracy = train_model(naive_bayes.ComplementNB(), X_train_tfidf, y_train, X_test_tfidf, y_test)
print("NB, TF-IDF Vectors: ", accuracy)

# Linear Classifier on Count Vectors
lr, accuracy = train_model(linear_model.LogisticRegression(class_weight='balanced'), X_train, y_train, X_test, y_test)
print ("LR, Count Vectors: ", accuracy)

# Linear Classifier on TF-IDF Vectors
lr_tfidf, accuracy = train_model(linear_model.LogisticRegression(class_weight='balanced'), X_train_tfidf, y_train, X_test_tfidf, y_test)
print ("LR, TF-IDF Vectors: ", accuracy)

# RF on Count Vectors
rf, accuracy = train_model(ensemble.RandomForestClassifier(class_weight='balanced'), X_train, y_train, X_test, y_test)
print("RF, Count Vectors: ", accuracy)

# RF on TF-IDF Vectors
rf_tfidf, accuracy = train_model(ensemble.RandomForestClassifier(class_weight='balanced'), X_train_tfidf, y_train, X_test_tfidf, y_test)
print("RF, TF-IDF Vectors: ", accuracy)


# In[ ]:





# In[ ]:





# In[53]:


# Naive Bayes on Count Vectors
nb, accuracy = train_model(naive_bayes.MultinomialNB(), X_train, y_train, X_test, y_test)
print("NB, Count Vectors: ", accuracy)


# In[54]:


# Naive Bayes on Count Vectors
nb_tfidf, accuracy = train_model(naive_bayes.MultinomialNB(), X_train_tfidf, y_train, X_test_tfidf, y_test)
print("NB, Count Vectors: ", accuracy)


# In[55]:


# Linear Classifier on Count Vectors
lr, accuracy = train_model(linear_model.LogisticRegression(), X_train, y_train, X_test, y_test)
print ("LR, Count Vectors: ", accuracy)


# In[56]:


# Linear Classifier on Count Vectors
lr_tfidf, accuracy = train_model(linear_model.LogisticRegression(), X_train_tfidf, y_train, X_test_tfidf, y_test)
print ("LR, Count Vectors: ", accuracy)


# In[70]:


# Linear Classifier on Count Vectors
lr_multinomial, accuracy = train_model(linear_model.LogisticRegression(multi_class='multinomial',solver ='newton-cg'), X_train, y_train, X_test, y_test)
print ("LR, Count Vectors: ", accuracy)


# In[58]:


# RF on Count Vectors
rf, accuracy = train_model(ensemble.RandomForestClassifier(), X_train, y_train, X_test, y_test)
print("RF, Count Vectors: ", accuracy)


# In[59]:


# RF on Count Vectors
rf_tfidf, accuracy = train_model(ensemble.RandomForestClassifier(), X_train_tfidf, y_train, X_test_tfidf, y_test)
print("RF, Count Vectors: ", accuracy)


# In[ ]:





# ### Best Model

# In[21]:


bestmodel = lr


# ### Classificataion Report

# In[22]:


y_pred = bestmodel.predict(X_test)
y_true = y_test


# In[23]:


target_names = encoder.classes_


# In[24]:


from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=target_names))


# In[ ]:





# In[ ]:





# In[67]:


def predict_news_section(test_article, encoder, vectorizer, bestmodel):
    sections = encoder.classes_
    X_unknown = vectorizer.transform([test_article])
    unknown_predicted = bestmodel.predict(X_unknown)
    result = sections[unknown_predicted[0]]
    return result


# In[68]:


test_business = "Paramount Was Hollywood’s ‘Mountain.’ Now It’s a Molehill"
test_sport = "Grab and Go: How Sticky Gloves Have Changed Football"
test_politics = "In Trump’s Immigration Announcement, a Compromise Snubbed All Around"
test_health = "Study Links Drug Maker Gifts for Doctors to More Overdose Deaths"


# In[69]:


print(predict_news_section(test_business, encoder, vectorizer, bestmodel))
print(predict_news_section(test_sport, encoder, vectorizer, bestmodel))
print(predict_news_section(test_politics, encoder, vectorizer, bestmodel))
print(predict_news_section(test_health, encoder, vectorizer, bestmodel))


# ## Save

# In[72]:


import pickle


# In[73]:


with open('news_section_classification_model.pk', 'wb') as output:
    pickle.dump(bestmodel, output)


# In[74]:


with open('news_section_vectorizer.pk', 'wb') as output:
    pickle.dump(vectorizer, output)


# In[75]:


with open('news_section_encoder.pk', 'wb') as output:
    pickle.dump(encoder, output)




