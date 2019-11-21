# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:23:33 2019

@author: quartz
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from keras.models import load_model


def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print("tf-idf with", str(np.array(X_train).shape[1]), "features")
    return (X_train, X_test)
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
X_train = newsgroups_train.data
X_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target

X_train,X_test = TFIDF(X_train,X_test)


#PCA

#from sklearn.decomposition import PCA
#pca = PCA(n_components=2000)
#X_train_new = pca.fit_transform(X_train)
#X_test_new = pca.transform(X_test)
#print("train with old features: ",np.array(X_train).shape)
#print("train with new features:" ,np.array(X_train_new).shape)
#print("test with old features: ",np.array(X_test).shape)
#print("test with new features:" ,np.array(X_test_new).shape)

#Autoencoder

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
#from tensorflow import keras
#from tensorflow.python.keras import backend as k
#from tensorflow.python.framework import ops
#ops.reset_default_graph()
# this is the size of our encoded representations
encoding_dim = 1500
n = 10000
# this is our input placeholder
input = Input(shape=(n,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(n, activation='sigmoid')(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input, decoded)
# this model maps an input to its encoded representation
encoder = Model(input, encoded)
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')



autoencoder.fit(X_train, X_train,
                epochs=30,
                batch_size=8,
                shuffle=True,
                validation_data=(X_test, X_test))


#input = Input(shape=(n,))
#autoenc = autoencoder(input)

autoencoder = autoencoder.save('autoencoder.h5')
autoencoder = load_model('autoencoder.h5')



#import scipy.io as sio
#sio.savemat('autoenc.mat', {'autoenc':autoenc})