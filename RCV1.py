# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:40:25 2018

@author: quartz
"""
from sklearn.datasets import fetch_rcv1
from sklearn.feature_selection import chi2
import numpy as np
rcv1 = fetch_rcv1()
# data: The feature matrix is a scipy CSR sparse matrix, with 804414 samples and 47236 features. 
#       Non-zero values contains cosine-normalized, log TF-IDF vectors. 
#       A nearly chronological split is proposed in [1]: 
#       The first 23149 samples are the training set. 
#       The last 781265 samples are the testing set. 
#       This follows the official LYRL2004 chronological split. The array has 0.16% of non zero values:
rcv1.data.shape

#target: The target values are stored in a scipy CSR sparse matrix, 
#        with 804414 samples and 103 categories
#        Each sample has a value of 1 in its categories, and 0 in others.
#        The array has 3.15% of non zero values:
rcv1.target.shape

# sample_id: Each sample can be identified by its ID, ranging (with gaps) from 2286 to 810596:
rcv1.sample_id[:3]

#target_names: The target values are the topics of each sample.
#              Each sample belongs to at least one topic, and to up to 17 topics.
#              There are 103 topics, each represented by a string
#              Their corpus frequencies span five orders of magnitude, 
#              from 5 occurrences for ‘GMIL’, to 381327 for ‘CCAT’:
rcv1.target_names[:10].tolist() 

# hha


train = fetch_rcv1(subset='train')
test = fetch_rcv1(subset='test')

X_train = train.data
X_test = test.data


y_train = train.target
y_test = test.target

X_train.shape
#y_train_dense = y_train.todense()
#x_train_dense = X_train.todense()



from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
chi2_selector = SelectKBest(chi2, k=1000)
X_train_kbest = chi2_selector.fit_transform(X_train, y_train)
X_test_kbest = chi2_selector.transform(X_test)
X_test_kbest.todense().shape
X_train_kbest.todense().shape


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

#
#def getFirstLabel( y_train):
#    y_train_first_label= []
#    for i in range(y_train.toarray().shape[0]):
#        y_train_first_label.append(    y_train.toarray()[i].nonzero()[0][0])
#    return y_train_first_label
    
def batch_prediction(data,clf=clf,size=128):
    # data,condition = x_train_dense,  y_train_dense[:,i]==1
    results= []
    for i in range(int(data.shape[0]/128)+1):
        start = i*   size
        end =  (i+1)*   size
        sub_samples = data[start:end] 
        results.append(clf.predict(sub_samples))
    return np.concatenate(np.array(results))



clf.fit(X_train_kbest.toarray(), getFirstLabel(y_train))

predictions = batch_prediction(X_test_kbest.toarray(),clf)
#predictions = clf.predict(X_test_kbest.toarray())

def firstLabelFast(data):
   
    rows,cols = data.toarray().nonzero()
    
    results=[]
    pointer =0 
    for i,j in zip(rows,cols):
        if pointer == i:
            results.append(j)
            pointer= pointer +1
    return results
#
#def getAccuracy(testSet, predictions):
#	correct = 0
#	for x in range(len(testSet)):
#		if testSet[x] == predictions[x]:
#			correct += 1
#	return (correct/float(len(testSet))) * 100.0
#
#acc=getAccuracy(firstLabelFast(y_test),predictions)
#print(acc)

print(sum(firstLabelFast(y_test) == predictions) *1.0 / len(predictions))

    
#import pandas as pd

#df= pd.DataFrame({"cols":cols, "rows":rows})

#sum(result == predictions) *1.0 / len(predictions)

#result = df.groupby("rows").apply(lambda group:group.reset_index()["cols"][0] )
# inport random
#result = df.groupby("rows").apply(lambda group:group.reset_index()["cols"][random.randint(0,len(group)-1)] )
#from keras.models import Sequential
#from keras.layers import Dense, Activation, Embedding
#from keras.layers import LSTM


#model = Sequential()
#model.add(Embedding(input_dim=(None,47236),output_dim=300,dropout=0.25))
#model.add(LSTM(350, dropout_W=0.4, dropout_U=0.4))  
#model.add(Dense(103))
#model.add(Activation('sigmoid'))


#model.compile(loss='binary_crossentropy',
#          optimizer='adam',
#          metrics=['accuracy'])
#
#print('Train...')
#model.fit(X_train, y_train, batch_size=32, nb_epoch=15,
#      validation_data=(X_test, y_test))
#score, acc = model.evaluate(X_test, y_test,
#                        batch_size=32)
#print('Test score:', score)
#print('Test accuracy:', acc)



    
