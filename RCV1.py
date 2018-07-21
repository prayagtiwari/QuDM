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


train = fetch_rcv1(subset='train')
test = fetch_rcv1(subset='test')

X_train = train.data
X_test = test.data


y_train = train.target
y_test = test.target

X_train.shape
y_train_dense = y_train.todense()
x_train_dense = X_train.todense()

def filter(data,condition,size=128):
    # data,condition = x_train_dense,  y_train_dense[:,i]==1
    results= []
    for i in range(int(data.shape[0]/128)+1):
        start = i*   size
        end =  (i+1)*   size
        subresult = data[start:end] [condition[start:end].getA().squeeze()]
        results.append(subresult.getA())
    return np.concatenate(np.array(results))

def chisqr_featselection(data,subdata_index,topk=100):
    a = sum(subdata_index)[0,0]
    b = data.shape[0] - a
    subfeature = filter(data,subdata_index)
    c = sum(np.sum(subfeature,1) < 1e-5)  # careful
    otherfeature = filter(data,~subdata_index)
    d = sum(np.sum(otherfeature,1) < 1e-5)
    # ....   
    
    return
    


#function fF = chisqr_featselection(F,L,m)
#
#    n = size(F,1);
# 
#    % number of examples with the feature and the category
#    a = sum((F.*L)>0);
#    % number of examples with the feature but not the category    
#    b = sum((F.*(L==0))>0);
#    % number of examples in the category but with no feature
#    c = sum((F.*L)==0);
#    % number of examples not in the category and with no feature
#    d = sum((F.*(L==0))==0);
#      
#%     a(1,18931)
#%     b(1,18931)
#%     c(1,18931)
#%     d(1,18931)
#    
#    w = n.*(a.*d - b.*c)./((a+b).*(a+c).*(b+d).*(c+d));
#   
#    w = fillmissing(w, 'constant',  -Inf);
#   
#    [sortvals, sortidx] = sort(w,2,'descend'); 
#        
#    B = zeros(size(w),class(w)); 
#    for K = 1 : size(w,1) 
#        B(K,sortidx(K,1:m)) = sortvals(K,1:m); 
#    end
#    
#    fF = B;
#            
#end

for i in range(rcv1.target.shape[-1]):
    subfeaters = filter(x_train_dense,  y_train_dense[:,i]==1)
    print(subfeaters.shape)
    chi2(subfeaters)    
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




for i in range(103):
    topic = i
    
    traintopic = y_train
    testtopic = y_test
    
    
