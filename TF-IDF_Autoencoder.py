# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:17:41 2019

@author: quartz
"""

from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')
categories = list(newsgroups_train.target_names)
raw_data = fetch_20newsgroups(subset = "train", categories = categories, shuffle = True, random_state = 42)

label = raw_data.target
label_name = [raw_data.target_names[i] for i in raw_data.target]
print(len(raw_data.data))
print(label[:21])
print(label_name[:21])
print(raw_data.data[0])
print(raw_data.filenames[0])
print(raw_data.target_names[raw_data.target[0]])

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
count_data = count_vect.fit_transform(raw_data.data)
print(count_data.shape)


