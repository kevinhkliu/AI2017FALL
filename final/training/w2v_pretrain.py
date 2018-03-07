# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 10:03:09 2017

@author: ChunJi
"""

import numpy as np

comment = np.load('data/post_comments.npy')

token = []

for i in range(0,len(comment)):
    for j in range(0,len(comment[i])):
        token.append(comment[i][j].split())



import gensim
for s in range(100,301,50):
    print(s)
    model = gensim.models.Word2Vec(token, size = s, window= 5, min_count=1,sg=1) #skip-gram     sg=0 CBOW
    model.save('./w2vfile/word2vec_'+ str(s) +'.bin')
