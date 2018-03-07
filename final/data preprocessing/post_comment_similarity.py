# -*- coding: utf-8 -*-
import numpy as np 
posts_summarize = np.load('data/posts_summarize.npy')
comments_summarize = np.load('data/comments_summarize.npy')
from sklearn.metrics.pairwise import cosine_similarity
import gensim

print("Loading the Glove word2vec model...")
w2vModel = gensim.models.KeyedVectors.load_word2vec_format('w2vfile/glove.6B.300d.word2vec')
print("Loading the Glove word2vec model Done...")

def avg_sentence_vector(words, w2vModel, num_features):
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word in w2vModel.wv.vocab:
            nwords = nwords+1
            featureVec = np.add(featureVec, w2vModel.wv[word])

    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec.reshape(1, -1)


score = []
post = []
comment = []
for idx in range(len(posts_summarize)):
    print(str(idx) + '/' + str(len(posts_summarize)))
    sentence_1 = posts_summarize[idx]
    sentence_1_avg_vector = avg_sentence_vector(sentence_1.split(), w2vModel, num_features=300)
    maxScore = 0
    for idx2 in range(len(comments_summarize[idx])):
        sentence_2 = comments_summarize[idx][idx2]
        sentence_2_avg_vector = avg_sentence_vector(sentence_2.split(), w2vModel, num_features=300)
        sen1_sen2_similarity =  cosine_similarity(sentence_1_avg_vector,sentence_2_avg_vector)
        if sen1_sen2_similarity[0][0] > maxScore:
            maxScore = sen1_sen2_similarity[0][0]
            commentStr = sentence_2
    post.append(sentence_1)
    comment.append(commentStr)
    score.append(maxScore)

np.save('data/post.npy', post)
np.save('data/comment.npy', comment)
np.save('data/score.npy', score)