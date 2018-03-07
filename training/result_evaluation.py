# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 10:24:35 2017

@author: ChunJi
"""
import gensim
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_wordvec(comment,word2vec):
    token = comment.split()
    vec = []    
    for i in range(0,len(token)):
        try:
            vec.append(word2vec[token[i]])
        except:
            continue
    
    avg_vec = np.mean(vec, axis=0)
    return(avg_vec)

def evaluation(predict_comment,ans_list,word2vec_dim):

    word2vec = gensim.models.Word2Vec.load('./w2vfile/word2vec_'+ str(word2vec_dim) +'.bin')
    
    predict_vec = (get_wordvec(predict_comment,word2vec)).reshape(1,-1)   
    
    similarity = []
    for i in range(0,len(ans_list)):
        ans_vec = (get_wordvec(ans_list[i],word2vec)).reshape(1,-1)
        for idx in range(len(cosine_similarity(predict_vec,ans_vec)[0])):
            cosine_similarity(predict_vec,ans_vec)[0][idx] = round(cosine_similarity(predict_vec,ans_vec)[0][idx], 3)
    
        similarity.append(cosine_similarity(predict_vec,ans_vec)[0])

    return(max(similarity))

comment = np.load('data/post_comments.npy')
post = np.load('data/post_context.npy')
post = post.tolist()

num_samples = 100

predict_comment = []
data_path = 'result/testResult.txt' 
lines = open(data_path,encoding="utf-8").read().split('\n')
for i in range(0,len(lines)-1):
    if(lines[i] == 'Decoded sentence:'):
        predict_comment.append(lines[i+1])


input_list = []
target_list = []
predict_list = []
idx_list = []
data_path = 'data/test.txt'
lines = open(data_path,encoding="utf-8").read().split('\n')
i = 0
for line in lines[: min(num_samples, len(lines) - 1)]:
    # update for tokenize string to word tokens
    # input_text, target_text = line.split('\t')
    
    input_text_str, target_text = line.split('\t')
    try:
        idx_list.append(post.index(input_text_str))
    except:
        #idx_list.append(-1)  # 5 post not in post_context
        i += 1
        continue
    input_list.append(input_text_str)
    target_list.append(target_text)
    predict_list.append(predict_comment[i])
    i+=1


#word2vec_dim = 100
for word2vec_dim in range(100,301,50):
    print(str(word2vec_dim))
    simi_predict_and_target = []
    for i in range(0,len(input_list)):
        predict_comment = predict_list[i]
        ans_list = [target_list[i]]
        max_similarity = evaluation(predict_comment,ans_list,word2vec_dim)
        simi_predict_and_target.append(max_similarity[0])
        
        
    info = []
    '''
    for i in range(0,len(simi_predict_and_target)):    
        info.append([input_list[i],target_list[i],predict_list[i],simi_predict_and_target[i]])
        
    information_df = pd.DataFrame(info, columns=['input', 'target','predict','similarity']) 
    information_df.to_csv('evaluation_result_predict_and_target_'+ str(word2vec_dim) +'.csv', encoding='UTF-8', index=False,sep = "\t") 
    '''
    
    fileName = 'result/evaluation_result_predict_and_target_'+ str(word2vec_dim) +'.txt'
    with open(fileName,'w', encoding='UTF-8') as f:
        for idx in range(0,len(simi_predict_and_target)):  
            line = 'post: ' + input_list[idx] + '\n' + 'Answer comment: ' + target_list[idx] + '\n' + 'decoded comment: ' + predict_list[idx] + '\n' + "cos score: " + str(simi_predict_and_target[idx])
            f.write(line + '\n')
            f.writelines("=========================================" + '\n')
     
    simi_predict_and_topN = []
    for i in range(0,len(input_list)):
        predict_comment = predict_list[i]
        ans_list = comment[idx_list[i]]
        max_similarity = evaluation(predict_comment,ans_list,word2vec_dim)
        simi_predict_and_topN.append(max_similarity)
    '''
    info = []
    for i in range(0,len(simi_predict_and_topN)):    
        info.append([input_list[i],target_list[i],predict_list[i],simi_predict_and_topN[i]])
        
    information_df = pd.DataFrame(info, columns=['input', 'target','predict','similarity']) 
    information_df.to_csv('evaluation_result_predict_and_top200_'+ str(word2vec_dim) +'.csv', encoding='UTF-8', index=False,sep = "\t") 
    '''
    fileName = 'result/evaluation_result_predict_and_top200_'+ str(word2vec_dim) +'.txt'
    with open(fileName,'w', encoding='UTF-8') as f:
        for idx in range(0,len(simi_predict_and_target)):  
            line = 'post: ' + input_list[idx] + '\n' + 'Answer comment: ' + target_list[idx] + '\n' + 'decoded comment: ' + predict_list[idx] + '\n' + "cos score: " + str(simi_predict_and_topN[idx])
            f.write(line + '\n')
            f.writelines("=========================================" + '\n')