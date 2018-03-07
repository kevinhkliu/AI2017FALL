# -*- coding: utf-8 -*-
from gensim.summarization import summarize
from gensim.summarization import textcleaner
import pandas as pd
import numpy as np 
'''
posts = pd.read_csv("data/FB_post(obama).csv")
#comments = pd.read_csv("data/FB_comments(obama).csv")
post_id = posts['post_ID'].tolist()
posts = posts['post_context'].tolist()
'''

post_context_list = np.load('data/post_context.npy')
post_context_list = post_context_list.tolist()
post_comments_list = np.load('data/post_comments.npy')



post_context=[]
idx = 0
for post in post_context_list:
    post = post.replace('--','')
    sentences = textcleaner.clean_text_by_sentences(post)
    words = post.split()
    print(idx)
    if len(words) > 50 and len(sentences) >= 2 :
        postN = summarize(post, word_count=50)
        if len(postN) != 0:
            post_context.append(postN.replace('\n','').replace('\t','').replace('\r', ''))
        else:
            post_context.append(post.replace('\n','').replace('\t','').replace('\r', ''))
    else:
        post_context.append(post.replace('\n','').replace('\t','').replace('\r', ''))
    idx = idx + 1
posts_summarize = []   
comments_summarize=[]
comments=[]
posts_context=[]
for idx in range(len(post_comments_list)):
    if len(post_comments_list[idx]) == 200:
        posts_summarize.append(post_context[idx])
        for comment in post_comments_list[idx]:
            sentences = textcleaner.clean_text_by_sentences(comment)
            words = comment.split()
            if len(words) > 20 and len(sentences) >= 2:
                commentN = summarize(comment, word_count=20)
                if len(commentN) != 0:
                    comments.append(commentN.replace('\n','').replace('\t','').replace('\r', ''))
                else:
                    comments.append(comment.replace('\n','').replace('\t','').replace('\r', ''))
            else:
                comments.append(comment.replace('\n','').replace('\t','').replace('\r', ''))
        comments_summarize.append(comments)
        comments = []
print(len(comments_summarize))
print(len(posts_summarize))

np.save('data/posts_summarize', posts_summarize)
np.save('data/comments_summarize', comments_summarize)



