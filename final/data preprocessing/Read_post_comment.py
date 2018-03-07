# -*- coding: utf-8 -*-
import numpy as np

post = np.load('data/post.npy')
comment = np.load('data/comment.npy')
score = np.load('data/score.npy')



with open("result/post_comment.txt",'w', encoding='UTF-8') as f:
    f.write('total post_comment: ' + str(len(post)) + '\n')
    for idx in range(len(post)):
        f.write('post_comment: ' + str(idx) + '\n')
        line = 'post: ' + post[idx] + '\n' + 'comment: ' + comment[idx] + '\n' + "cos score: " + str(score[idx])
        f.write(line + '\n')
        f.writelines("=========================================" + '\n')
        
f.close()
    
post_train = post[0:1200]
post_val = post[1200:]
comment_train = comment[0:1200]
comment_val = comment[1200:]

print("training data:" + str(len(post_train)))
print("testing data:" + str(len(post_val)))

train_file = open("train/train.txt", "w",encoding="utf-8")
for idx in range(len(post_train)):
    line = post_train[idx] + "\t" + comment_train[idx] + "\n"
    train_file.write(line)
train_file.close()

text_file = open("train/test.txt", "w",encoding="utf-8")
for idx in range(len(post_val)):
    line = post_val[idx] + "\t" + comment_val[idx] + "\n"
    text_file.write(line)
text_file.close()