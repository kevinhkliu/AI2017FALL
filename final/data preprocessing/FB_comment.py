# -*- coding: utf-8 -*-
import requests
import pandas as pd
from dateutil.parser import parse
import numpy as np 
token = 'EAACEdEose0cBAA16sdURGnxcqvIDfNKTNghYMIu21I6IBoOlwIhWe8sKwoiJbBnN0y8LTjEXXjnYZCVbQNCnZAqY66KgRYIVNhYAZACZAs9j8uR1h5J1cSxSXBXmt3HWC5NuCd3QaAlcYVzevNurldDB3Jz0eXribldi4NZBzfzq6IZAoJy5Hf3Pv81AZBFj7SEOmOEVgPR8QZDZD' 
#your token

fanpage = {'6815841748':'Barack Obama'} 

#fanpage id and name 

idx = 0
comments_list=[]
post_ID_list = []
post_comments_list = []
for ele in fanpage:
    #抓貼文(1篇)
    res = requests.get('https://graph.facebook.com/v2.9/{}/posts?limit=100&access_token={}'.format(ele, token))
    post_index = 0
    while 'paging' in res.json(): 
        for index, information in enumerate(res.json()['data']):
            if 'message' in information:
                if information.get('id') is not None and parse(information.get('created_time')) is not None and information['message'] is not None:
                    post = information['message'].replace('\t','').replace('\n','').replace('\r', '')
                    words = post.split()
                    if len(words) <= 100:       
                        res_comment = requests.get('https://graph.facebook.com/v2.9/{}/comments?&limit=3000&access_token={}'.format(information['id'], token))
                        post_index = post_index + 1
                        print("---------------------posts number:" + str(post_index))
                        commentNum = 0
                        comments_list=[]
                        if res_comment.json().get('data'):
                            for eles in res_comment.json()['data']:
                                comment =  eles['message'].replace('\n','').replace('\t','').replace('\r', '')
                                words = comment.split()
                                if len(words) >= 5 and len(words) <= 30 and commentNum < 200:
                                    comments_list.append(str(comment))
                                    commentNum = commentNum + 1
                                if commentNum == 200:
                                    break
                            print(commentNum)
                        if commentNum == 200:  
                            post_ID_list.append(information.get('id'))  
                            post_comments_list.append(comments_list)
                        
                    if post_index == 2500:
                        break;     
        if post_index == 2500:
            break;                             
        if 'next' in res.json()['paging']:
            res = requests.get(res.json()['paging']['next'])
        else:
            break        
         
np.save('data/post_ID.npy', post_ID_list)
np.save('data/post_comments.npy', post_comments_list)

'''  
information_df = pd.DataFrame(post_comments_list, columns=['post_id', 'comment'])
information_df.to_csv('data/FB_comments(obama).csv', encoding='UTF-8', index=False) 
'''

