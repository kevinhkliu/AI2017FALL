# -*- coding: utf-8 -*-
import requests
import pandas as pd 
from dateutil.parser import parse
import numpy as np 
token = 'EAACEdEose0cBAKfIbVwnfpZCSngC5pOzFhM16dRB7vCWLYT52tbHeJmu1d2ZA48pGFTF7kQUe17yay3i90ZBUEvGnHyjoEh3ecsostVSZBP0ZA6vR0UF2Qg8gSAHPJ7VWbYOQT4sUnL1Yqc4Hpr7UYGEN7wOgQbqZB6z4zksZArS6yXEZAyZBqQeItqQcziqENY4bPd4HMz9aRgZDZD' 

fanpage = {'6815841748':'Barack Obama'} 
posts = 0
information_list = []
post_ID_list = []
post_context_list = []
for ele in fanpage:
    res = requests.get('https://graph.facebook.com/v2.9/{}/posts?limit=100&access_token={}'.format(ele, token))
    count_message=0
    while 'paging' in res.json(): 
        for information in res.json()['data']:
            if 'message' in information:
                if information.get('id') is not None and parse(information.get('created_time')) is not None and information['message'] is not None:
                    message = information['message'].replace('\t','').replace('\n','').replace('\r', '')
                    words = message.split()
                    if len(words) <= 100:
                        information_list.append([information.get('id'),message.strip(' \t\n\r')])
                        post_ID_list.append(information.get('id'))
                        post_context_list.append(message.strip(' \t\n\r'))
                        posts = posts + 1
                        print(posts)
                    if posts == 2500:
                        break;
        if posts == 2500:
            break;
        if 'next' in res.json()['paging']:
            res = requests.get(res.json()['paging']['next'])
        else:
            break
        
        
   
np.save('data/post_ID_list.npy', post_ID_list)
np.save('data/post_context.npy', post_context_list)
     

if len(information_list) != 0:   
    information_df = pd.DataFrame(information_list, columns=['post_ID', 'post_context']) 
information_df.to_csv('data/FB_post(obama).csv', encoding='UTF-8', index=False) 
# -*- coding: utf-8 -*-
import requests
import pandas as pd 
from dateutil.parser import parse
import numpy as np 
token = 'EAACEdEose0cBAKfIbVwnfpZCSngC5pOzFhM16dRB7vCWLYT52tbHeJmu1d2ZA48pGFTF7kQUe17yay3i90ZBUEvGnHyjoEh3ecsostVSZBP0ZA6vR0UF2Qg8gSAHPJ7VWbYOQT4sUnL1Yqc4Hpr7UYGEN7wOgQbqZB6z4zksZArS6yXEZAyZBqQeItqQcziqENY4bPd4HMz9aRgZDZD' 

fanpage = {'6815841748':'Barack Obama'} 
posts = 0
information_list = []
post_ID_list = []
post_context_list = []
for ele in fanpage:
    res = requests.get('https://graph.facebook.com/v2.9/{}/posts?limit=100&access_token={}'.format(ele, token))
    count_message=0
    while 'paging' in res.json(): 
        for information in res.json()['data']:
            if 'message' in information:
                if information.get('id') is not None and parse(information.get('created_time')) is not None and information['message'] is not None:
                    message = information['message'].replace('\t','').replace('\n','').replace('\r', '')
                    words = message.split()
                    if len(words) <= 100:
                        information_list.append([information.get('id'),message.strip(' \t\n\r')])
                        post_ID_list.append(information.get('id'))
                        post_context_list.append(message.strip(' \t\n\r'))
                        posts = posts + 1
                        print(posts)
                    if posts == 2500:
                        break;
        if posts == 2500:
            break;
        if 'next' in res.json()['paging']:
            res = requests.get(res.json()['paging']['next'])
        else:
            break
        
        
   
np.save('data/post_ID_list.npy', post_ID_list)
np.save('data/post_context.npy', post_context_list)
     

if len(information_list) != 0:   
    information_df = pd.DataFrame(information_list, columns=['post_ID', 'post_context']) 
information_df.to_csv('data/FB_post(obama).csv', encoding='UTF-8', index=False) 
