data preprocessing: 

0. fb_posts.py and fb_comment.py are crawler. To produce post_comments.npy and post_context.npy
1. run the TextSummery.py first, produce post and comment summery npy
2. run the post_comment_similarity.npy, produce the post and comment npy 
3. run the Read_post_comment.py, produce the train.txt and test.txt 

======================================
training 

1. run seq2seqEmbedChabotsEngWordLevel.py for training 
2. run seq2seqEmbedChabotsTestEngWordLevel.py for testing 