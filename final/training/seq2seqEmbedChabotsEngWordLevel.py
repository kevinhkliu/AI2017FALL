# -*- coding: utf-8 -*-
import _pickle as pk
import tensorflow as tf
import nltk
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, merge, Flatten, Reshape, Dropout
import numpy as np
from keras.preprocessing import sequence
import gensim

np.random.seed(1234)  # for reproducibility

def tokenize(sentence):    
    sentence_tokens = []
    for word in nltk.word_tokenize(sentence):
        sentence_tokens.append(word)
    return sentence_tokens


latent_dim = 300  # Latent dimensionality of the encoding space.
embedding_size = 300 #100
num_samples = 10000  # Number of samples to train on.
data_path = 'data/train.txt'
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
vocab_characters = set()

'''====================read train data========================================================================================'''
lines = open(data_path,encoding="utf-8").read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text_str, target_text_str = line.split('\t')
    input_text_tokens = tokenize(input_text_str)
    target_text_tokens = tokenize(target_text_str)
    input_text = input_text_tokens
    target_text = ['BOS'] + target_text_tokens + ['EOS']
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
            # update for word vocabulary
        if char not in vocab_characters:
            vocab_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
        if char not in vocab_characters:
            vocab_characters.add(char)
            
vocab_characters = sorted(list(vocab_characters), reverse=True)
vocab_size = len(vocab_characters)
vocab_size = vocab_size + 1
num_encoder_tokens = vocab_size
num_decoder_tokens = vocab_size

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

maxlen_input = max(max_encoder_seq_length, max_decoder_seq_length)
max_encoder_seq_length, max_decoder_seq_length = maxlen_input, maxlen_input
maxlen_output = maxlen_input

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Vocab zise (num of tokens):', vocab_size)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

'''====================build dictionary========================================================================================'''
vocab_token_index = dict(
    [(char, i+1) for i, char in enumerate(vocab_characters)])
print(vocab_token_index['BOS'], vocab_token_index['EOS'])
pk.dump(vocab_token_index, open("data/vocab_token_index", 'wb'))
input_token_index = vocab_token_index
target_token_index = vocab_token_index
print(target_token_index['BOS'], target_token_index['EOS'])

'''====================load pretrain embedding matrix========================================================================================'''
def w2v_embedding(vocab_token_index, vocab_size, embedding_size):
 
    print("load gensim word2vector")
    w2vModel = gensim.models.KeyedVectors.load_word2vec_format('w2vfile/glove.6B.300d.word2vec')
    
    embedding_matrix = np.zeros((vocab_size, embedding_size))
    for word, i in vocab_token_index.items():
        if i > vocab_size:
            continue
        if word in w2vModel.wv.vocab:
            embedding_matrix[i] = w2vModel.wv[word]
    print("save embedding_matrix")
    np.save('data/embedding_matrix', embedding_matrix)
    
#w2v_embedding(vocab_token_index, vocab_size, embedding_size)
embedding_matrix = np.load('data/embedding_matrix.npy')
print("load embedding_matrix done")

'''====================Creating the training data========================================================================================'''
unknown_token = 'NONE'

for i, sent in enumerate(input_texts):
    input_texts[i] = [w if w in vocab_token_index else unknown_token for w in sent]
    
for i, sent in enumerate(target_texts):
    target_texts[i] = [w if w in vocab_token_index else unknown_token for w in sent]

X = np.asarray([[vocab_token_index[w] for w in sent] for sent in input_texts])
Y = np.asarray([[vocab_token_index[w] for w in sent] for sent in  target_texts])

encoder_input_data = sequence.pad_sequences(X, maxlen=maxlen_input)
decoder_input_data = sequence.pad_sequences(Y, maxlen=maxlen_output, padding='post')

'''====================build model=============================================================================================================='''


encoder_inputs = Input(shape=(max_encoder_seq_length,),dtype='int32', name='encoder_input')

encoder_embedding = Embedding(input_dim=num_encoder_tokens, output_dim=embedding_size,
                        weights=[embedding_matrix], input_length=max_encoder_seq_length,
                        trainable=False)(encoder_inputs)

encoder_embedding_LSTM = LSTM(latent_dim)(encoder_embedding)
encoder_embedding_LSTM = Dropout(0.3)(encoder_embedding_LSTM)

decoder_inputs = Input(shape=(max_decoder_seq_length,), dtype='int32', name='decorder_input')
decoder_embedding = Embedding(input_dim=num_decoder_tokens, output_dim=embedding_size,
                        weights=[embedding_matrix], input_length=max_decoder_seq_length,
                        trainable=False)(decoder_inputs)

decoder_embedding_LSTM = LSTM(latent_dim)(decoder_embedding)
decoder_embedding_LSTM = Dropout(0.3)(decoder_embedding_LSTM)



'''
encoder_inputs = Input(shape=(max_encoder_seq_length,),dtype='int32', name='encoder_input')
enc_wordembedding = Embedding(input_dim=num_encoder_tokens, output_dim=embedding_size, weights=[embedding_matrix],input_length=max_encoder_seq_length)
encoder_embedding = enc_wordembedding(encoder_inputs)
LSTM_encoder = LSTM(latent_dim, init='lecun_uniform')
encoder_embedding_LSTM = LSTM_encoder(encoder_embedding)

decoder_inputs = Input(shape=(max_decoder_seq_length,), dtype='int32', name='decorder_input')
dec_wordembedding = Embedding(input_dim=num_encoder_tokens, output_dim=embedding_size, weights=[embedding_matrix],input_length=max_encoder_seq_length)
decoder_embedding = dec_wordembedding(decoder_inputs)
LSTM_decoder = LSTM(latent_dim, init='lecun_uniform')
decoder_embedding_LSTM = LSTM_decoder(decoder_embedding)
'''
'''
merge_layer = merge([encoder_embedding_LSTM, decoder_embedding_LSTM], mode='concat', concat_axis=1)
dense_outputs = Dense(int(num_decoder_tokens/2), activation='relu')(merge_layer)
dense_layer = Dense(num_decoder_tokens, activation='softmax')
outputs = dense_layer(dense_outputs)
'''
attn = merge([encoder_embedding_LSTM, decoder_embedding_LSTM], mode='dot', dot_axes=[1, 1])
#attn = Flatten()(attn)
attn = Dense(latent_dim)(attn)

encoder_attn = merge([encoder_embedding_LSTM, attn], mode = 'sum')
#encoder_attn = Flatten()(encoder_attn)
dense_outputs = Dense(int(num_decoder_tokens/2), activation='relu')(encoder_attn)
dense_layer = Dense(num_decoder_tokens, activation='softmax')
outputs = dense_layer(dense_outputs)


model = Model([encoder_inputs, decoder_inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='categorical_crossentropy')

'''====================decode_sequence=============================================================================================================='''
reverse_vocab_char_index = dict(
    (i, char) for char, i in vocab_token_index.items())
reverse_input_char_index = reverse_vocab_char_index
reverse_target_char_index = reverse_vocab_char_index

def decode_sequence(input_seq):
    # Encode the input as state vectors. 
    model.load_weights('model/model.h5')
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, max_decoder_seq_length))
    # Populate the first character of target sequence with the start character.
    target_seq[0, -1] = target_token_index['BOS']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    i= 0
    while not stop_condition:
        prediction = model.predict([input_seq, target_seq])
        # Sample a token
        pred_probs = prediction[0,:]
        prob = np.max(pred_probs)
        sampled_token_index = np.argmax(prediction)
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char + ' '

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == 'EOS' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        # Update the target sequence (of length 1).
        # target_seq = np.zeros((1, max_decoder_seq_length))
        target_seq[0, 0:-1] = target_seq[0, 1:]
        target_seq[0, -1] =  sampled_token_index
    return decoded_sentence

q = encoder_input_data
a = decoder_input_data
n_test = 100
num_subsets = 1
n_exem = len(input_texts)
dictionary_size = vocab_size


def print_result(input):
    ans_partial = np.zeros((1,maxlen_input))
    ans_partial[0, -1] = vocab_token_index['BOS']  #  the index of the symbol BOS (begin of sentence)
    for k in range(maxlen_input - 1):
        ye = model.predict([input, ans_partial])
        mp = np.argmax(ye)
        ans_partial[0, 0:-1] = ans_partial[0, 1:]
        ans_partial[0, -1] = mp
    text = ''
    for k in ans_partial[0]:
        k = k.astype(int)
        if k < (dictionary_size-2):
            w = reverse_vocab_char_index[k]
            #text = text + w[0] + ' '
            text = text + w + ' '
    return(text)

qt = q[0:n_test,:]
at = a[0:n_test,:]
q = q[n_test + 1:,:]
a = a[n_test + 1:,:]

Epochs = 100
BatchSize = 128

print('Number of exemples = %d'%(n_exem - n_test))
step = int(np.around((n_exem - n_test)/num_subsets))
print(step)
round_exem = step * num_subsets
print(round_exem)

'''===================Bot training=============================================================================================================='''
x = range(0,Epochs) 
valid_loss = np.zeros(Epochs)
train_loss = np.zeros(Epochs)
for m in range(Epochs):
    # Loop over training batches due to memory constraints:
    for n in range(0,round_exem,step):
        
        q2 = q[n:n+step]
        s = q2.shape
        count = 0
        for i, sent in enumerate(a[n:n+step]):
            l = np.where(sent==vocab_token_index['EOS'])  #  the position od the symbol EOS
            limit = l[0][0]
            count += limit + 1
            
        Q = np.zeros((count,maxlen_input))
        A = np.zeros((count,maxlen_input))
        Y = np.zeros((count,dictionary_size))
        
        # Loop over the training examples:
        count = 0
        for i, sent in enumerate(a[n:n+step]):
            ans_partial = np.zeros((1,maxlen_input))
            
            # Loop over the positions of the current target output (the current output sequence):
            l = np.where(sent==vocab_token_index['EOS'])  #  the position of the symbol EOS
            limit = l[0][0]

            for k in range(1,limit+1):
                # Mapping the target output (the next output word) for one-hot codding:
                y = np.zeros((1, dictionary_size))
                y[0, sent[k]] = 1

                # preparing the partial answer to input:

                ans_partial[0,-k:] = sent[0:k]

                # training the model for one epoch using teacher forcing:
                
                Q[count, :] = q2[i:i+1] 
                A[count, :] = ans_partial 
                Y[count, :] = y
                count += 1
                
        print('Training epoch: %d, training examples: %d - %d'%(m,n, n + step))
        model.fit([Q, A], Y, batch_size=BatchSize, epochs=1)
         
        test_input = qt[31:32]
        print(print_result(test_input))
        train_input = q[31:32]
        print(print_result(train_input))        
        
    model.save_weights("model/model.h5", overwrite=True)

'''===================Output train result=============================================================================================================='''
for seq_index in range(0,5):    
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)