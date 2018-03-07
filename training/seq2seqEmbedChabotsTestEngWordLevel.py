# -*- coding: utf-8 -*-
import _pickle as pk
import tensorflow as tf
import nltk
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, merge
import numpy as np
from keras.preprocessing import sequence


def tokenize(sentence):    
    sentence_tokens = []
    for word in nltk.word_tokenize(sentence):
        sentence_tokens.append(word)
    return sentence_tokens
'''====================read train data========================================================================================'''
latent_dim = 300  # Latent dimensionality of the encoding space.
embedding_size = 300
num_samples = 10000  # Number of samples to train on.
data_path = 'data/train.txt'

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
vocab_characters = set()

lines = open(data_path,encoding="utf-8").read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text_str, target_text_str = line.split('\t')
    input_text_tokens = tokenize(input_text_str)
    target_text_tokens = tokenize(target_text_str)
    input_text = input_text_tokens
    target_text = ['BOS'] + target_text_tokens + ['EOS']
    #print(target_text)
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
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

'''====================Creating the training data========================================================================================'''
unknown_token = 'NONE'
# Replacing all words not in our vocabulary with the unknown token:

for i, sent in enumerate(input_texts):
    input_texts[i] = [w if w in vocab_token_index else unknown_token for w in sent]
    
for i, sent in enumerate(target_texts):
    target_texts[i] = [w if w in vocab_token_index else unknown_token for w in sent]
   
X = np.asarray([[vocab_token_index[w] for w in sent] for sent in input_texts])
Y = np.asarray([[vocab_token_index[w] for w in sent] for sent in  target_texts])

encoder_input_data = sequence.pad_sequences(X, maxlen=maxlen_input)
decoder_input_data = sequence.pad_sequences(Y, maxlen=maxlen_output, padding='post')

'''====================build model=============================================================================================================='''
embedding_matrix = np.load('data/embedding_matrix.npy')
print("load embedding_matrix done")

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
merge_layer = merge([encoder_embedding_LSTM, decoder_embedding_LSTM], mode='concat', concat_axis=1)
dense_outputs = Dense(int(num_decoder_tokens/2), activation='relu')(merge_layer)
dense_layer = Dense(num_decoder_tokens, activation='softmax')
outputs = dense_layer(dense_outputs)
'''
attn = merge([encoder_embedding_LSTM, decoder_embedding_LSTM], mode='dot', dot_axes=[1, 1])
#attn = Flatten()(attn)
attn = Dense(latent_dim)(attn)

dense_outputs = Dense(int(num_decoder_tokens/2), activation='relu')(attn)
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
'''==================testing=============================================================================================================='''
def test():
    num_samples = 100  # Number of samples to train on.
    inputList=[]
    input_texts=[]
    target_texts=[]
    data_path = 'data/test.txt'
    vocab_token_index['NONE'] = 0
    lines = open(data_path,encoding="utf-8").read().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:
        # update for tokenize string to word tokens
        # input_text, target_text = line.split('\t')
        input_text_str, target_text = line.split('\t')
        input_text_tokens = tokenize(input_text_str)
        input_texts.append(input_text_tokens)
        inputList.append(input_text_tokens)
        target_texts.append(target_text)
    for i, sent in enumerate(input_texts):
        input_texts[i] = [w if w in vocab_token_index else 'NONE' for w in sent]

    X = np.asarray([[vocab_token_index[w] for w in sent] for sent in input_texts])
    encoder_input_data = sequence.pad_sequences(X, maxlen=maxlen_input)

    return encoder_input_data, input_texts, target_texts, inputList

with open("result/testResult.txt",'w', encoding='UTF-8') as f:
    encoder_input_data, input_texts, target_texts, inputList = test()
    for seq_index in range(len(input_texts)):
        # Take one sequence (part of the training test)
        # for trying out decoding.
        print(seq_index)
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        #print('-')
        #print('Input sentence:', inputList[seq_index])
        #print('Answer  sentence:', target_texts[seq_index])
        #print('Decoded sentence:', decoded_sentence)
        inputStr = ' '.join(inputList[seq_index])
        decoded_sentence = decoded_sentence.replace('EOS','')
        line = 'post: ' + '\n' + inputStr + '\n'  + '\n'  +'Answer  sentence: ' + '\n' + target_texts[seq_index] + '\n' + '\n'  +"Decoded sentence:"  + '\n' + decoded_sentence
        f.write(line + '\n')
        f.writelines("=========================================" + '\n')    
    
f.close()    
print('DONE')
    
    
    
    
    
    
    