# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:25:08 2019

@author: 1449486
"""

from pyexcel_ods import get_data
doc = get_data("ques_ans.ods")

import random
from random import shuffle

c1=doc['train']
c1=[x for x in c1 if x]
random.shuffle(c1)

c2=doc['test']
c2=[x for x in c2 if x]
random.shuffle(c2)

train1=[x[0] for x in c1]
train1 = [x.lower() for x in train1]

train11=[]
for i in range(0,len(train1)):
    h=train1[i]
    t=' '.join([h[:len(h)]]*3)
    train11.append(t)
    
    
train2=[x[1] for x in c1]
train2 = [x.lower() for x in train2]

y_train=[x[2] for x in c1]

test1=[x[0] for x in c2]
test1 = [x.lower() for x in test1]

test11=[]
for i in range(0,len(test1)):
    h=test1[i]
    t=' '.join([h[:len(h)]]*3)
    test11.append(t)


test2=[x[1] for x in c2]
test2 = [x.lower() for x in test2]


y_test=[x[2] for x in c2]


#............................................................................


text = open("corpus.txt").read()
my_list = text.split("\n")

my_list1 = [x for x in my_list if x]
my_list2 = [x.lower() for x in my_list1]

my_list2 = train11+train2

#...........................................................................

from nltk.tokenize import word_tokenize

mm=[]
for i in my_list2:
    tokens=word_tokenize(i)
    tokens=[w.lower() for w in tokens]
    mm.append(tokens)

#...........................................................................
    
#..................TRAINING WORD EMBEDDINGS USING FASTTEXT..............................


EMBEDDING_DIM=50

from gensim.models import FastText
model_f = FastText(mm, size=EMBEDDING_DIM, window=5, min_count=1, workers=4,sg=1)

words=list(model_f.wv.vocab)

#.......................saving the embeddings matrix................................

filename='chatbot.txt'
model_f.wv.save_word2vec_format(filename,binary=False)


#.......................getting back the embeddings.........................
import numpy as np

import os
embeddings_index={}
f=open(os.path.join('','chatbot.txt'),encoding="utf-8")
for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:])
    embeddings_index[word]=coefs
f.close()



from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


  
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(mm)
sequences=tokenizer_obj.texts_to_sequences(mm)
  
word_index=tokenizer_obj.word_index
    
num_words=len(word_index)+1    
embedding_matrix=np.zeros((num_words,EMBEDDING_DIM))
max_length=max([len(s) for s in mm])


for word,i in word_index.items():
    if i>num_words:
        continue
    embedding_vector=embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector
        
e=embedding_matrix[1:len(embedding_matrix)]


#...............................................................................

from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout,GRU,GlobalAveragePooling1D
from keras.layers import Conv1D,GlobalMaxPooling1D,SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.layers import merge
from keras.models import Sequential, Model
from keras.layers import Concatenate,Flatten,Reshape,Dense, LSTM, Input, concatenate
import keras.backend as K
import keras
from keras.layers import multiply
from keras.layers import Input, dot, subtract, multiply,add,average
from keras.layers import Bidirectional
from keras.layers.core import Lambda
import tensorflow as tf
from keras.layers import TimeDistributed,Input

input_sentence_length1=30
input_sentence_length2=30
input_dim = EMBEDDING_DIM
num_steps = 50


#......................Input data conversion to padding.........................
#take data from mm..................................
                      



X_train1_tokens=tokenizer_obj.texts_to_sequences(train11)
X_train2_tokens=tokenizer_obj.texts_to_sequences(train2)


X_train1_pad=pad_sequences(X_train1_tokens,maxlen=input_sentence_length1,
                           padding='post')

X_train2_pad=pad_sequences(X_train2_tokens,maxlen=input_sentence_length2,
                           padding='post')


X_test1_tokens=tokenizer_obj.texts_to_sequences(test11)
X_test2_tokens=tokenizer_obj.texts_to_sequences(test2)


X_test1_pad=pad_sequences(X_test1_tokens,maxlen=input_sentence_length1,
                          padding='post')

X_test2_pad=pad_sequences(X_test2_tokens,maxlen=input_sentence_length2,
                          padding='post')


#............................THE TRACING NEWORK................................
                      

#............................THE TRACING NEWORK................................
                      
sequence_input1 = Input(shape=(input_sentence_length1, ))
x = Embedding(num_words,EMBEDDING_DIM, embeddings_initializer=Constant(e),
              trainable = True)(sequence_input1)
x = SpatialDropout1D(0.2)(x)
x = Bidirectional(LSTM(32, return_sequences=True,
                       dropout=0.5,recurrent_dropout=0.5))(x)
x = Conv1D(16, kernel_size = 3, padding = "valid", 
           kernel_initializer = "glorot_uniform")(x)
avg_pool1 = GlobalAveragePooling1D()(x)
max_pool1 = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool1, max_pool1])
#........................................................................

sequence_input2 = Input(shape=(input_sentence_length2, ))
y = Embedding(num_words,EMBEDDING_DIM, embeddings_initializer=Constant(e),
              trainable = True)(sequence_input2)
y = SpatialDropout1D(0.2)(y)
y = Bidirectional(LSTM(32, return_sequences=True,
                       dropout=0.5,recurrent_dropout=0.5))(y)
y = Conv1D(16, kernel_size = 3, padding = "valid", 
           kernel_initializer = "glorot_uniform")(y)
avg_pool2 = GlobalAveragePooling1D()(y)
max_pool2 = GlobalMaxPooling1D()(y)
y = concatenate([avg_pool2, max_pool2])


direction=dot([x, y],axes=-1)

sub=subtract([x, y])

mult=multiply([sub,sub])

distance=Sequential()
distance=Dense(1,activation='relu',kernel_initializer=keras.initializers.Ones(),
                bias_initializer='zeros',trainable=False)(mult)


concat = concatenate([direction,distance])

mm=Sequential()
mm=Dense(1,activation='sigmoid')(concat)

model = Model(inputs=[sequence_input1,sequence_input2], outputs=mm)

adam=keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, 
                           decay=0, amsgrad=False)

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc'])

#................................MODEL SUMMARY..................................

print(model.summary())


#................................MODEL FITTING.................................
h=model.fit([X_train1_pad,X_train2_pad],y_train,batch_size=32,epochs=200,
            verbose=1,validation_data=([X_test1_pad,X_test2_pad],y_test))


import matplotlib.pyplot as plt


# summarize history for accuracy
plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.title('model BiLSTM=32')
plt.ylabel('accuracy')
plt.xlabel('Epochs')
plt.legend(['Training_set', 'Validation_set'], loc='upper left')
plt.show()















