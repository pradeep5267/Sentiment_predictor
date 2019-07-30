#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.utils import shuffle

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dropout, Dense, Embedding, LSTM, SpatialDropout1D, \
    CuDNNLSTM, Bidirectional, Activation,Input,GlobalMaxPool1D
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
import re

#%%
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
#%%
K.clear_session()
#%%
data = pd.read_csv("./train_modified_innoplexus_sentiment.csv")
data.head()
#%%
# Keeping only the neccessary columns
data = data[['text_new','sentiment']]

#%%

data['text_new'] = data['text_new'].apply(lambda x: x.lower())
data['text_new'] = data['text_new'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

# sanity check
# print(data[ data['sentiment'] == 0].size)
# print(data[ data['sentiment'] == 1].size)
# print(data[ data['sentiment'] == 2].size)
#%%
max_fatures = 2000
embed_dim = 128
lstm_out = 196
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text_new'].values)
X = tokenizer.texts_to_sequences(data['text_new'].values)
X = pad_sequences(X)

#%%

#%%
# model 1 CuDNN
def cu_LSTM():
    max_fatures = 2000
    embed_dim = 128
    lstm_out = 196
    model1 = Sequential()
    model1.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model1.add(CuDNNLSTM(lstm_out))
    model1.add(Dropout(0.3))
    model1.add(Dense(3,activation='softmax'))
    model1.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model1.summary())
    return model1

#%%
# model 2 Vanilla LSTM
def vanilla_LSTM():
    max_fatures = 2000
    embed_dim = 128
    lstm_out = 196

    model2 = Sequential()
    model2.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
    model2.add(SpatialDropout1D(0.4))
    model2.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model2.add(Dense(3,activation='softmax'))
    model2.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model2.summary())
    return model2


#%%
# model 3 bidirectional LSTM
def bi_LSTM():
    max_fatures = 2000
    embed_dim = 128
    lstm_out = 196
    inp = Input(shape=(10781,))
    x = Embedding(max_fatures, embed_dim,input_length = X.shape[1])(inp)
    x = Bidirectional(CuDNNLSTM(100, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(100, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(3, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#%%
Y = pd.get_dummies(data['sentiment']).values
X_train, Y_train = shuffle(X, Y)

#%%
# if you want to use train test split then uncomment this block
# Y = pd.get_dummies(data['sentiment']).values
# X_train, Y_train = train_test_split(X,Y, random_state = 42)
# print(X_train.shape,Y_train.shape)
# print(X_test.shape,Y_test.shape)


#%%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
batch_size = 32
model = cu_LSTM()
model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 1)
#%%
model.save('./sentiment_cu_LSTM_f1.h5')
#%%
def process_text(data,ip_maxlen):
    data['text_new'] = data[['text', 'drug']].apply(lambda x: ' '.join(x), axis=1)
    data['text_new'] = data['text_new'].apply(lambda x: x.lower())
    data['text_new'] = data['text_new'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(data['text_new'].values)
    X = tokenizer.texts_to_sequences(data['text_new'].values)
    X = pad_sequences(X,maxlen=ip_maxlen, dtype='int32', value=0)
    return X
#%%
# get input layer details
model = load_model('./sentiment_cu_LSTM_f1.h5')
ip = model.get_layer(name='embedding_1')
ip_details = ip.get_config()
ip_maxlen = ip_details['input_length']
print(ip_details)
#%%
#%%
test_data = pd.read_csv('./test_innoplexus_sentiment.csv')

# unique_hash = test_data.unique_hash.to_list


new_test = pd.DataFrame(test_data,columns=['unique_hash','text','drug'])
#%%
X_list = []
sentiment = []
X = process_text(new_test,ip_maxlen)
#%%
# create a list of ip_maxlen columns and 2421 something rows
for i in X:
    X_list.append(i)
#%%
sentiment = []
for i in range(0,len(X)):
    y = model.predict(np.array( [X[i],] ) ,verbose=2)
    y = np.argmax(y)
    sentiment.append((y))
    # for sanity check
    # if i==20:
    #     break
# print(sentiment)
#%%
my_submission_test = pd.DataFrame({'unique_hash': new_test.unique_hash,'sentiment': sentiment})
# you could use any filename. We choose submission here
my_submission_test.to_csv('test_submission.csv', index=False)