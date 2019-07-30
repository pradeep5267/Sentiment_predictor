#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
# try:
# 	os.chdir(os.path.join(os.getcwd(), 'NLP_saves'))
# 	print(os.getcwd())
# except:
# 	pass

#%%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


#%%
del(data)
del(data_new)
#%%
data = pd.read_csv("./train_innoplexus_sentiment.csv")
# Keeping only the neccessary columns
data = data[['text','drug','sentiment']]


#%%
print(data[ data['sentiment'] == 0].size)
print(data[ data['sentiment'] == 1].size)
print(data[ data['sentiment'] == 2].size)

data_new = data.drop(data.query('sentiment == 2').sample(frac=0.825).index)
data_new['text'] = data_new['text'].apply(lambda x: x.lower())
data_new['text'] = data_new['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))

#%%
data_new.head()

#%%
print(data_new.drug.nunique)

#%%
data_new['text_new'] = data_new[['text', 'drug']].apply(lambda x: ' '.join(x), axis=1)
data_new.head()
#%%
data_new.drop(['text', 'drug'],inplace = True,axis = 1)
data_new.head()
#%%
data_new.to_csv("./train_modified_innoplexus_sentiment.csv")