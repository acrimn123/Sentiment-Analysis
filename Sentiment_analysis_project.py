#%%
#import packages

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input,LSTM,Dense,Bidirectional,Embedding
import re 
import numpy as np

#%%
# Data Loading

df = pd.read_csv('https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv') #url need to be in raw format to be load using read_csv


#%%
# Data Inspection

df.describe()
df.info()

missing = df.isna().sum()           #to check NaN but no missing
duplicated = df.duplicated().sum()  # to check for duplicated 418 found

#%%
# Data Cleaning

df = df.drop_duplicates()     #drop duplicated review

review = df['review']         # features
sentiment = df['sentiment']   # target 

print('\n',review[10])        # check text review

for index,rev in review.items():                                        
    review[index] = re.sub('<.*?>', ' ', rev)                           #loop to remove html
    review[index] = re.sub('[^a-zA-Z]', ' ',review[index]).lower()      #loop to keep alphabet and lower

print('\n',review[10])        # check text review again


#%%
# Features selection
#%%
# Data preprocessing

#Features (X)
#Tokenization
num_words = 5000
oov_token = '<OOV>'
pad_type ='post'
trunc_type ='post'

tokenizer = Tokenizer(num_words=num_words,oov_token=oov_token)
tokenizer.fit_on_texts(review)

word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(review)

#padding + truncating
train_sequences = pad_sequences(train_sequences,maxlen=200,padding=pad_type,truncating=trunc_type)

#Target label (Y)
#One hot encoder

ohe = OneHotEncoder(sparse=False)
train_sentiment = ohe.fit_transform(sentiment[::,None])

train_sequences = np.expand_dims(train_sequences,-1)

#train test split

x_train,x_test,y_train,y_test = train_test_split(train_sequences,train_sentiment,shuffle=True,random_state=12345)

#%%
embedding_size = 64

# Model Development
model = Sequential()

#model.add(Input(x_train.shape[1:]))
model.add(Embedding(num_words,embedding_size))                          #embedding output size must = to next layer
model.add(Bidirectional(LSTM(embedding_size,return_sequences=True)))    
model.add(LSTM(64))
model.add(Dense(2,activation='softmax'))

model.summary()

#%%
#model compilation
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

#
hist = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5)
#%%
# Model Evaluation
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

y_pred = model.predict(x_test)
y_pred = (np.argmax(y_pred,axis=1))
y_true = (np.argmax(y_test,axis=1))

print(confusion_matrix(y_true,y_pred))           
print(classification_report(y_true,y_pred))   #f1 score 
print(accuracy_score(y_true,y_pred))   #

#%%
# saving models

#save tokenizer
import json
with open('tokenizer.json','w') as f:
    json.dump(tokenizer.to_json(),f)

#save ohe
import pickle
with open('ohe.pkl','wb') as f:
    pickle.dump(ohe,f)

#save model

model.save('gpt.h5')

# %%
