#%% 
# import packages

import pandas as pd 
import re

#%%
#1. data loading

# read csv
df = pd.read_csv("https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv")

# %%
#2. data inspection

#%%
#2.1 .head()
df.head()

#%%
#2.2 .info()
df.info()

#%%
#2.3 .describe()
df.describe().T

# %%
#3. data cleaning

#%%
#3.1 check for duplicated and missing
print(df.isna().sum())
print(df.duplicated().sum())

#3.2 remove duplicated and missing
df = df.drop_duplicates()

print(df.duplicated().sum())

review = df['review']
sentiment  = df['sentiment']

for index,rev in review.items():                                        
    review[index] = re.sub('<.*?>', ' ', rev)                           #loop to remove html
    review[index] = re.sub('[^a-zA-Z]', ' ',review[index]).lower()      #loop to keep alphabet and lower
# %%
