#%%
#load model

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import json,pickle,numpy as np

loaded_model = load_model('gpt.h5')

loaded_model.summary()
# %%
# load tokenizer

with open('tokenizer.json','r') as f:
    loaded_tokenizer = json.load(f)

loaded_tokenizer = tokenizer_from_json(loaded_tokenizer)
#%%
#load pickle

with open('ohe.pkl','rb') as f:
    loaded_ohe = pickle.load(f)

#%%
# deploy

test_review = ['I don t understand why the critics are bashing on this one. I honestly didn t know what I was going into and it was such a delightful surprise! It really exceeded my expectations and I had fun during the whole movie. Critics are really being harsh and I can t believe that beauty and the beast was more critically praised. The movie was so bland and boring while this one has so much energy and life to it. The characters had amazing chemistry with each other and the lines were delivered smoothly. The two things I was worried about that subsided as soon as the movie started are: 1- Jafar s character: Marwan tries his best to make but the material he s given isn t strong enough. I wish they developed his character more. However, I didn t mind the change they did it with his character as it fits more with this version. 2- Cultural representation: The trailers made the movie look like a production from bollywood. Not hating on the Indian culture, but it really annoyed me as Arab to see a Middle Eastern/Arabic folk tale that we all know long before Disney introduced it to the world to be represented in a fully different culture. Fortunately, the movie looks and feels way more Arabian such Arabic writings here and there, hearing some people speaking Arabic briefly, the names and appearance of the people of Agrabah and of course the amazing score! Although there are some slight hints of India/South Asia in the clothing and dancing, but I didn t mind that at all. I would ve give it a 10 if the cinematography was better. This is my only issue with movie as it looked like high budget TV soap opera in some scenes.']

test_review = loaded_tokenizer.texts_to_sequences(test_review)

test_review = pad_sequences(test_review,200,padding='post',truncating='post')

# %%
#predict

predict = np.argmax(loaded_model.predict(test_review))

print(predict)

if predict == 1:
    out = [0,1]
else:
    out = [1,0]

print(loaded_ohe.inverse_transform([out]))
# %%
