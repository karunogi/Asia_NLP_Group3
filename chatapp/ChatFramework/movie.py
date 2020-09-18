#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


# In[2]:
sentence = ''


#movie, tag_label, max_class = 0, 0, 0
def question(sentence):#선호하는 것 묻기
    answer=sentence
    from konlpy.tag import Okt
    okt=Okt()
    okt_tokens=okt.morphs(answer)  
    oktTag = []
    for token in okt_tokens:
        oktTag += okt.pos(token)
    stopwords=['을','은','가','할','수','건','것','추천','해','주세요','줘',
          '없','어요','뭐','있','지','를','합니다','는','저','영화']
    stoppos=['Determiner','Adverb','Conjunction','Josa','PreEomi','Eomi','Suffix',
           'Punctuation','Foreign','Alpha','Number','Unknown', 'Adjective']
    sentence=[]
    for tag in oktTag:
        if tag[1] not in stoppos:
            if tag[0] not in stopwords:
                sentence.append(tag[0])
    return sentence


# In[3]:


def descript_to_list():
    movies=pd.read_csv('chatapp/ChatFramework/data/movie/movie_labeled.csv')
    from konlpy.tag import Okt#줄거리를 일부를 나눠서 태그 구성
    okt=Okt()
    oktTag = []
    wordlist=[]
    for i in movies['내용']:  
        word=okt.morphs(i)
        oktTag = []
        for token in word:
            oktTag += okt.pos(token)
        stopwords = ['의','가','이','은','을','들','는','좀','잘','걍','과','도','를','으로','다','이','가','자','와','한','하다','에','에서','께서',
                 '이다','에게','으로','이랑','까지','부터','하다','하는','것','데','짜리','했지만','된다',
                '로','전','차','하지','고','않','단','뿐']
        stoppos=['Determiner','Adverb','Conjunction','Josa','PreEomi','Eomi','Suffix',
               'Punctuation','Foreign','Alpha','Number','Unknown', 'Adjective']
        desc=[]
        for tag in oktTag:
            if tag[1] not in stoppos:
                if tag[0] not in stopwords:
                    desc.append(tag[0])
        wordlist.append(desc)
    #print(len(words))
    return movies,wordlist


# In[4]:


def movie_user_model(sentence):
    movies,wordlist=descript_to_list()
    tags=[]
    for i in wordlist:
        for j in i:
            tags.extend(i)
    tags=list(set(tags))

    tag_label=np.zeros(len(tags))
    for i in range(len(tags)):
        for a in sentence:
            if a in tags[i]:
                tag_label[i]+=1
    max_class=max_class=int(np.max(tag_label))
    return tags,tag_label,max_class


# In[5]:


def data_split():
    tags,tag_label,max_class=movie_user_model(sentence)
    train_data=tags[:1500]
    test_data=tags[1500:]

    train_label=tag_label[:1500]
    test_label=tag_label[1500:]    
    max_words = 200 # 실습에 사용할 단어의 최대 개수
    #num_classes= tag_label 
    return train_data,test_data,train_label,test_label,max_words


# In[6]:


def prepare_data(train_data,test_data): 
    train_data,test_data,train_label,test_label,max_words=data_split()
    t = Tokenizer(num_words = max_words) 
    t.fit_on_texts(train_data)
    X_train = t.texts_to_matrix(train_data, mode='count')
    X_test = t.texts_to_matrix(test_data, mode='count') 
    return X_train, X_test, train_label, test_label, max_words


# In[7]:


def set_data():
    #from tensorflow.tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical
    tags,tag_label,max_class=movie_user_model(sentence)
    train_data,test_data,train_label,test_label,max_words=data_split()
    X_train, X_test, train_label, test_label, max_word=prepare_data(train_data,test_data)
    y_train = to_categorical(train_label, max_class+1) 
    y_test = to_categorical(test_label, max_class+1) 

    return X_train,X_test,y_train,y_test,max_class


# In[8]:


def fit_and_evaluate():
    X_train,X_test,y_train,y_test,max_class=set_data()
    from tensorflow.keras import models #태그구분 모델링
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense,Dropout
    train_data,test_data,train_label,test_label,max_words=data_split()
    model=Sequential()
    model.add(Dense(256,activation='relu',input_shape=(max_words,)))
    model.add(Dropout(0.5))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(max_class+1,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(X_train,y_train,batch_size=128,epochs=5,validation_split=0.2)
    model.summary()
    score=model.evaluate(X_test,y_test,batch_size=128)
    return model


# In[9]:


def movie_predict():#내용을 바탕으로 원하는 영화 추출
    movies=pd.read_csv('chatapp/ChatFramework/data/movie/movie_labeled.csv')
    movie_x = movies[['제목','감독','시간','장르','내용']]
    movie_feature = movies['내용']
    t = Tokenizer(num_words = 200) 
    t.fit_on_texts(movie_feature)
    movie_data2 = t.texts_to_matrix(movie_feature, mode = 'count')
    return movie_x,movie_data2


# In[10]:


def movie_apply_predict():
    model = fit_and_evaluate()
    movie_x,movie_data2=movie_predict()
    predict_value = model.predict(movie_data2)
    predict_label = []
    for i in range(len(predict_value)):
        predict_label.append(np.argmax(predict_value[i]))

    movie_x['label'] = predict_label
    return movie_x


# In[11]:


def recommendation():
    movie_x=movie_apply_predict()
    final = movie_x[movie_x['label'] == movie_x['label'].max()].reset_index(drop=True)
    final_1 = final.sample(n=1)
    final_1 = final.to_dict(orient="record")[0]
    answer = '당신에게 {} 장르의 {}감독이 만든 {}을 추천합니다'.format(final_1['장르'],final_1['감독'],
                                                final_1['제목'])
    return answer


# In[12]:


def get_answer(_sentence):
    global sentence
    sentence = _sentence
    _sentence=question(_sentence)
    tags,tag_label,max_class=movie_user_model(_sentence)
    answer=recommendation()
    return answer


# In[ ]:

