#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


# In[2]:

movie, tag_label, max_class = 0, 0, 0

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


# In[71]:

def movie_user_model(sentence):   
    movie=pd.read_csv('chatapp/ChatFramework/data/movie/movie_labeled.csv')
    tag_label = np.zeros((len(movie['태그'])))
    for i in range(len(movie['태그'])):
        for a in sentence:
            if a in movie['태그'][i]:
                tag_label[i]+=1
    max_class=max_class=int(np.max(tag_label))
    return movie,tag_label ,max_class


# In[72]:



# In[55]:
# In[57]:


def data_split(data):
    train_data=data[:2000]
    test_data=data[2000:]
    max_words = 200 # 실습에 사용할 단어의 최대 개수
    #num_classes= tag_label 
    return train_data,test_data,max_words


# In[58]:


def data_detail_and_genre():#내용을 넣고 그에 맞는 장르를 찾음
    train_data,test_data,max_words=data_split(movie)
    train_detail=train_data['내용']
    test_detail=test_data['내용']
    train_genre=train_data['장르_l']
    test_genre=test_data['장르_l']
    return train_detail, test_detail, train_genre,test_genre


# In[59]:


def prepare_data(train_data,test_data): 
    train_detail, test_detail, train_genre,test_genre=data_detail_and_genre()
    t = Tokenizer(num_words = max_words) 
    t.fit_on_texts(train_data)
    X_train = t.texts_to_matrix(train_data, mode='count')
    X_test = t.texts_to_matrix(test_data, mode='count') 
    return X_train, X_test,max_words


# In[60]:


def set_data():
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical
    train_data,test_data,max_words=data_split(movie)
    X_train,X_test,max_words=prepare_data(train_data,test_data)
    train_detail, test_detail, train_genre,test_genre=data_detail_and_genre()
    y_train = to_categorical(train_genre, 23) 
    y_test = to_categorical(test_genre, 23) 

    #print('훈련 샘플 본문의 크기 : {}'.format(X_train.shape)) 
    # 모든 단어들의 빈도수를 세고 가장 빈도수가 높은 200개 뽑음.
    #print('훈련 샘플 레이블의 크기 : {}'.format(y_train.shape))
    #print('테스트 샘플 본문의 크기 : {}'.format(X_test.shape))
    #print('테스트 샘플 레이블의 크기 : {}'.format(y_test.shape))
    return X_train,X_test,y_train,y_test


# In[61]:


def fit_and_evaluate():
    X_train,X_test,y_train,y_test=set_data()
    from tensorflow.keras import models #내용구분 모델링
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense,Dropout
    model=Sequential()
    model.add(Dense(256,activation='relu',input_shape=(max_words,)))
    model.add(Dropout(0.5))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(23,activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(X_train,y_train,batch_size=128,epochs=5,validation_split=0.2)
    model.summary()
    score=model.evaluate(X_test,y_test,batch_size=128)
    return model


# In[62]:


def movie_predict():#내용을 바탕으로 원하는 영화 추출
    movie=pd.read_csv('chatapp/ChatFramework/data/movie/movie_labeled.csv')
    movie_x = movie[['제목','감독','시간','장르','내용']]
    movie_feature = movie['내용']
    t = Tokenizer(num_words = 200) 
    t.fit_on_texts(movie_feature)
    movie_data2 = t.texts_to_matrix(movie_feature, mode = 'count')
    return movie_x,movie_data2

def movie_apply_predict():
    model = fit_and_evaluate()
    movie_x,movie_data2=movie_predict()
    predict_value = model.predict(movie_data2)
    predict_label = []
    for i in range(len(predict_value)):
        predict_label.append(np.argmax(predict_value[i]))
        
    movie_x['label'] = predict_label
    return movie_x


# In[63]:


def recommendation():
    movie_x=movie_apply_predict()
    final = movie_x[movie_x['label'] == movie_x['label'].max()].reset_index(drop=True)
    final_1 = final.sample(n=1)
    final_1 = final.to_dict(orient="record")[0]
    answer = '당신에게 {} 장르의 {}감독이 만든 {}을 추천합니다'.format(final_1['장르'],final_1['감독'],
                                                    final_1['제목'])
    return answer


# In[64]:


def get_answer(sentence):
    global movie, tag_label, max_class
    sentence=question(sentence)
    movie,gen_label,max_class=movie_user_model(sentence)
    answer=recommendation()
    return answer
