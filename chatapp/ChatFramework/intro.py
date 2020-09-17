############################################ 라이브러리 로드 #########################################

#Json파일 로드와 저장을 위한 라이브러리
import json
import pickle
import random

#자연어 토큰화를 위한 라이브러리
from konlpy.tag import Okt
twitter = Okt()

#데이터 로드와 모델 사용을 위한 라이브러리
from keras.models import Sequential
import keras
import tensorflow as tf
import numpy as np
import pandas as pd

######################################################################################################


############################################ 데이터 로드 ##############################################

#[웹툰,음악,영화]데이터 분류모델 로드, 해당 기능 단어사전데이터 로드
opening_model = keras.models.load_model('chatapp/ChatFramework/data/webtoon/opening_conv.h5')
opening_model_data = np.load('chatapp/ChatFramework/data/webtoon/opening_model_data.npy',allow_pickle='TRUE').item()
open_classes = np.load('chatapp/ChatFramework/data/webtoon/open_classes.npy',allow_pickle='TRUE').item()['file']
open_words = np.load('chatapp/ChatFramework/data/webtoon/open_words.npy',allow_pickle='TRUE').item()['file']

#######################################################################################################


############################################ 함수 정의 #################################################

def clean_up_sentence(sentence,stop_word):
    """들어온 문장을 토큰화해서 리턴해주는 함수"""

    pos_result = twitter.pos(sentence, norm=True, stem=True)
    sentence_words = [lex for lex, pos in pos_result if lex not in stop_word]

    return sentence_words

def bow(sentence, words, show_details=False, stop_word = []):
    """문장과 단어책을 받아서 clean_up_sentence함수로 토큰화된 단어들을 받고 Bag of word생성"""

    sentence_words = clean_up_sentence(sentence,stop_word)
    print(sentence_words)

    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def predict_class(sentense,model,data_len,classes,words,stop_word=[]):
    """사용자에 입력에 따라 값(웹툰,음악,영화)을 예측하는 함수 리턴값은 예측 라벨과 각 라벨에 대한 확률값"""
    p = bow(sentense, words, stop_word=stop_word)
    p = np.array(p)
    p = p.reshape(1,data_len)
    predict_value = model.predict(p)
    class_value = classes[np.argmax(predict_value[0])]
    return class_value, predict_value

#######################################################################################################


############################################ 시나리오 ##################################################

def opening_first_conv(user_answer):
    answer, _ = predict_class(user_answer,opening_model, opening_model_data['words'], open_classes, open_words)
    return answer

#######################################################################################################