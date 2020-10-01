# -*- encoding: utf-8 -*-
import chatapp.ChatFramework.intro as intro
import chatapp.ChatFramework.webtoon as webtoon
import chatapp.ChatFramework.music as music
import chatapp.ChatFramework.movie as movie
import numpy as np

# 챗봇 클래스
class Bot() :

    def __init__(self) :
        self.on_webtoon = 0
        self.context = ''
        self.gerne = []
        self.webtoon_data = np.load('chatapp/ChatFramework/data/webtoon/web_toon_data.npy',allow_pickle='TRUE').item()

        self.on_music = 0

        self.on_movie = 0

        self.step = 0

    # sentence (사용자의 질문)에 대한 챗봇의 답변을 받는 함수
    def get_answer(self, sentence) :
        answer = ''
        if (self.on_webtoon | self.on_music | self.on_movie) == 0 :
            category = intro.opening_first_conv(sentence)
            if category == 'webtoon' : self.on_webtoon = 1
            elif category == 'music' : self.on_music = 1
            elif category == 'movie' : self.on_movie = 1
            else : return '카테고리를 분류하지 못했습니다.<br/>다시 질문 해주시겠어요?'

        if self.on_webtoon :
            if not self.step :
                answer = '어떤 장르의 웹툰을 추천해 드릴까요?'
                self.step = 1
            elif self.step == 1 :
                self.context, self.gerne = webtoon.webtoon_first_conv(sentence, self.context,self.gerne)
                self.step = 2
                answer = '음.. ' + self.gerne[0] + '..<br/>어떤 스토리의 웹툰을 추천해 드릴까요?'
            elif self.step == 2 :
                answer = webtoon.webtoon_second_conv(sentence, self.context, self.gerne)
                answer = '이런 작품은 어떠세요?' + self.process_webtoon_answer(answer)
                self.context = ''
                self.gerne = []
                self.on_webtoon = self.step = 0
        elif self.on_music :
            if not self.step :
                answer = '당신이 좋아하는 음악에 대해 알려주세요!<br/>예) 힙한 힙스터의 그루브'
                self.step = 1
            else :
                answer = music.get_answer(sentence)
                self.on_music = self.step = 0
        elif self.on_movie :
            if not self.step :
                answer = '어떤 영화가 보고싶으세요?'
                self.step = 1
            else :
                answer = movie.get_answer(sentence)
                self.on_movie = self.step = 0

        return answer

    def process_webtoon_answer(self, webtoons) :
        html = ''

        for title in webtoons :
            html += '<br/><br/><br/><a href="' + self.webtoon_data[title]['detail_url'] + '" target="_blank">'
            html += '<img src="' + self.webtoon_data[title]['img_url'] + '" /></a><br/>'
            html += '작품: ' + self.webtoon_data[title]['title'] + '<br/>'
            html += '카테고리: ' + self.webtoon_data[title]['category'] + '<br/>'
            html += '장르: ' + self.webtoon_data[title]['gerne'] + ' > ' + self.webtoon_data[title]['sub_gerne'] + '<br/>'
            html += '평점: ' + self.webtoon_data[title]['star'] + '<br/>'
            html += '컷툰: ' + ('O' if self.webtoon_data[title]['cuttoon'] == 1 else 'X') + '<br/>'
            html += '줄거리: ' + self.webtoon_data[title]['summary'] + '<br/>'
            html += '플랫폼: ' + self.webtoon_data[title]['platform']

        return html
