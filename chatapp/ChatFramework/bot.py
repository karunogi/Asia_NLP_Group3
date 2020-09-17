# -*- encoding: utf-8 -*-
import chatapp.ChatFramework.intro as intro
import chatapp.ChatFramework.webtoon as webtoon
import chatapp.ChatFramework.music as music
import chatapp.ChatFramework.movie as movie

# 챗봇 클래스
class Bot() :

    def __init__(self) :
        self.on_webtoon = 0
        self.webtoon_step = 0
        self.context = ''
        self.gerne = []

        self.on_music = 0
        self.music_step = 0

        self.on_movie = 0
        self.movie_step = 0

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
            if not self.webtoon_step :
                answer = '어떤 장르의 웹툰을 추천해 드릴까요?'
                self.webtoon_step = 1
            elif self.webtoon_step == 1 :
                self.context, self.gerne = webtoon.webtoon_first_conv(sentence, self.context,self.gerne)
                self.webtoon_step = 2
                answer = '음.. ' + self.gerne[0] + '..<br/>어떤 스토리의 웹툰을 추천해 드릴까요?'
            elif self.webtoon_step == 2 :
                answer = webtoon.webtoon_second_conv(sentence, self.context, self.gerne)
                answer = '이런 작품은 어떠세요?<br/><br/>' + '<br/>'.join(answer.tolist())
                self.context = ''
                self.gerne = []
                self.on_webtoon = self.webtoon_step = 0
        elif self.on_music :
            if not self.music_step :
                answer = '당신이 좋아하는 음악에 대해 알려주세요!<br/>예) 힙한 힙스터의 그루브'
                self.music_step = 1
            else :
                answer = music.get_answer(sentence)
                self.on_music = self.music_step = 0
        elif self.on_movie :
            if not self.movie_step :
                answer = '어떤 영화가 보고싶으세요?'
                self.movie_step = 1
            else :
                answer = movie.get_answer(sentence)
                self.on_movie = self.movie_step = 0

        return answer
