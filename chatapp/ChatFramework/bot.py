# -*- encoding: utf-8 -*-

# 챗봇 클래스
class Bot() :

    def __init__(self) :
        # 사용자의 콘텐츠 선호도에 대한 정보를 담는 딕셔너리
        self.user_prefer_data = { }

    # sentence (사용자의 질문)에 대한 챗봇의 답변을 받는 함수
    def get_answer(self, sentence) :
        return '답변'

    # 선호도 조사에 사용할 웹툰, 영화, 음악 목록을 반환.
    # 선호도 조사가 끝나면 set_user_prefer 함수를 통해 선호도를 설정한다.
    def contents_for_prefer_research(self):
        contents = {'웹툰': {
                    '웹툰1': {'제목': '제목1', '그림체': '그림체1', '줄거리': '줄거리1', '선호': 0},
                    '웹툰2': {'제목': '제목2', '그림체': '그림체2', '줄거리': '줄거리2', '선호': 0}},

                    '영화': {
                    '영화1': {'제목': '제목1', '감독': '감독1', '줄거리': '줄거리1', '선호': 0},
                    '영화2': {'제목': '제목2', '감독': '감독2', '줄거리': '줄거리2', '선호': 0}},

                    '음악': {
                    '음악1': {'곡명': '곡명1', '가사': '가사1', '선호': 0},
                    '음악2': {'곡명': '곡명2', '가사': '가사2', '선호': 0}}}

        return contents

    # 선호도 조사를 마치고 user_prefer_data 의 '선호' 부분에
    # 0 혹은 1의 값이 들어간 상태로 다시 전달받는다.
    def set_user_prefer(self, user_prefer_data) :
        self.user_prefer_data = user_prefer_data
        print(self.user_prefer_data)
