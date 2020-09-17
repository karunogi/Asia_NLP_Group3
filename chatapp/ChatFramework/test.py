import webtoon
import intro

########################################## 인트로 테스트 ##############################################

# user_answer변수에 사용자의 대화를 담아서 intro.opening_first_conv함수에 전달하면 라벨이 나옵니다(웹툰,음악,영화)

user_answer = input('어떤 콘텐츠를 추천해 드릴까요?')
answer = intro.opening_first_conv(user_answer)

print(answer)

######################################################################################################


########################################## 웹툰 테스트 ################################################

# 문자열 context와 리스트형 gerne를 선언하고 사용자 대답과 함께 webtoon.webtoon_first_conv함수에 넘겨주면
# context(사용자의 대답을 다 모아두는 문자열), gerne(분류된 웹툰의 10개의 장르)를 리턴해 줍니다.
# 다시 사용자에게 질문을 해서 받은 값을 user_answer에 담고 webtoon.webtoon_second_conv함수에 user_answer, context, gerne를 넘겨주면
# 추천 결과값을 데이터프레임으로 리턴해줍니다.(10개)

# 추후에 이미지 로드 하실때는 webtoon.webtoon_data에 웹툰 이름을 키 값으로 접근 하시면 ['img_url']에 이미지 링크를 사용하시면 될 것 같습니다.
# 예시) webtoon.webtoon_data['독립일기']['img_url'] => 'https://shared-comic.pstatic.net/thumb/webtoon/748105/thumbnail/thumbnail_IMAG10_becd3e24-cc09-4243-a1c9-646270f4a8db.jpg'
# 데이터 안에 ['img_url'] 말고도 많은 데이터가 있습니다. 답변 창 구성하실때 이용하시면 될 것 같습니다.

context = ''
gerne = []

user_answer = input('어떤 장르의 웹툰을 추천해 드릴까요?')
context, gerne = webtoon.webtoon_first_conv(user_answer, context, gerne)

user_answer = input('어떤 스토리의 웹툰을 추천해 드릴까요?')
result = webtoon.webtoon_second_conv(user_answer, context, gerne)

print(result)

#######################################################################################################