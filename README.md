# 영화 추천 
저는 영화 추천 알고리즘을 만드는데 상당한 시간이 소요되었습니다. 데이터가 많아 처리하는데 시간이 오래 걸렸고 사용자마다 좋아하는 영화가 국가,시간,장르,내용마다 다르기 때문에 어떤 변수를 넣은게 편리한지 여러번 고민하였고 딥러닝 모델을 만드는 과정에서 정확도가 낮아서 다른 방법을 생각해보았습니다. 생각해본 결과 줄거리를 단어,형태소별로 분류해 불용어처리하고 태그를 만들어서 사용자가 선호하는 내용이 담긴 영화를 추천하기로 기획하였습니다. 결론은 사용자가 선호하는 내용을 태그로 만들어 본인이 입력한 것을 태그에 연결해 이에 맞는 영화를 자동으로 추천해주는 것입니다. 

## 데이터 가공
네이버 영화 내 다운로드에 들어가 네이버 시리즈온에서 21250개의 작품을 크롤링하고 영화진흥위원회 db검색을 통해 1980-01=01~2020-09-16일에 개봉한 영화 중 일반영화를 대상으로 필터링을 거쳐 크롤링한 데이터에 취합했습니다. 크롤링 과정에서 감독 및 장르가 누락된 것이 많았기 때문에 객관적인 자료를 찾고자 했습니다.

## 초반 계획
좋아하는 국가, 적당한 러닝타임,선호하는 장르와 내용에 맞게 국가,시간,장르를 라벨링하고 내용을 단어벡터화시켜서 딥러닝 모델을 만들는 것이 목표였으나 단어벡터를 행렬로 만드는 과정에서 고전하였습니다. 내용을 bow로 만들고 단어인덱스를 넣은 상태에서 국가,시간,장르를 라벨링한 상태로 취합한 것을 딥러닝모델로 만들었지만 정확도가 0인 관계로 실패했습니다. 
<pre>
<code>
reviews=movie['내용']
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
# 가장 빈도가 높은 100개의 단어만 선택하도록 Tokenizer 객체를 만듭니다.
tokenizer = Tokenizer(num_words=200)
# 단어 인덱스를 구축합니다.
tokenizer.fit_on_texts(reviews)
# 문자열을 정수 인덱스의 리스트로 변환합니다.
sequences = tokenizer.texts_to_sequences(reviews)
# DTM
BoW_results = tokenizer.texts_to_matrix(reviews, mode='binary')
#bow는 순서가 없고 원핫인코딩은 순서가 있음
# 계산된 단어 인덱스를 구합니다.
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
another=movie[['국가','시간','장르']].values
tests=np.concatenate((another, BoW_results), 1)

</code>
</pre>


