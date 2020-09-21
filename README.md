# 또, 오셨네요. RE,COMMA
RE,COMMA는 사용자의 질문을 NLP를 통해 분석하여, 사용자 맞춤형 콘텐츠를 추천해주는 대화형 콘텐츠 추천 챗봇입니다.

추천을 뜻하는 Recommendation과 쉼표라는 의미의 Comma를 합쳐 휴식을 추천해준다는 의미를 담고있습니다.

## 작동 예시
![Honeycam 2020-09-19 22-57-42](https://user-images.githubusercontent.com/34763810/93669403-d6130880-face-11ea-9299-80a97e744e33.gif)
![Honeycam export 2020-09-19 23-11-30](https://user-images.githubusercontent.com/34763810/93669404-d6ab9f00-face-11ea-92a9-becf857692ff.jpg)
![Honeycam export 2020-09-19 23-11-41](https://user-images.githubusercontent.com/34763810/93669405-d7443580-face-11ea-8133-6503bacaf0c2.jpg)
![Honeycam export 2020-09-19 23-11-52](https://user-images.githubusercontent.com/34763810/93669464-4cb00600-facf-11ea-8fd5-c9dbc765b7eb.jpg)
![Honeycam export 2020-09-19 23-12-00](https://user-images.githubusercontent.com/34763810/93669475-8123c200-facf-11ea-8fde-be92797bfb8d.jpg)
![Honeycam export 2020-09-19 23-12-15](https://user-images.githubusercontent.com/34763810/93669398-d4e1db80-face-11ea-9888-cfe092a7912e.jpg)
![Honeycam export 2020-09-19 23-12-23](https://user-images.githubusercontent.com/34763810/93669400-d4e1db80-face-11ea-8d26-17879fa3a3da.jpg)
![Honeycam export 2020-09-19 23-12-46](https://user-images.githubusercontent.com/34763810/93669401-d57a7200-face-11ea-8354-bf46b87d7ac7.jpg)
![Honeycam export 2020-09-19 23-12-54](https://user-images.githubusercontent.com/34763810/93669402-d6130880-face-11ea-908e-37f94a0719f6.jpg)

## 실행 방법
- [HowToRun.txt](https://github.com/karunogi/Asia_NLP_Group3/blob/master/HowToRun.txt)
--------------
- 이 프로젝트는 아시아경제 청년취업아카데미: AI 딥러닝 기반 자연어처리 전문가 양성과정 수강생들의 작품입니다.
- 과정 수강중 강의해주신 [박형식](https://github.com/arkwith7/ArkChatBot) 아크위드(주) 대표님의 도움을 받아 제작되었습니다.


# 영화 추천
담당: 채송화(덕성여자대학교/전공:수학과,부전공:정보통계학과)
SNS:@eclair0715(instagram)

## 기획과정
저는 영화 추천 알고리즘을 만드는데 상당한 시간이 소요되었습니다. 데이터가 많아 처리하는데 시간이 오래 걸렸고 사용자마다 좋아하는 영화가 국가,시간,장르,내용마다 다르기 때문에 어떤 변수를 넣은게 편리한지 여러번 고민하였고 딥러닝 모델을 만드는 과정에서 정확도가 낮아서 다른 방법을 생각해보았습니다. 생각해본 결과 줄거리를 단어,형태소별로 분류해 불용어처리하고 태그를 만들어서 사용자가 선호하는 내용이 담긴 영화를 추천하기로 기획하였습니다. 결론은 사용자가 선호하는 내용을 태그로 만들어 본인이 입력한 것을 태그에 연결해 이에 맞는 영화를 자동으로 추천해주는 것입니다. 

## 데이터 가공
네이버 영화 내 다운로드에 들어가 네이버 시리즈온에서 21250개의 작품을 크롤링하고 영화진흥위원회 db검색을 통해 1980-01-01~2020-09-16일에 개봉한 영화 중 일반영화를 대상으로 필터링을 거쳐 크롤링한 데이터에 취합했습니다. 크롤링 과정에서 감독 및 장르가 누락된 것이 많았기 때문에 객관적인 자료를 찾고자 했습니다.
[네이버 시리즈온](https://serieson.naver.com/movie/categoryList.nhn?categoryCode=ALL,"series on")
[영화진흥위원회](http://www.kobis.or.kr/kobis/business/mast/mvie/searchMovieList.do, "kobis link")

## 초반 계획
좋아하는 국가,사용자에게 적당한 러닝타임,선호하는 장르와 내용에 맞게 국가,장르를 라벨링하고 내용을 단어벡터화시켜서 딥러닝 모델을 만들는 것이 목표였으나 단어벡터를 행렬로 만드는 과정에서 고전하였습니다. 내용을 bow로 만들고 단어인덱스를 넣은 상태에서 국가,시간,장르를 라벨링한 상태로 취합한 것을 딥러닝모델로 만들었지만 정확도가 0인 이유로 실패했습니다. 
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

## 중반부
### 태그 제작-1
내용을 단어/형태소 분류하여 태그로 만들고 이를 라벨링해서 앞서 라벨링 실행한 것으로 딥러닝 모델을 제작하려 했지만 한개당 리스트로 들어간 상태에 문자열로 만들어져있어서 입력받은 단어로 영화를 찾아내는 데 실패했습니다.
<pre>
<code>
from konlpy.tag import Okt
okt=Okt()
oktTag = []
wordlist=[]
for i in movie['내용']:  
    word=okt.morphs(i)
    oktTag = []
    for token in word:
        oktTag += okt.pos(token)
    stopwords = ['의','가','이','은','을','들','는','좀','잘','걍','과','도','를','으로','다','이','가','자','와','한','하다','에','에서','께서',
             '이다','에게','으로','이랑','까지','부터','하다','하는','것','데','짜리','했지만','된다',
            '로','전','차','하지','고','않','단','뿐']
    stoppos=['Determiner','Adverb','Conjunction','Josa','PreEomi','Eomi','Suffix',
           'Punctuation','Foreign','Alpha','Number','Unknown', 'Adjective']
    sentence=[]
    for tag in oktTag:
        if tag[1] not in stoppos:
            if tag[0] not in stopwords:
                sentence.append(tag[0])
    wordlist.append(sentence)
    
script=[]
for i in wordlist:
    word=[]
    for j in i:
        if j not in stopwords:
            word.append(j)
    script.append(word)#형태소 나눈 것을 리스트화
    
for i in range(len(movie)):
    movie['태그'][i]=wordlist[i] 
</code>
</pre>

### 태그 제작-2
결국 라벨링한 것을 거부하고 태그를 다시 만들었습니다.
<pre>
<code>
def descript_to_list():
    movies=pd.read_csv('movie_labeled.csv')
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
</code>
</pre>

### 선호하는 내용 묻기
선호하는 내용의 일부를 입력 받아서 함수 내에서 형태소,단어 분류 후 불용어를 제거
<pre>
<code>
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
</code>
</pre>

### 사용자 맞춤 모델 제작
영화 데이터와 태그를 만들기 위한 단어리스트를 받고 태그의 중복을 방지하기 위해 tags를 새로 제작했습니다
입력받은 sentence를 가지고 해당 단어가 tag에 있으면 1을 추가해 가장 많이 나온 태그를 np.max를 사용해서 max class를 만들었습니다
<pre>
<code>    
def movie_user_model(sentence):
movies,wordlist=descript_to_list()
tags=[]
for i in wordlist:
    for j in i:
        tags.extend(i)
tags=list(set(tags))#태그 백터를 만들어 중복을 제거

tag_label=np.zeros(len(tags))
for i in range(len(tags)):
    for a in sentence:
        if a in tags[i]:
            tag_label[i]+=1
max_class=max_class=int(np.max(tag_label))
return tags,tag_label,max_class
</code>
</pre>

### 데이터 분류 및 딥러닝 모델 설계
tags,tag_label 둘 다 train,test로 나누고 dtm을 만들어 x_train,x_test를 생성하고 y_train,y_test를 원핫인코딩하였습니다.
딥러닝 모델을 설계후 각각 256,128개의 dense와 dropout을 둘 다 0.5로 정하여 학습한 결과 loss는 갈수록 줄어들지만 정확도는 약 80% 되는 것을 알 수 있습니다.
tag_label에서 가장 많이 나온 값인 max_class에 1을 추가해서 최종 Dense의 층으로 결정했습니다.(같은 단어를 가지고 있을 경우)
<pre>
<code>
Train on 1200 samples, validate on 300 samples
Epoch 1/5
1200/1200 [==============================] - 2s 1ms/sample - loss: 1.5393 - acc: 0.7892 - val_loss: 1.4667 - val_acc: 0.7900
Epoch 2/5
1200/1200 [==============================] - 0s 35us/sample - loss: 1.2984 - acc: 0.8008 - val_loss: 1.1501 - val_acc: 0.7900
Epoch 3/5
1200/1200 [==============================] - 0s 34us/sample - loss: 0.9330 - acc: 0.8008 - val_loss: 0.7641 - val_acc: 0.7900
Epoch 4/5
1200/1200 [==============================] - 0s 32us/sample - loss: 0.6930 - acc: 0.8008 - val_loss: 0.6465 - val_acc: 0.7900
Epoch 5/5
1200/1200 [==============================] - 0s 34us/sample - loss: 0.6163 - acc: 0.8008 - val_loss: 0.6007 - val_acc: 0.7900
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 256)               51456     
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               32896     
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 645       
=================================================================
Total params: 84,997
Trainable params: 84,997
Non-trainable params: 0
_________________________________________________________________
25874/25874 [==============================] - 0s 15us/sample - loss: 0.5831 - acc: 0.8081
</code>
</pre>
### 내용을 바탕으로 원하는 영화 추출 작업
핵심을 뽑아내기 위해 movies 데이터에 제목,감독,시간,장르,내용만 뽑고 feature를 내용을 중심으로 최대 단어 200개를 가지고 dtm을 만들었습니다.
<pre>
<code>
movies=pd.read_csv('movie_labeled.csv')
movie_x = movies[['제목','감독','시간','장르','내용']]
movie_feature = movies['내용']
t = Tokenizer(num_words = 200) 
t.fit_on_texts(movie_feature)
movie_data2 = t.texts_to_matrix(movie_feature, mode = 'count')
</code>
</pre>
movie_data2를 model.predict에 삽입해 predict_label을 만들고 가장 높은 값이 들어있는 곳의 위치를 출력해 predict_label을 만들고 movie_x에 label 항복을 만들어 넣었습니다.
그 후 해당 라벨이 가장 큰 값이 final이 되어서 사용자에게 가장 잘 어울리는 영화를 추천해주는 데이터를 만들었습니다.
따라서 최종 answer는 이렇게 됩니다.
<pre>
<code>
def recommendation():
    movie_x=movie_apply_predict()
    final = movie_x[movie_x['label'] == movie_x['label'].max()].reset_index(drop=True)
    final_1 = final.sample(n=1)
    final_1 = final.to_dict(orient="record")[0]
    answer = '당신에게 {} 장르의 {}감독이 만든 {}을 추천합니다'.format(final_1['장르'],final_1['감독'],
                                                final_1['제목'])
    return answer
</code>
</pre>

### 최종 출력
사용자가 필요하는 것을 input을 이용하여 입력받아 최종적으로 영화를 추천해주는 것입니다.
<pre>
<code>
def get_answer(sentence):
    sentence=question(sentence)
    tags,tag_label,max_class=movie_user_model(sentence)
    answer=recommendation()
    return answer

sentence=input()
get_answer(sentence)
</code>
</pre>
출력 예시는 다음과 같습니다.
<pre>
<code>
엄마 아빠가 보고싶을 때 추천해주세요
'당신에게 코미디 장르의 이철하감독이 만든 오케이 마담을 추천합니다'
</code>
</pre>
## 결론
1. 첫번째,방대한 데이터로 소요시간이 오래 걸렸던 점과 잦은 기획 변경으로 최종 알고리즘 제작에 시간이 오래 걸렸다는 점이 아쉬웠다.
2. 두번째,발표 1시간을 남겨놓고 잦은 에러와 긴 소요시간으로 원인을 찾아 헤맸지만 결국 챗봇 내 영화 카테고리 딥러닝 모델 만드는데 성공했다는 것이다.
3. 세번째,선호하는 키워드가 없을 때 어떻게 대응할지 조금만 고려했다면 완성도있는 결과가 나오지 않았을까 하는 생각이 든다.
