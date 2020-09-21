# 영화 추천

## 기획과정
저는 영화 추천 알고리즘을 만드는데 상당한 시간이 소요되었습니다. 데이터가 많아 처리하는데 시간이 오래 걸렸고 사용자마다 좋아하는 영화가 국가,시간,장르,내용마다 다르기 때문에 어떤 변수를 넣은게 편리한지 여러번 고민하였고 딥러닝 모델을 만드는 과정에서 정확도가 낮아서 다른 방법을 생각해보았습니다. 생각해본 결과 줄거리를 단어,형태소별로 분류해 불용어처리하고 태그를 만들어서 사용자가 선호하는 내용이 담긴 영화를 추천하기로 기획하였습니다. 결론은 사용자가 선호하는 내용을 태그로 만들어 본인이 입력한 것을 태그에 연결해 이에 맞는 영화를 자동으로 추천해주는 것입니다. 

## 데이터 가공
네이버 영화 내 다운로드에 들어가 네이버 시리즈온에서 21250개의 작품을 크롤링하고 영화진흥위원회 db검색을 통해 1980-01-01~2020-09-16일에 개봉한 영화 중 일반영화를 대상으로 필터링을 거쳐 크롤링한 데이터에 취합했습니다. 크롤링 과정에서 감독 및 장르가 누락된 것이 많았기 때문에 객관적인 자료를 찾고자 했습니다.

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

## 중반부
### 태그 제작-1
내용을 단어/형태소 분류하여 태그로 만들고 이를 라벨링해서 앞서 라벨링 실행한 것으로 딥러닝 모델을 제작하려 했지만 한개당 리스트로 들어간 상태이고 문자열로 만들어져있어서 입력받은 단어로 영화를 찾아내는 데 실패했습니다.
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
<pre>
<code>
def data_split():
    tags,tag_label,max_class=movie_user_model(sentence)
    train_data=tags[:1500]
    test_data=tags[1500:]
    train_label=tag_label[:1500]
    test_label=tag_label[1500:]    
    max_words = 200 # 실습에 사용할 단어의 최대 개수
    #num_classes= tag_label 
    return train_data,test_data,train_label,test_label,max_words

def prepare_data(train_data,test_data): 
    train_data,test_data,train_label,test_label,max_words=data_split()
    t = Tokenizer(num_words = max_words) 
    t.fit_on_texts(train_data)
    X_train = t.texts_to_matrix(train_data, mode='count')#단어의 출현 빈도를 표시한 DTM 생성.
    X_test = t.texts_to_matrix(test_data, mode='count')  # 샘플 수 × max_words 크기의 행렬 생성
    return X_train, X_test, train_label, test_label, max_words

def set_data():
    #from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.utils import to_categorical
    tags,tag_label,max_class=movie_user_model(sentence)
    train_data,test_data,train_label,test_label,max_words=data_split()
    X_train, X_test, train_label, test_label, max_word=prepare_data(train_data,test_data)
    y_train = to_categorical(train_label, max_class+1) 
    y_test = to_categorical(test_label, max_class+1) # 원-핫 인코딩
    return X_train,X_test,y_train,y_test,max_class

def fit_and_evaluate():
    X_train,X_test,y_train,y_test,max_class=set_data()
    from tensorflow.keras import models #태그 구분 모델링
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
</code>
</pre>
## 결론
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

