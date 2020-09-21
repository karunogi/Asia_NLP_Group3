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

# 음악 추천

## 기획의도
'드라이브갈 때 듣기 좋은 신나는 여름 노래 추천해줘', '카페에서 듣기 좋은 잔잔한 재즈 추천해줘' 등으로 사용자의 취향에 맡는 음악 특성에 따라 음악을 추천해주는 챗봇을 구현하고자 했습니다.

## 코드 설명

### 크롤링
- 해당 데이터는 음악유통사이트인 'Bugs'에서 크롤링 되었습니다.
<pre>
<code>
def tag_crawling():
    tagging_list = []
    for i in range(500):        
        base_url = 'https://music.bugs.co.kr/musicpd'
        params = {'order':'list',
                  'page':i}
        headers = {
            'User-Agent': 'User-Agent'
        }
        print('{}페이지 크롤링 중'.format(i))

        resp = requests.get(base_url,params = params, headers=headers)
        soup = BeautifulSoup(resp.text)

        p_tags = soup.select('p.theme')
        
        for p in p_tags:    
            a_tags = p.select('a')
            sum_tag = ''
            
            for tag in a_tags:
                sum_tag += (tag.text).replace('#','').replace('/',' ') + ' '
            tagging_list.append(sum_tag)

    return tagging_list
</code>
</pre>
- 해당 코드를 통해 10000여개의 태그리스트 데이터인 music_tag.csv 파일을 만들었습니다.
- music_tag.csv는 학습용 데이터로 사용되었습니다.
- 추가로 1700여개의 노래 제목 / 가수 / tag 데이터인 music_title.csv 파일을 만들었습니다.
- music_title.csv 파일은 예측용 데이터로 사용되었습니다.
- 예측오류를 방지하고자, 비교적 특징적이지 않은 국내, 해외, 국내외, 가요 등의 데이터는 삭제하였습니다.


### music.py 
<pre>
<code>
def music_make_user_model(sentence):
    
    unique_x = music_get_unique_x('chatapp/ChatFramework/data/music/music_tag.csv')
    favor_label = np.zeros((len(unique_x)))
    sentence = music_preprocessing_sentence(sentence)

    for word in sentence:
        for i in range(len(unique_x)):
            if word in unique_x[i]:
                favor_label[i] += 1 

    max_class = int(np.max(favor_label))
    
    return unique_x, favor_label, max_class
</code>
</pre>
- 전처리로 키워드만 남기고, 각각 tag list의 row마다 해당 키워드가 있을경우 1을 추가합니다.
- 높은 label값을 가질수록 높은 선호도를 나타냅니다.

<pre>
<code>
def music_fit_and_evaluate():
    
    X_train, X_test, y_train, y_test, max_class = music_set_data()
    max_words = music_prepare_data()[4]
    
    model = Sequential() 
    
    model.add(Dense(256, input_shape = (max_words,), activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(max_class+1, activation = 'softmax'))
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer= 'rmsprop',
                 metrics =['accuracy'])
    
    model.fit(X_train,y_train, batch_size = 128, epochs = 6, verbose = 1, validation_split = 0.1)
    results = model.evaluate(X_test, y_test, batch_size = 128, verbose = 0)
    
    # 정확도 확인
    print(results[1])
    
    return model
</code>
</pre>
- 키워드 수에 따라 label 값이 달라지므로, 다중분류를 진행합니다.
- 사용자의 대답에 맞게 모델을 설정하므로, 빠른 응답을 위해 epochs값을 10미만으로 설정했습니다.
- max_words는 500으로 설정하였습니다.

<pre>
<code>
def music_apply_predict():
    
    model = music_fit_and_evaluate()
    return_data, return_feature = music_prediction()    
    predict_value = model.predict(return_feature)
    
    predict_label = []
    
    for i in range(len(predict_value)):
        predict_label.append(np.argmax(predict_value[i]))
        
    return_data['label'] = predict_label
    
    return return_data
</code>
</pre>
- 약 1700개의 예측 데이터에 사용자 모델을 적용해 Y값을 예측합니다.
- 이후 코드에서 가장 높은 label값을 가진 데이터 중에서 random한 값을 출력해 사용자에게 추천해줍니다.





