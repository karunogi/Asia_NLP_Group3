from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
from chatapp.ChatFramework.bot import Bot

bot = Bot()

# Create your views here.
context = {}

# def index(request):
#     msg = '박형식 홈페이지'
#     return render(request, 'chatapp/index.html', {'message': msg})

def index(request):
    template = loader.get_template('chatapp/index.html')
    context = { }
    return HttpResponse(template.render(context, request))

def chat_home(request):
    template = loader.get_template('chatapp/chat_home_screen.html')
    context = {
        'login_success' : False,
        'initMessages' : ["반가워요~",
                          "어떤 콘텐츠를 추천해 드릴까요?"]
    }
    return HttpResponse(template.render(context, request))

def popup_chat_home(request):
    template = loader.get_template('chatapp/popup_chat_home_screen.html')
    context = {
        'login_success' : False,
        'initMessages' : ["반가워요~",
                          "어떤 콘텐츠를 추천해 드릴까요?"]
    }
    return HttpResponse(template.render(context, request))

def call_chatbot(request):
    if request.method == 'POST':
        if request.is_ajax():
            # userID = request.POST['user']
            sentence = request.POST['message']
            # logger.debug("question[{}]".format(sentence))
            # answer = clean_up_sentence(sentence)
            # answer = bot.response(sentence, userID)
            # answer = bot.get_answer(sentence, userID)
            answer = bot.get_answer(sentence)
            # logger.debug("answer[{}]".format(answer))
            return HttpResponse(answer)
    return ''
