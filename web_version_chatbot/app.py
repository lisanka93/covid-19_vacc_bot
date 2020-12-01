from flask import Flask, render_template, request
from flask import session
import uuid

import random

import re
import time
from datetime import datetime
import numpy as np
import pickle
import os
from nltk.stem import WordNetLemmatizer
import copy

cwd = os.getcwd()

pickle_path = cwd + "/chatlogs_web/"

filename = 'finalized_model.sav'
filename1 = 'finalized_model_agreement.sav'
filename2 = "transformer.sav"
filename3 = "transformer_agreement.sav"

# load the model from disk
concern_model = pickle.load(open(filename, 'rb'))
concern_model_agreement = pickle.load(open(filename1, 'rb'))
transformer = pickle.load(open(filename2, 'rb'))
transformer_agreement = pickle.load(open(filename3, 'rb'))

stop_words_file = 'SmartStoplist.txt'

stop_words = []

with open(stop_words_file, "r") as f:
    for line in f:
        stop_words.extend(line.split())

with open('concern_dic.pickle', 'rb') as handle:
    concern_dic = pickle.load(handle)

with open('id_dic.pickle', 'rb') as handle:
    id_dic = pickle.load(handle)



def preprocess(raw_text):


    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split
    words = letters_only_text.lower().split()

    # remove stopwords
    cleaned_words = []

    for word in words:
        if word not in stop_words:
            cleaned_words.append(word)

    lemmatised_words = []
    lemmatizer = WordNetLemmatizer()
    for word in cleaned_words:
        word = lemmatizer.lemmatize(word)
        lemmatised_words.append(word)

    return lemmatised_words

def get_top_k_predictions_(model,X_test,k):

    # get probabilities instead of predicted labels, since we want to collect top 3
    probs = model.predict_proba(X_test)

    prob_list = list(probs[0])
    prob_list.sort(reverse=True)

    # GET TOP K PREDICTIONS BY PROB - note these are just index
    best_n = np.argsort(probs, axis=1)[:,-k:]

    # GET CATEGORY OF PREDICTIONS
    preds=[[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]

    preds=[ item[::-1] for item in preds]
    preds = preds[0]

    pred_prob = prob_list[:k]

    return preds, pred_prob

def return_arg_and_concern(user_mes, concern_dic ): #, prev_cb_responses):

    response_id = 0
    user_mes = user_mes.lower()

    # checking first whether person agrees - then no need to check for counterarg

    disagree = ['dont', "don't", "not", 'lie', 'disagree', 'no']


    bool_disagree = any(x in user_mes.split() for x in disagree)



    # if user agrees we use a default argument
    if bool_disagree == True and len(user_mes.split()) < 7:
        print('disagreement.')
        concern = 'disagree'
        possible_responses = concern_dic['default']
        return (concern, response_id)



    # now lets preprocess and look for a match in the KB
    message_prep = preprocess(user_mes)
    message_sen = sen = ' '.join(message_prep) # as string for classifier
    message_features = transformer_agreement.transform([message_sen])
    concerns_, preds = get_top_k_predictions_(concern_model_agreement, message_features, 2)
    #print(concerns_, preds)
    print(concerns_)

    """  check if agreement   """
    if concerns_[0] == 'agree' and preds[0] > 0.7 and len(user_mes.split()) < 13:
        concern = 'agree'

        possible_responses = concern_dic['default']
        if possible_responses == []:
            concern = "no concern"
            return (concern, response_id)
        else:
            response_id = possible_responses[0]
            return (concern, response_id)

    message_features = transformer.transform([message_sen])
    concerns_, preds = get_top_k_predictions_(concern_model, message_features, 2)
    #print(concerns_, preds)
    print(concerns_)
    """
    NO CONCERN - SO RETURN DEFAULT ARGUMENT OR IF THATS EMPTY RESPONSE ID = 0
    """
    if preds[0] <= 0.4:
        concern = 'default'
        possible_responses = concern_dic[concern]
        if possible_responses == []:
            concern = "no concern"
            return (concern, response_id)
        else:
            response_id = possible_responses[0]
            return (concern, response_id)

    #TWO CONCERN - CHECK FOR FIRST CONCERN AND RETURN IT, IF EMPTY CHECK FOR SECOND, IF AGAIN FAIL CHECK DEFAULT

    elif preds[0] > 0.4 and preds[0] <= 0.5:

            concern_1 = concerns_[0]
            concern_2 = concerns_[1]

            possible_responses_1 = concern_dic[concern_1]
            possible_responses_2 = concern_dic[concern_2]
            possible_responses_3 = concern_dic['default']
            if possible_responses_1 == []:
                if possible_responses_2 != []:
                    response_id = possible_responses_2[0]
                    return (concern_2, response_id)
                else:
                    if possible_responses_3 != []:
                        response_id = possible_responses_3[0]
                        return ('default', response_id)
                    else:
                        concern = "no concern"
                        return (concern, response_id)

            else:
                response_id = possible_responses_1[0]
                return (concern_1, response_id)



    else:
        concern = concerns_[0]

        possible_responses_1 = concern_dic[concern]
        possible_responses_2 = concern_dic['default']

        if possible_responses_1 == []:
            if possible_responses_2 != []:
                response_id = possible_responses_2[0]
                return('default', response_id)
            else:
                concern = "no concern"
                return (concern, response_id)
        else:
            response_id = possible_responses_1[0]
            return (concern, response_id)





app = Flask(__name__)
app.static_folder = 'static'



app.secret_key = "coronavirus"

@app.before_request
def make_session_permanent():
    session.permanent = False

@app.route("/")
def home():
    print(session)
    if "user_id" not in session:
        print('new user')
        # use this session["user_id"], try printing
        user_id = str(uuid.uuid1())
        session["user_id"] = uuid.uuid1()
        session['bot_replies'] = ['test1', 'test2', 'test3']
        session['concern_dic'] = copy.deepcopy(concern_dic)
        session['chatlogs'] = []



    return render_template("index.html", user_id=session["user_id"])

@app.route("/get")
def get_bot_response():

    user_mes = request.args.get('msg')
    stop = user_mes.lower()

    if "prolific_id" not in session:
        session['prolific_id'] = user_mes
        bot_reply = "Great, thanks. Don't forget, you can always let me know you want to end the chat by typing 'quit'.\n \nSo tell me, why would you not consider getting a COVID-19 vaccine, should one be developed?"
        session['chatlogs'].append(user_mes)
        #time.sleep(random.randint(5, 10))
        return bot_reply

    elif stop == 'quit':
        bot_reply = "You are ending the chat. It has been nice talking with you. I hope you think about my points and do consider taking the vaccine if it becomes available. Please return to the survey. Good bye! "

        log_mes= "User: " + user_mes
        session['chatlogs'].append(log_mes)

        pickle_file_name = pickle_path + str(session["user_id"]) + ".pickle"

        with open(pickle_file_name, 'wb') as handle:
            pickle.dump(session["chatlogs"], handle, protocol=pickle.HIGHEST_PROTOCOL)
        #time.sleep(random.randint(5, 10))
        return bot_reply

    else:
        concern, response_id = return_arg_and_concern(user_mes, session['concern_dic'])

        if concern == 'disagree':
            disag = ['Why?', 'Why not?']
            bot_reply = random.choice(disag)
            #time.sleep(random.randint(3, 5))
            return bot_reply

        if response_id == 0:
            bot_reply = "I'll stop here. It has been nice talking with you. I hope you think about my points and do consider taking the vaccine if it becomes available. Please return to the survey. Good bye! "

            #adding last argument where no match was found to chatlog
            log_mes= "User: " + user_mes
            session['chatlogs'].append(log_mes)

            pickle_file_name = pickle_path + str(session["user_id"]) + ".pickle"
            with open(pickle_file_name, 'wb') as handle:
                pickle.dump(session["chatlogs"], handle, protocol=pickle.HIGHEST_PROTOCOL)
            #time.sleep(random.randint(5, 10))
            return bot_reply

        else:
            add_a = False
            add_d = False
            add_default = ["Ok, but ", 'I understand, however, ', 'But have you considered that ', 'Nevertheless, ', 'Nonetheless, ', 'Despite that, ']
            add_agree = ['Thanks. Also, have you considered that ', 'I\'m glad. Also, ', 'I\'m happy you agree. Don\'t you also think that ']
            #check whether agree or DEFAULT
            print(concern)
            if concern == 'default':
                add_d = True

            if concern == 'agree':
                add_a = True
                concern = 'default'



            log_mes= "User: " + user_mes
            session['chatlogs'].append(log_mes)
            #retrieving the concern dictionary for the user

            con_dic_user = session['concern_dic']
            bot_reply = id_dic[response_id][0]

            con_dic_user[concern] = con_dic_user[concern][1:]
            session['concern_dic'] = con_dic_user


            chatbot_response = "CB: " + str(response_id)
            session['chatlogs'].append(chatbot_response)

            if add_a == True:
                ag = random.choice(add_agree)
                bot_reply = ag + bot_reply
            if add_d == True:
                ag = random.choice(add_default)
                bot_reply = ag + bot_reply

            #time.sleep(random.randint(5, 10))
            return bot_reply




    return bot_reply

if __name__ == "__main__":
    app.run(debug=True)
