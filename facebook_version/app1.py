import random
from flask import Flask, request
from pymessenger.bot import Bot
import re
import time
from datetime import datetime
import numpy as np
import pickle
import os
from nltk.stem import WordNetLemmatizer
import copy
import requests
cwd = os.getcwd()



"""
****************************************************************
FLASK AP
****************************************************************
"""

#initialising the flask application and connecting to facebook app
#no need to change anything unless assigned facebook page is changed (generate new access token)
app = Flask(__name__)
#ACCESS_TOKEN = 'EAAQhZA5wQrngBAHB3Uv9K3Jh23nukICyyS5d32H4PC9OYegzT4AZC54wXEsvNTAFxZAP9IZB97lExu9zcqYi8ZBIAvNsoXfRLEOFVv6qcuBroCWnxSurFK0P4EqYAjKQwUxe9292t7KmU9vueNXhkOABqqbXxEfelkjnv1yCJnq3C717tIFIh'
ACCESS_TOKEN = "EAACJjQRK0ucBAJ4tXZCue2PmF3IhZCZAjCPlCB9qTFeCU3hMnEvMI7ZCTdl0MTU80VZAiXAelb16jvZBOyDzNikImyTZCbXNqeoApC9WyLKrd7303ptPZAC8BrkGp2kDZAsJMf2E3g9m9gxOb6ZC36IcUlHRgaXod2NXmSAaXq1ZAZBrfgZDZD"
VERIFY_TOKEN = 'UNIQE_TOKEN'
bot = Bot(ACCESS_TOKEN)

#storing user IDs - needed to send welcomemessage
user_ids = []
#storing user IDs again to proceed with chat after they provided their prolific ID
prolific_ids = []
# stores the depth1 arguments for each user and deletes the ones used - once the list is empty - chat is ended
user_ids_dic = {}
# stores the chat logs for each user
chat_logs = {}
# stores start and end time for each user
timestamps = {}

pickle_path = cwd + "/chatlogs6/"

#concern dics for each user so we can delete used arguments from them
con_dic_users = {}






def typing_on(user_id):
    data = {
        "recipient": {"id": user_id},
        "sender_action":"typing_on"


    }
    resp = requests.post("https://graph.facebook.com/v2.6/me/messages?access_token=" + ACCESS_TOKEN, json=data)





"""
****************************************************************
PREPROCESSING & SIMILARITY MEASUREMENT
****************************************************************
"""



# stopwords and preprocessing step
stop_words_file = 'SmartStoplist.txt'

stop_words = []

with open(stop_words_file, "r") as f:
    for line in f:
        stop_words.extend(line.split())
#print(stop_words)

#stop_words = stop_words +  ['vaccine', 'people', 'covid', 'virus', 'vaccinated', 'vaccinate', 'vaccination']
#df_lisa



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




"""
****************************************************************
CONCERN CLASSIFICATION
****************************************************************
"""


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


filename = 'finalized_model.sav'
filename1 = 'finalized_model_agreement.sav'
filename2 = "transformer.sav"
filename3 = "transformer_agreement.sav"

# load the model from disk
concern_model = pickle.load(open(filename, 'rb'))
concern_model_agreement = pickle.load(open(filename1, 'rb'))
transformer = pickle.load(open(filename2, 'rb'))
transformer_agreement = pickle.load(open(filename3, 'rb'))

"""
****************************************************************
NEEDED DFS & DICS (of pro  args from graph)
****************************************************************
"""


with open('concern_dic.pickle', 'rb') as handle:
    concern_dic = pickle.load(handle)

with open('id_dic.pickle', 'rb') as handle:
    id_dic = pickle.load(handle)



#pro_args_depth1_ = ["testing bot2", "testingbot1", "testingbot0"]
"""
****************************************************************
CONCERN CLASSIFICATION AND ARGUMENT ID RETURN
****************************************************************
"""

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



""" ***************************************************+

                THE APP

**************************************************** """



# Importing standard route and two requst types: GET and POST.
# We will receive messages that Facebook sends our bot at this endpoint
@app.route('/', methods=['GET', 'POST'])
def receive_message():
    if request.method == 'GET':
        # Before allowing people to message your bot Facebook has implemented a verify token
        # that confirms all requests that your bot receives came from Facebook.
        token_sent = request.args.get("hub.verify_token")

        return verify_fb_token(token_sent)
    # If the request was not GET, it  must be POSTand we can just proceed with sending a message
    # back to user
    else:
        #print(con_dic_users)
        output = request.get_json()
        recipient_id = output['entry'][0]['messaging'][0]['sender']['id']  #unicode, should i typecast it into string or int? lets see...
        timestamp_ = output['entry'][0]['messaging'][0]['timestamp']
        timestamp = int(timestamp_) /1000
        dt_object = datetime.fromtimestamp(timestamp)

        str_dt = str(dt_object)
        mes_time = str_dt.split()[1]


        try:
            user_mes = output['entry'][0]['messaging'][0]['message']['text']


        except Exception as e:
            print(e)
            response_sent_text = "Please send a reply in text format :)"
            typing_on(recipient_id)
            time.sleep(random.randint(5, 8))
            send_message(recipient_id, response_sent_text)
            return "ok"

        stop = user_mes.lower()


        if recipient_id not in user_ids:
            user_ids_dic[recipient_id] = recipient_id
            con_dic_users[recipient_id] = copy.deepcopy(concern_dic)


            chat_logs[recipient_id] = []
            timestamps[recipient_id] = [mes_time]
            response_sent_text = "Hey! Before we start, please provide your prolific ID." #" and click on the following link (it contains the google form to fill out after the chat)."

            user_ids.append(recipient_id)
            typing_on(recipient_id)
            time.sleep(random.randint(5, 8))
            send_message(recipient_id, response_sent_text)
            return "oK"

        elif recipient_id not in prolific_ids:

            response_sent_text = "Great, thanks. Don't forget, you can always let me know you want to end the chat by typing 'quit'.\n \nSo tell me, why would you not consider getting a COVID-19 vaccine, should one be developed?"
            chat_logs[recipient_id].append(user_mes) #important! prolific id
            typing_on(recipient_id)
            time.sleep(random.randint(5, 8))
            send_message(recipient_id, response_sent_text)
            prolific_ids.append(recipient_id)
            return "ok"

        elif stop == "quit":
            response_sent_text = "You are ending the chat. It has been nice talking with you. I hope you think about my points and do consider taking the vaccine if it becomes available. Please click on the following link to complete the study: https://columbia.az1.qualtrics.com/jfe/form/SV_6YljOc37mILSloh"
            typing_on(recipient_id)
            time.sleep(random.randint(5, 8))
            send_message(recipient_id, response_sent_text)

            log_mes= "User: " + user_mes
            chat_logs[recipient_id].append(log_mes)

            timestamps[recipient_id].append(mes_time)
            recipient_logs = chat_logs[recipient_id]
            recipient_times = timestamps[recipient_id]

            data = recipient_logs + recipient_times

            pickle_file_name = pickle_path + recipient_id + ".pickle"
            with open(pickle_file_name, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return "ok"


        else:
            concern, response_id = return_arg_and_concern(user_mes, con_dic_users[recipient_id])


            if concern == 'disagree':
                disag = ['Why?', 'Why not?']
                response_sent_text = random.choice(disag)
                typing_on(recipient_id)
                time.sleep(random.randint(1, 3))
                send_message(recipient_id, response_sent_text)
                return "ok"

            if response_id == 0:
                response_sent_text = "I'll stop here. It has been nice talking with you. I hope you think about my points and do consider taking the vaccine if it becomes available. Please click on the following link to complete the study: https://columbia.az1.qualtrics.com/jfe/form/SV_6YljOc37mILSloh"
                typing_on(recipient_id)
                time.sleep(random.randint(5, 8))
                send_message(recipient_id, response_sent_text)
                timestamps[recipient_id].append(mes_time)
                #adding last argument where no match was found to chatlog
                log_mes= "User: " + user_mes
                chat_logs[recipient_id].append(log_mes)

                recipient_logs = chat_logs[recipient_id]
                recipient_times = timestamps[recipient_id]

                data = recipient_logs + recipient_times

                pickle_file_name = pickle_path + recipient_id + ".pickle"
                with open(pickle_file_name, 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

                return "ok"


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
                chat_logs[recipient_id].append(log_mes)
                #retrieving the concern dictionary for the user
                con_dic_user = con_dic_users[recipient_id]
                response_sent_text = id_dic[response_id][0]

                con_dic_user[concern] = con_dic_user[concern][1:]
                con_dic_users[recipient_id] = con_dic_user


                chatbot_response = "CB: " + str(response_id)
                chat_logs[recipient_id].append(chatbot_response)

                if add_a == True:
                    ag = random.choice(add_agree)
                    response_sent_text = ag + response_sent_text
                if add_d == True:
                    ag = random.choice(add_default)
                    response_sent_text = ag + response_sent_text

                typing_on(recipient_id)
                time.sleep(random.randint(5, 8))
                send_message(recipient_id, response_sent_text)
                return "ok"




    return "Message Processed"


def verify_fb_token(token_sent):
    # take token sent by Facebook and verify it matches the verify token you sent
    # if they match, allow the request, else return an error
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'


# Uses PyMessenger to send response to the user
def send_message(recipient_id, response):
    # sends user the text message provided via input response parameter
    bot.send_text_message(recipient_id, response)
    #return "success"

# Add description here about this if statement.
if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=5000, threaded=True)
    #app.run()
