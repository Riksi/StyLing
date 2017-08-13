from flask import Flask, request, session, url_for, redirect, \
     render_template, abort, g, flash, _app_ctx_stack, jsonify
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, Embedding
from keras.optimizers import RMSprop, Adam
import pickle
import requests
import json
from collections import Counter


app = Flask(__name__)

MAX_LEN = 30 
EMBED_DIM = 300
PERPLEXITY = 174
WIDTHS = [0,25,75,100]
VOCAB_SIZE = 10000



all_vectors = pickle.load(open('fastTextHindi.vec','rb'))

vec_dict = pickle.load(open('vec_pickle.pkl','rb'))

word_indices = vec_dict['word_indices']

indices_word = vec_dict['indices_word']

vector_rows = vec_dict['vector_rows']

vectors = all_vectors[vector_rows]
vectors = np.concatenate([np.zeros(vectors[0:1].shape), vectors],axis=0)

del all_vectors


print("word_indices", type(word_indices), "length:",len(word_indices) )
print("indices_words", type(indices_word), "length", len(indices_word))

print('Num vectors', len(vectors))

def get_word_vec(word):
    return vectors[word_indices[word]]

#Using equations 8 and 9 in http://web.stanford.edu/class/cs224n/lecture_notes/cs224n-2017-notes5.pdf
#for 'corpuses' of sizes 1 to length of input text


#The idea is that our model is trained on the 'right' corpus so a low
#perplexity would suggest that this corpus is right whilst a high one would suggest
#otherwise

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response



def build_model():
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, input_shape=(MAX_LEN, EMBED_DIM)) )
    model.add(Dropout(0.2))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(VOCAB_SIZE))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=RMSprop())
    model.load_weights('weights_hindi_epoch_11.pkl')
    return model


def get_metrics(tokens):
    model = build_model()
    sent = [get_word_vec(token) for token in tokens[:30]]
    losses = []
    probs = []
    for token in tokens[30:]:
        out = np.zeros((1, VOCAB_SIZE), dtype=np.int32)
        ind = word_indices[token]
        out[0,ind] = 1
        x = np.expand_dims(sent,axis=0)
        loss = model.evaluate(x,out,verbose=False)
        preds = model.predict(x,verbose=False)
        probs.append(preds[0, ind])
        sent.append(get_word_vec(token))
        del sent[0]
        if len(losses) > 0:
            loss = losses[-1]+loss
        losses.append(loss)
    perplexities = np.exp(losses/np.arange(1,len(losses)+1))
    return probs, perplexities


@app.route('/')
def display():
    return render_template('styLing.html')

@app.route('/translation' ,methods=['POST'])
def translate():
    print('Getting translation')
    text = request.form.get('text')
    url = "https://glosbe.com/gapi/translate?from=hin&dest=eng&format=json&phrase=%s"%text
    result = requests.get(url).json()
    
    if len(result['tuc']) and result['tuc'][0].get('phrase'):
        trans =  result['tuc'][0].get('phrase').get('text')
    else:
        trans = "Not available"
    return jsonify({'trans':trans})



@app.route('/', methods=['POST'])
def evaluate():
    text = request.form.get('text')
    tokens = text.strip().split()
    if (len(tokens) < 100):
        return jsonify({'message': 'Please enter at least 100 words'})
    pr,px = get_metrics(tokens)
    # pr = np.random.random(len(tokens)-30)
    # px = [0,200]
    #Reverse for purposes of display
    pr2 = np.zeros_like(pr)
    maxmap = dict(zip(np.argsort(-np.array(pr)),np.arange(len(pr))))
    sort_asc = np.sort(pr)
    for i,j in maxmap.items():
        pr2[i] = sort_asc[j]

    width = 100*PERPLEXITY/float(px[-1])
    return jsonify(
        {'probs':[float(i) for i in pr2], 
        'orig_probs':[float(i) for i in pr],
        'tokens': tokens, 
        'width':min(100,width)}
    )


app.debug = True
app.run(port=5555, host='0.0.0.0')
