#!flask/bin/python
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
import json

app = Flask(__name__)

tweets_data = []
x = []
y = []
vectorizer = CountVectorizer(stop_words='english')


def predictFun(input):
    input1 = vectorizer.fit_transform(x)
    input = vectorizer.transform([input])
    model = joblib.load('SocialMediaDepressionPrediction.pk1')
    predict = model.predict(input)
    if predict == 1:
        return "Positive"
    elif predict == 0:
        return "Neutral"
    elif predict == -1:
        return "Negative"
    else:
        return "Nothing"

def start():
    vectorizer = CountVectorizer(stop_words='english')
    tweets_data_path = 'tweetdata.txt'
    tweets_file = open(tweets_data_path, "r")
    for line in tweets_file:
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet)
        except:
            continue
     
    sent = pd.read_excel('output.xlsx')
    for i in range(len(tweets_data)):
        if tweets_data[i]['id']==sent['id'][i]:
            x.append(tweets_data[i]['text'])
            y.append(sent['sentiment'][i])


@app.route('/')
def home():
    
    start()
    return jsonify([{"Status":"OK"}])

@app.route('/prediction', methods=['GET','POST'])
def get_tasks():
    if len(x) == 0:
        start()
    #inputTweet = request.args.get('tweet')
    if request.method == "POST":
        inputTweet = request.form['tweet']
    else:
        return jsonify([{"prediction":"Error"}]) 
    ans = predictFun(inputTweet)
    return jsonify([{"prediction":ans}])

if __name__ == '__main__':
    app.run(debug=True)