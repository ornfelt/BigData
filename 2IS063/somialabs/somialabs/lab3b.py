# -*- coding: utf-8 -*-
import pandas as pd
from ipywidgets import (
    interact,
    interactive,
    fixed,
    interact_manual,
    widgets
)
import re
import textmining as tm
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer
)
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
nltk.download('punkt')
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import numpy as np
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
import pandas as pd
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def view_20_pos_tweets():
    fp = open("pos_tweets.txt")
    for line in fp.readlines()[:20]:
        print(line)
        
def view_20_neg_tweets():
    fp = open("neg_tweets.txt")
    for line in fp.readlines()[:20]:
        print(line)
        
def create_tdm_from_tweets():
    text_pos = []
    labels_pos = []
    with open("pos_tweets.txt") as f:
        for i in f: 
            text_pos.append(i) 
            labels_pos.append('pos')

    text_neg = []
    labels_neg = []
    with open("neg_tweets.txt") as f:
        for i in f: 
            text_neg.append(i)
            labels_neg.append('neg')
    training_text = text_pos[:int((.8)*len(text_pos))] + text_neg[:int((.8)*len(text_neg))]
    training_labels = labels_pos[:int((.8)*len(labels_pos))] + labels_neg[:int((.8)*len(labels_neg))]
    test_text = text_pos[int((.8)*len(text_pos)):] + text_neg[int((.8)*len(text_neg)):]
    test_labels = labels_pos[int((.8)*len(labels_pos)):] + labels_neg[int((.8)*len(labels_neg)):]
    vectorizer = CountVectorizer(
        analyzer = 'word',
        lowercase = True,
        max_features = 85
    )
    features = vectorizer.fit_transform(
    training_text + test_text)
    features_nd = features.toarray() # for easy use
    return pd.DataFrame(features_nd, columns=vectorizer.vocabulary_)

def view_log_regression_accuracy():
    text_pos = []
    labels_pos = []
    with open("pos_tweets.txt") as f:
        for i in f: 
            text_pos.append(i) 
            labels_pos.append('pos')

    text_neg = []
    labels_neg = []
    with open("neg_tweets.txt") as f:
        for i in f: 
            text_neg.append(i)
            labels_neg.append('neg')
    training_text = text_pos[:int((.8)*len(text_pos))] + text_neg[:int((.8)*len(text_neg))]
    training_labels = labels_pos[:int((.8)*len(labels_pos))] + labels_neg[:int((.8)*len(labels_neg))]
    test_text = text_pos[int((.8)*len(text_pos)):] + text_neg[int((.8)*len(text_neg)):]
    test_labels = labels_pos[int((.8)*len(labels_pos)):] + labels_neg[int((.8)*len(labels_neg)):]
    vectorizer = CountVectorizer(
        analyzer = 'word',
        lowercase = True,
        max_features = 85
    )
    features = vectorizer.fit_transform(
    training_text + test_text)
    features_nd = features.toarray() # for easy use
    from sklearn.model_selection import train_test_split 
    X_train, X_test, y_train, y_test  = train_test_split(
        features_nd[0:len(training_text)], 
        training_labels,
        train_size=0.80, 
        random_state=1234)
    log_model = LogisticRegression()
    log_model = log_model.fit(X=X_train, y=y_train)
    test_pred = log_model.predict(X_test)
    spl = random.sample(range(len(test_pred)), 10)
    for text, sentiment in zip(test_text, test_pred[spl]):
        print(sentiment, text)
        

def view_log_regression_accuracy_score():
    text_pos = []
    labels_pos = []
    with open("pos_tweets.txt") as f:
        for i in f: 
            text_pos.append(i) 
            labels_pos.append('pos')

    text_neg = []
    labels_neg = []
    with open("neg_tweets.txt") as f:
        for i in f: 
            text_neg.append(i)
            labels_neg.append('neg')
    training_text = text_pos[:int((.8)*len(text_pos))] + text_neg[:int((.8)*len(text_neg))]
    training_labels = labels_pos[:int((.8)*len(labels_pos))] + labels_neg[:int((.8)*len(labels_neg))]
    test_text = text_pos[int((.8)*len(text_pos)):] + text_neg[int((.8)*len(text_neg)):]
    test_labels = labels_pos[int((.8)*len(labels_pos)):] + labels_neg[int((.8)*len(labels_neg)):]
    vectorizer = CountVectorizer(
        analyzer = 'word',
        lowercase = True,
        max_features = 85
    )
    features = vectorizer.fit_transform(
    training_text + test_text)
    features_nd = features.toarray() # for easy use
    from sklearn.model_selection import train_test_split 
    X_train, X_test, y_train, y_test  = train_test_split(
        features_nd[0:len(training_text)], 
        training_labels,
        train_size=0.80, 
        random_state=1234)
    log_model = LogisticRegression()
    log_model = log_model.fit(X=X_train, y=y_train)
    test_pred = log_model.predict(X_test)
    spl = random.sample(range(len(test_pred)), 10)
    print(accuracy_score(y_test, test_pred))
    
def train_naive_bayes_and_show_most_informative_features():
    def format_sentence(sent):
        return({word: True for word in nltk.word_tokenize(sent)})
    pos = []
    with open("pos_tweets.txt") as f:
        for i in f: 
            pos.append([format_sentence(i), 'pos'])
    neg = []
    with open("neg_tweets.txt") as f:
        for i in f: 
            neg.append([format_sentence(i), 'neg'])
    training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
    test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]
    classifier = NaiveBayesClassifier.train(training)
    return classifier.show_most_informative_features()

def interact_naive_bayes():
    def format_sentence(sent):
        return({word: True for word in nltk.word_tokenize(sent)})
    pos = []
    with open("pos_tweets.txt") as f:
        for i in f: 
            pos.append([format_sentence(i), 'pos'])
    neg = []
    with open("neg_tweets.txt") as f:
        for i in f: 
            neg.append([format_sentence(i), 'neg'])
    training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
    test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]
    classifier = NaiveBayesClassifier.train(training)
    def classify(text):
        print(classifier.classify(format_sentence(text)))
    interact(classify, text=widgets.Text(
        value='Uppsala University is great!',
        description='Text to clasify:',
        disabled=False
))
    
print("Social Media and Digital Methods Lab 3b initialized... OK!")