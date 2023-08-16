# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from ipywidgets import (
    interact,
    interactive,
    fixed,
    interact_manual,
    widgets
)
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from langdetect import detect
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()
from nltk.corpus import stopwords
from itertools import chain
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.models import Phrases
from gensim import corpora
from gensim import models
import numpy as np
from nltk.corpus import stopwords


def view_bbc_corpus_head():
    data = pd.read_csv('articles_bbc_2018_01_30.csv')
    for article in data.articles:
        print(article)
        
def view_bbc_corpus_by_lang(lang='en'):
    data = pd.read_csv('articles_bbc_2018_01_30.csv')
    for article in data.articles[data.lang == lang]:
        print(article)
        
def interact_bbc_corpus_by_lang():
    data = pd.read_csv('articles_bbc_2018_01_30.csv')
    interact(view_bbc_corpus_by_lang, lang=widgets.Dropdown(
        options=data.lang.dropna().unique(),
        value='en',
        description='Language:',
        disabled=False
    ))
    
def preprocess_to_pos_tagging():
    data = pd.read_csv('articles_bbc_2018_01_30.csv')
    data = data.loc[data.lang=='en']
    print("Tokenizing sentences...")
    data['sentences'] = data['articles'].progress_map(sent_tokenize)
    print("Tokenizing words...")
    data['tokens_sentences'] = data['sentences'].progress_map(lambda sentences: [word_tokenize(sentence) for sentence in sentences])
    print("Tokenizing POS tagging...")
    data['POS_tokens'] = data['tokens_sentences'].progress_map(lambda tokens_sentences: [pos_tag(tokens) for tokens in tokens_sentences])
    return data

    
def preprocess_to_pos_tagging_and_get_first():
    data = preprocess_to_pos_tagging()
    print("Selecting first article and POS tags...")
    return pd.DataFrame(data.POS_tokens.head(1).tolist()[0][3], columns=["Word", "Tag"])

def preprocess_to_lemmatization():
    data = preprocess_to_pos_tagging()
    print("Lemmatizing words...")
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''
    lemmatizer = WordNetLemmatizer()
    data['tokens_sentences_lemmatized'] = data.POS_tokens.progress_map(
        lambda list_tokens_POS: [
            [
                lemmatizer.lemmatize(el[0], get_wordnet_pos(el[1])) 
                if get_wordnet_pos(el[1]) != '' else el[0] for el in tokens_POS
            ] 
            for tokens_POS in list_tokens_POS
        ]
    )
    return data

def preprocess_to_lemmatization_and_get_first():
    data = preprocess_to_lemmatization()
    print("Selecting first article and lemmatized list...")
    return pd.DataFrame(data.tokens_sentences_lemmatized.head(1).tolist()[0][3], columns=["Word"])

def interact_lemmatization():
    def get_wordnet_pos(treebank_tag):
            if treebank_tag.startswith('J'):
                return wordnet.ADJ
            elif treebank_tag.startswith('V'):
                return wordnet.VERB
            elif treebank_tag.startswith('N'):
                return wordnet.NOUN
            elif treebank_tag.startswith('R'):
                return wordnet.ADV
            else:
                return ''
    def lemmatize(text):
        lemmatizer = WordNetLemmatizer()
        return print("Lemmatized text:", " ".join(list(map(lambda list_tokens_POS: [
                [
                    lemmatizer.lemmatize(el[0], get_wordnet_pos(el[1])) 
                    if get_wordnet_pos(el[1]) != '' else el[0] for el in tokens_POS
                ] 
                for tokens_POS in list_tokens_POS
            ], [[pos_tag(tokens) for tokens in [word_tokenize(text)]]]))[0][0]))
    _ = interact(lemmatize, text="type something here")

def gensim_on_bbc_docs():
    data = preprocess_to_lemmatization()
    stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can']
    stopwords_other = ['one', 'mr', 'bbc', 'image', 'getty', 'de', 'en', 'caption', 'also', 'copyright', 'something']
    my_stopwords = stopwords.words('english') + stopwords_verbs + stopwords_other
    data['tokens'] = data['tokens_sentences_lemmatized'].map(lambda sentences: list(chain.from_iterable(sentences)))
    data['tokens'] = data['tokens'].map(lambda tokens: [token.lower() for token in tokens if token.isalpha() 
                                                        and token.lower() not in my_stopwords and len(token)>1])
    tokens = data['tokens'].tolist()
    bigram_model = Phrases(tokens)
    trigram_model = Phrases(bigram_model[tokens], min_count=1)
    tokens = list(trigram_model[bigram_model[tokens]])
    dictionary_LDA = corpora.Dictionary(tokens)
    dictionary_LDA.filter_extremes(no_below=3)
    corpus = [dictionary_LDA.doc2bow(tok) for tok in tokens]
    np.random.seed(123456)
    num_topics = 20
    lda_model = models.LdaModel(corpus, num_topics=num_topics, \
                                      id2word=dictionary_LDA, \
                                      passes=4, alpha=[0.01]*num_topics, \
                                      eta=[0.01]*len(dictionary_LDA.keys()))
    for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=20):
        print(str(i)+": "+ topic)
        print()
    print("Article text:")
    print(data.articles.loc[0][:500])
    print("Topic number / weight")
    print(lda_model[corpus[0]])
    
def interact_gensim_on_bbc_docs():
    data = preprocess_to_lemmatization()
    stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can']
    stopwords_other = ['one', 'mr', 'bbc', 'image', 'getty', 'de', 'en', 'caption', 'also', 'copyright', 'something']
    my_stopwords = stopwords.words('english') + stopwords_verbs + stopwords_other
    data['tokens'] = data['tokens_sentences_lemmatized'].map(lambda sentences: list(chain.from_iterable(sentences)))
    data['tokens'] = data['tokens'].map(lambda tokens: [token.lower() for token in tokens if token.isalpha() 
                                                        and token.lower() not in my_stopwords and len(token)>1])
    tokens = data['tokens'].tolist()
    bigram_model = Phrases(tokens)
    trigram_model = Phrases(bigram_model[tokens], min_count=1)
    tokens = list(trigram_model[bigram_model[tokens]])
    dictionary_LDA = corpora.Dictionary(tokens)
    dictionary_LDA.filter_extremes(no_below=3)
    corpus = [dictionary_LDA.doc2bow(tok) for tok in tokens]
    np.random.seed(123456)
    num_topics = 20
    lda_model = models.LdaModel(corpus, num_topics=num_topics, \
                                      id2word=dictionary_LDA, \
                                      passes=4, alpha=[0.01]*num_topics, \
                                      eta=[0.01]*len(dictionary_LDA.keys()))
    def get_article(article_num):
        print("Article text:")
        print(data.articles.loc[article_num][:2500])
        print("Topic number / weight")
        print(lda_model[corpus[article_num]])
    for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=20):
        print(str(i)+": "+ topic)
        print()
    interact(get_article, article_num=widgets.IntSlider())
    
def visualize_topic_distribution():
    data = preprocess_to_lemmatization()
    def topics_document_to_dataframe(topics_document, num_topics):
        res = pd.DataFrame(columns=range(num_topics))
        for topic_weight in topics_document:
            res.loc[0, topic_weight[0]] = topic_weight[1]
        return res
    stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can']
    stopwords_other = ['one', 'mr', 'bbc', 'image', 'getty', 'de', 'en', 'caption', 'also', 'copyright', 'something']
    my_stopwords = stopwords.words('english') + stopwords_verbs + stopwords_other
    data['tokens'] = data['tokens_sentences_lemmatized'].map(lambda sentences: list(chain.from_iterable(sentences)))
    data['tokens'] = data['tokens'].map(lambda tokens: [token.lower() for token in tokens if token.isalpha() 
                                                        and token.lower() not in my_stopwords and len(token)>1])
    tokens = data['tokens'].tolist()
    bigram_model = Phrases(tokens)
    trigram_model = Phrases(bigram_model[tokens], min_count=1)
    tokens = list(trigram_model[bigram_model[tokens]])
    dictionary_LDA = corpora.Dictionary(tokens)
    dictionary_LDA.filter_extremes(no_below=3)
    corpus = [dictionary_LDA.doc2bow(tok) for tok in tokens]
    np.random.seed(123456)
    num_topics = 20
    lda_model = models.LdaModel(corpus, num_topics=num_topics, \
                                      id2word=dictionary_LDA, \
                                      passes=4, alpha=[0.01]*num_topics, \
                                      eta=[0.01]*len(dictionary_LDA.keys()))
    topics = [lda_model[corpus[i]] for i in range(len(data))]
    document_topic = \
        pd.concat([topics_document_to_dataframe(topics_document, num_topics=num_topics) for topics_document in topics]) \
          .reset_index(drop=True).fillna(0)
    import seaborn as sns; sns.set(rc={'figure.figsize':(10,20)})
    return sns.heatmap(document_topic.loc[document_topic.idxmax(axis=1).sort_values().index])
    
def visualize_lda_model():
    data = preprocess_to_lemmatization()
    stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'use', 'would', 'can']
    stopwords_other = ['one', 'mr', 'bbc', 'image', 'getty', 'de', 'en', 'caption', 'also', 'copyright', 'something']
    my_stopwords = stopwords.words('english') + stopwords_verbs + stopwords_other
    data['tokens'] = data['tokens_sentences_lemmatized'].map(lambda sentences: list(chain.from_iterable(sentences)))
    data['tokens'] = data['tokens'].map(lambda tokens: [token.lower() for token in tokens if token.isalpha() 
                                                        and token.lower() not in my_stopwords and len(token)>1])
    tokens = data['tokens'].tolist()
    bigram_model = Phrases(tokens)
    trigram_model = Phrases(bigram_model[tokens], min_count=1)
    tokens = list(trigram_model[bigram_model[tokens]])

    dictionary_LDA = corpora.Dictionary(tokens)
    dictionary_LDA.filter_extremes(no_below=3)
    corpus = [dictionary_LDA.doc2bow(tok) for tok in tokens]
    np.random.seed(123456)
    num_topics = 20
    lda_model = models.LdaModel(corpus, num_topics=num_topics, \
                                      id2word=dictionary_LDA, \
                                      passes=4, alpha=[0.01]*num_topics, \
                                      eta=[0.01]*len(dictionary_LDA.keys()))
    lda_viz = gensimvis.prepare(lda_model, corpus, dictionary_LDA)
    pyLDAvis.enable_notebook()
    return pyLDAvis.display(lda_viz)
    
print("Social Media and Digital Methods Lab 4 initialized... OK!")
