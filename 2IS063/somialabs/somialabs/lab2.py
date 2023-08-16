# -*- coding: utf-8 -*-
import pandas as pd
from ipywidgets import (
    interact,
    interactive,
    fixed,
    interact_manual,
    widgets
)
from sklearn.feature_extraction.text import CountVectorizer
from textmining import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

bok_tweets = pd.read_csv(
    "https://s3.eu-west-2.amazonaws.com/uu-datamining-assets/lab1-data.tsv",
    encoding="utf-8", sep="\t", index_col="id"
)
sample_corpus = pd.Series([
    "The loveliest of lovely meetings at Gothenburg Book Fair w @KinsellaSophie Thank you 💗 #bokmässan #bookblogger @Marimekkoglobal @wsoykirjat",
    "The 1st day of 3 at #Bokmässan today 📖",
    "This storybook app is a great conversation starter for #parents & #teachers 👨‍👩‍👧‍👦https://youtu.be/LUSz-7dmyRs  FREE DOWNLOAD 🙌🏻 #bokmässan #SEL",
    "1) this is my dream workspace 2) this #bokmässan session on picture books reminds me of my (currently dormant 😭) dream of being a librarian.",
    "The unboxing moment 😇😀😊☺️📚📚📚📙📖📑📓📔📕📖🎆🎇🎈🎂🎊🎉🎁 #mittulricehamn #författarlivet #bokmässan… https://www.instagram.com/p/BZeIyhvDoGP/"
])

my_stop_words = ["och", "det", "att", "i", "en", "jag", "hon",
                "som", "han", "paa", "den", "med", "var", "sig",
                "foer", "saa", "till", "aer", "men", "ett",
                "om", "hade", "de", "av", "icke", "mig", "du",
                "henne", "daa", "sin", "nu", "har", "inte",
                "hans", "honom", "skulle", "hennes", "daer",
                "min", "man", "ej", "vid", "kunde", "naagot",
                "fraan", "ut", "naer", "efter", "upp", "vi",
                "dem", "vara", "vad", "oever", "aen", "dig",
                "kan", "sina", "haer", "ha", "mot", "alla",
                "under", "naagon", "eller", "allt", "mycket",
                "sedan", "ju", "denna", "sjaelv", "detta",
                "aat", "utan", "varit", "hur", "ingen", "mitt",
                "ni", "bli", "blev", "oss", "din", "dessa",
                "naagra", "deras", "blir", "mina", "samma",
                "vilken", "er", "saadan", "vaar", "blivit",
                "dess", "inom", "mellan", "saadant", "varfoer",
                "varje", "vilka", "ditt", "vem", "vilket",
                "sitta", "saadana", "vart", "dina", "vars",
                "vaart", "vaara", "ert", "era", "vilka"]

def first_5_tweets():
    """
    for i, item in enumerate(bok_tweets.text.head()):
        print("Tweet {}: {}".format(i, item))
    """
    global bok_tweets
    for i, item in enumerate(bok_tweets.text.head()):
        print("Tweet {}: {}".format(i, item))

def view_table():
    global bok_tweets
    return bok_tweets

def view_table_info():
    global bok_tweets
    return bok_tweets.info()

def filter_emojis():
    global sample_corpus
    def filter_emojis(filter_emojis):
        if filter_emojis:
            encode2ascii = lambda x: x.encode('ascii', errors='ignore').decode('utf-8')
            clean_tweets = sample_corpus.apply(encode2ascii)
            for tweet in clean_tweets:
                print(tweet)
        else:
            for tweet in sample_corpus:
                print(tweet)
    interact(filter_emojis, filter_emojis=False)

def filter_urls():
    global sample_corpus
    def filter_urls(filter_urls):
        if filter_urls:
            clean_tweets = sample_corpus.str.replace(r'http\S+', '')
            for tweet in clean_tweets:
                print(tweet)
        else:
            for tweet in sample_corpus:
                print(tweet)
    interact(filter_urls, filter_urls=False)

def create_term_document_matrix(sample_size, min_df=1):
    corpus = bok_tweets.head(sample_size).text
    cvec = CountVectorizer(min_df=min_df, stop_words=stopwords)
    tfmatrix = cvec.fit_transform(corpus)
    return pd.DataFrame(data=tfmatrix.toarray(), columns=cvec.get_feature_names())

def make_tdm():
    global bok_tweets
    interact(create_term_document_matrix, sample_size=widgets.IntSlider(min=1,max=15,step=1,value=3), min_df=fixed(1))

def top_words(num_word_instances, top_words):
    tweets = bok_tweets.text
    tdm_df = create_term_document_matrix(len(tweets), min_df=2)
    word_frequencies = tdm_df[[x for x in tdm_df.columns if len(x) > 1]].sum()
    sorted_words = word_frequencies.sort_values(ascending=False)
    top_sorted_words = sorted_words[:num_word_instances]
    top_sorted_words[:top_words].plot.bar()
    return top_sorted_words

def plot_top_words():
    global bok_tweets
    interact(top_words, num_word_instances=widgets.IntSlider(min=1, value=50, continuous_update=False), top_words=widgets.IntSlider(min=1, value=30, continuous_update=False))

def make_lowercase():
    global sample_corpus
    def lowercase(lowercase):
        if lowercase:
            clean_tweets = sample_corpus.str.lower()
            for tweet in clean_tweets:
                print(tweet)
        else:
            for tweet in sample_corpus:
                print(tweet)
    interact(lowercase, lowercase=False)

def remove_small_words():
    global sample_corpus
    def small_words(small_words):
        if small_words:
            clean_tweets = sample_corpus.str.findall('\w{3,}').str.join(' ')
            for tweet in clean_tweets:
                print(tweet)
        else:
            for tweet in sample_corpus:
                print(tweet)
    interact(small_words, small_words=False)

remove_stopwords = lambda x: ' '.join(y for y in x.split() if y not in my_stop_words)

def remove_stop_words():
    global bok_tweets
    sample_corpus = bok_tweets.sample(5).text
    def stop_words(stop_words):
        if stop_words:
            clean_tweets = sample_corpus.apply(remove_stopwords)
            for tweet in clean_tweets:
                print(tweet)
        else:
            for tweet in sample_corpus:
                print(tweet)
    interact(stop_words, stop_words=False)

def plot_top_words_with_filters():
    global bok_tweets
    def create_term_document_matrix2(corpus, min_df=1):
        cvec = CountVectorizer(min_df=min_df, stop_words=stopwords)
        tfmatrix = cvec.fit_transform(corpus)
        return pd.DataFrame(data=tfmatrix.toarray(), columns=cvec.get_feature_names())
    def plot_top_words_with_filters(num_word_instances, top_words, stop_words, small_words, lower):
        tweets = bok_tweets.text
        if lower:
            tweets = tweets.str.lower()
        if stop_words:
            tweets = tweets.apply(remove_stopwords)
        if small_words:
            tweets = tweets.str.findall('\w{3,}').str.join(' ')
        tdm_df = create_term_document_matrix2(tweets, min_df=2)
        word_frequencies = tdm_df[[x for x in tdm_df.columns if len(x) > 1]].sum()
        sorted_words = word_frequencies.sort_values(ascending=False)
        top_sorted_words = sorted_words[:num_word_instances]
        top_sorted_words[:top_words].plot.bar()
        return top_sorted_words
    interact(plot_top_words_with_filters,
        num_word_instances=widgets.IntSlider(min=1, value=50, continuous_update=False),
        top_words=widgets.IntSlider(min=1, value=30, continuous_update=False),
        stop_words=widgets.Checkbox(value=False, description='Filter stop words', continuous_update=False),
        small_words=widgets.Checkbox(value=False, description='Filter small words', continuous_update=False),
        lower=widgets.Checkbox(value=False, description='Apply lowercase', continuous_update=False)
        )

def plot_top_words_with_custom_stopwords():
    global bok_tweets
    def create_term_document_matrix2(corpus, min_df=1):
        cvec = CountVectorizer(min_df=min_df, stop_words=stopwords)
        tfmatrix = cvec.fit_transform(corpus)
        return pd.DataFrame(data=tfmatrix.toarray(), columns=cvec.get_feature_names())
    def plot_top_words_with_filters(num_word_instances, top_words, stop_words, small_words, lower, more_stop_words):
        tweets = bok_tweets.text
        if lower:
            tweets = tweets.str.lower()
        if stop_words:
            tweets = tweets.apply(remove_stopwords)
        if small_words:
            tweets = tweets.str.findall('\w{3,}').str.join(' ')
        if len(more_stop_words) > 0:
            remove_more_stopwords = lambda x: ' '.join(y for y in x.split() if y not in (x.strip() for x in more_stop_words.split(',')))
            tweets = tweets.apply(remove_more_stopwords)
        tdm_df = create_term_document_matrix2(tweets, min_df=2)
        word_frequencies = tdm_df[[x for x in tdm_df.columns if len(x) > 1]].sum()
        sorted_words = word_frequencies.sort_values(ascending=False)
        top_sorted_words = sorted_words[:num_word_instances]
        top_sorted_words[:top_words].plot.bar()
        return top_sorted_words
    interact(plot_top_words_with_filters,
        num_word_instances=widgets.IntSlider(min=1, value=50, continuous_update=False),
        top_words=widgets.IntSlider(min=1, value=30, continuous_update=False),
        stop_words=widgets.Checkbox(value=False, description='Filter stop words', continuous_update=False),
        small_words=widgets.Checkbox(value=False, description='Filter small words', continuous_update=False),
        lower=widgets.Checkbox(value=False, description='Apply lowercase', continuous_update=False),
        more_stop_words=widgets.Text(value='aar,år', description='Additional stop words:', continuous_update=False))

def plot_wordcloud():
    global bok_tweets
    def create_term_document_matrix2(corpus, min_df=1):
        cvec = CountVectorizer(min_df=min_df, stop_words=stopwords)
        tfmatrix = cvec.fit_transform(corpus)
        return pd.DataFrame(data=tfmatrix.toarray(), columns=cvec.get_feature_names())
    def plot_top_words_with_filters(num_word_instances, stop_words, small_words, lower, more_stop_words):
        tweets = bok_tweets.text
        if lower:
            tweets = tweets.str.lower()
        if stop_words:
            tweets = tweets.apply(remove_stopwords)
        if small_words:
            tweets = tweets.str.findall('\w{3,}').str.join(' ')
        if len(more_stop_words) > 0:
            remove_more_stopwords = lambda x: ' '.join(y for y in x.split() if y not in (x.strip() for x in more_stop_words.split(',')))
            tweets = tweets.apply(remove_more_stopwords)
        tdm_df = create_term_document_matrix2(tweets, min_df=2)
        word_frequencies = tdm_df[[x for x in tdm_df.columns if len(x) > 1]].sum()
        sorted_words = word_frequencies.sort_values(ascending=False)
        top_sorted_words = sorted_words[:num_word_instances]
        wordcloud = WordCloud(max_font_size=40)
        wordcloud.fit_words(top_sorted_words.to_dict())
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
    interact(plot_top_words_with_filters,
        num_word_instances=widgets.IntSlider(min=1, value=50, continuous_update=False),
        stop_words=widgets.Checkbox(value=False, description='Filter stop words', continuous_update=False),
        small_words=widgets.Checkbox(value=False, description='Filter small words', continuous_update=False),
        lower=widgets.Checkbox(value=False, description='Apply lowercase', continuous_update=False),
        more_stop_words=widgets.Text(value='aar,år', description='Additional stop words:', continuous_update=False))

print("Social Media and Digital Methods Lab 2 initialized... OK!")
