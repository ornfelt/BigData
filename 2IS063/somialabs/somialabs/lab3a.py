# -*- coding: utf-8 -*-
import pandas as pd
from ipywidgets import (
    interact,
    interactive,
    fixed,
    interact_manual,
    widgets
)
import textmining as tm
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

complaints = pd.read_csv("Consumer_Complaints-sliced.tar.gz")
simple_corpus = pd.Series([
    'She watches bandy and football',
    'Alice likes to play bandy',
    'Karl loves to play football'])

def view_complaints_table():
    global complaints
    return complaints

def view_complaints_table_info():
    return view_complaints_table().info()

def view_sample_narratives():
    complaints = view_complaints_table()
    sample_corpus = complaints["Consumer complaint narrative"].dropna().sample(5)
    for i, narrative in enumerate(sample_corpus):
        print("Complaint {}: {}\n".format(i + 1, narrative))

def filter_stem():
    sampled_corpus = complaints["Consumer complaint narrative"].dropna().sample(100)
    def filter_stem(filter_stem):
        if filter_stem:
            stem_doc = lambda x: ' '.join(tm.stem(x.split()))
            tdm_stemmed = tm.TermDocumentMatrix()
            for complaint in sampled_corpus:
                tdm_stemmed.add_doc(stem_doc(complaint))
            tdm_stemmed_df = tdm_stemmed.to_df(cutoff=1)
            return tdm_stemmed_df.head()
        else:
            tdm = tm.TermDocumentMatrix()
            for complaint in sampled_corpus:
                tdm.add_doc(complaint)
            tdm_df = tdm.to_df(cutoff=1)
            return tdm_df.head()
    interact(filter_stem, filter_stem=False)

def compare_stemmed():
    sampled_corpus = complaints["Consumer complaint narrative"].dropna().sample(100)
    stem_doc = lambda x: ' '.join(tm.stem(x.split()))
    tdm = tm.TermDocumentMatrix()
    for complaint in sampled_corpus:
        tdm.add_doc(complaint)
    tdm_df = tdm.to_df(cutoff=1)
    tdm_stemmed = tm.TermDocumentMatrix()
    for complaint in sampled_corpus:
        tdm_stemmed.add_doc(stem_doc(complaint))
    tdm_stemmed_df = tdm_stemmed.to_df(cutoff=1)
    compare_stemmed = pd.concat([
        pd.Series(sorted(tdm_df.columns)),
        pd.Series(sorted(tdm_stemmed_df.columns))
    ], ignore_index=True, axis=1)
    compare_stemmed.columns = ["Raw terms", "Stemmed terms"]
    return compare_stemmed

def filter_stopwords():
    sampled_corpus = complaints["Consumer complaint narrative"].dropna().sample(100)
    def filter_stopped(filter_stopped):
        if filter_stopped:
            tdm_stopped = tm.TermDocumentMatrix()
            for complaint in sampled_corpus:
                complaint_stopped = ' '.join(tm.simple_tokenize_remove_stopwords(complaint))
                tdm_stopped.add_doc(complaint_stopped)
            tdm_stopped_df = tdm_stopped.to_df(cutoff=1)
            return tdm_stopped_df.head()
        else:
            tdm = tm.TermDocumentMatrix()
            for complaint in sampled_corpus:
                tdm.add_doc(complaint)
            tdm_df = tdm.to_df(cutoff=1)
            return tdm_df.head()
    interact(filter_stopped, filter_stopped=False)

def compare_stopped():
    sampled_corpus = complaints["Consumer complaint narrative"].dropna().sample(100)
    tdm = tm.TermDocumentMatrix()
    for complaint in sampled_corpus:
        tdm.add_doc(complaint)
    tdm_df = tdm.to_df(cutoff=1)
    tdm_stopped = tm.TermDocumentMatrix()
    for complaint in sampled_corpus:
        complaint_stopped = ' '.join(tm.simple_tokenize_remove_stopwords(complaint))
        tdm_stopped.add_doc(complaint_stopped)
    tdm_stopped_df = tdm_stopped.to_df(cutoff=1)
    compare_stopped = pd.concat([
        pd.Series(sorted(tdm_df.columns)),
        pd.Series(sorted(tdm_stopped_df.columns))
    ], ignore_index=True, axis=1)
    compare_stopped.columns = ["Raw terms", "Stopped terms"]
    return compare_stopped

def simple_tdm():
    global simple_corpus
    count_vectorizer = CountVectorizer(min_df=1)
    term_freq_matrix = count_vectorizer.fit_transform(simple_corpus)
    tf_df = pd.DataFrame(data=term_freq_matrix.toarray(), columns=count_vectorizer.get_feature_names())
    tf_df.style.set_caption('Term Document Matrix')
    return tf_df

def simple_tf_idf_matrix():
    global simple_corpus
    count_vectorizer = CountVectorizer(min_df=1)
    term_freq_matrix = count_vectorizer.fit_transform(simple_corpus)
    tfidf = TfidfTransformer()
    tfidf.fit(term_freq_matrix)
    tf_idf_matrix = tfidf.transform(term_freq_matrix)
    tf_idf_df = pd.DataFrame(data=tf_idf_matrix.toarray(), columns=count_vectorizer.get_feature_names())
    tf_idf_df.style.set_caption('Term Frequency-Inverse Document Frequency Matrix')
    return tf_idf_df

def interact_tdm():
    def make_tdm(corpus):
        count_vectorizer = CountVectorizer(min_df=1)
        term_freq_matrix = count_vectorizer.fit_transform(corpus.splitlines())
        tf_df = pd.DataFrame(data=term_freq_matrix.toarray(), columns=count_vectorizer.get_feature_names())
        tf_df.style.set_caption('Term Document Matrix')
        return tf_df
    interact(make_tdm, corpus=widgets.Textarea(
        value='She watches bandy and football\n'
        'Alice likes to play bandy\n'
        'Karl loves to play football',
        description='Corpus:',
        disabled=False
    ))

def interact_tf_idf_matrix():
    def make_tf_idf(corpus):
        count_vectorizer = CountVectorizer(min_df=1)
        term_freq_matrix = count_vectorizer.fit_transform(corpus.splitlines())
        tfidf = TfidfTransformer()
        tfidf.fit(term_freq_matrix)
        tf_idf_matrix = tfidf.transform(term_freq_matrix)
        tf_idf_df = pd.DataFrame(data=tf_idf_matrix.toarray(), columns=count_vectorizer.get_feature_names())
        tf_idf_df.style.set_caption('Term Frequency-Inverse Document Frequency Matrix')
        return tf_idf_df
    interact(make_tf_idf, corpus=widgets.Textarea(
        value='She watches bandy and football\n'
        'Alice likes to play bandy\n'
        'Karl loves to play football',
        description='Corpus:',
        disabled=False
    ))

def interact_complaints_tdm():
    sampled_corpus = complaints["Consumer complaint narrative"].dropna()
    def make_complaints_tdm(corpus_size, stopwords, stemming):
        if stopwords and stemming:
            analyzer = CountVectorizer().build_analyzer()
            def stemmed_words(doc):
                return (tm.stem(w) for w in analyzer(doc))
            count_vectorizer_stemmed = CountVectorizer(min_df=1, analyzer=stemmed_words, stop_words=tm.stopwords)
            term_freq_matrix_stemmed = count_vectorizer_stemmed.fit_transform(sampled_corpus.sample(corpus_size))
            tdm_stopped_stemmed_df = pd.DataFrame(data=term_freq_matrix_stemmed.toarray(), columns=count_vectorizer_stemmed.get_feature_names())
            return tdm_stopped_stemmed_df
        elif stopwords:
            count_vectorizer_stopped = CountVectorizer(min_df=1, stop_words=tm.stopwords)
            term_freq_matrix_stopped = count_vectorizer_stopped.fit_transform(sampled_corpus.sample(corpus_size))
            tdm_stopped_df = pd.DataFrame(data=term_freq_matrix_stopped.toarray(), columns=count_vectorizer_stopped.get_feature_names())
            return tdm_stopped_df
        elif stemming:
            analyzer = CountVectorizer().build_analyzer()
            def stemmed_words(doc):
                return (tm.stem(w) for w in analyzer(doc))
            count_vectorizer_stemmed = CountVectorizer(min_df=1, analyzer=stemmed_words)
            term_freq_matrix_stemmed = count_vectorizer_stemmed.fit_transform(sampled_corpus.sample(corpus_size))
            tdm_stemmed_df = pd.DataFrame(data=term_freq_matrix_stemmed.toarray(), columns=count_vectorizer_stemmed.get_feature_names())
            return tdm_stemmed_df
        else:
            count_vectorizer = CountVectorizer(min_df=1)
            term_freq_matrix = count_vectorizer.fit_transform(sampled_corpus.sample(corpus_size))
            tdm_df = pd.DataFrame(data=term_freq_matrix.toarray(), columns=count_vectorizer.get_feature_names())
            return tdm_df
    interact(make_complaints_tdm,
        corpus_size=widgets.IntSlider(min=1, max=10000, value=1000, continuous_update=False),
        stopwords=widgets.Checkbox(value=False, description='Remove stopwords', continuous_update=False),
        stemming=widgets.Checkbox(value=False, description='Apply stemming', continuous_update=False)

    )

def create_tfidf_matrix(tdm, features):
    transformer = TfidfTransformer()
    tf_idf_matrix = transformer.fit_transform(tdm)
    tfidf_df = pd.DataFrame(data=tf_idf_matrix.toarray(), columns=features)
    return tfidf_df

def interact_complaints_tf_idf_matrix():
    sampled_corpus = complaints["Consumer complaint narrative"].dropna()
    def make_complaints_tdf_idf(corpus_size, stopwords, stemming):
        if stopwords and stemming:
            analyzer = CountVectorizer().build_analyzer()
            def stemmed_words(doc):
                return (tm.stem(w) for w in analyzer(doc))
            count_vectorizer_stemmed = CountVectorizer(min_df=1, analyzer=stemmed_words, stop_words=tm.stopwords)
            term_freq_matrix_stemmed = count_vectorizer_stemmed.fit_transform(sampled_corpus.sample(corpus_size))
            return create_tfidf_matrix(term_freq_matrix_stemmed, count_vectorizer_stemmed.get_feature_names())
        elif stopwords:
            count_vectorizer_stopped = CountVectorizer(min_df=1, stop_words=tm.stopwords)
            term_freq_matrix_stopped = count_vectorizer_stopped.fit_transform(sampled_corpus.sample(corpus_size))
            return create_tfidf_matrix(term_freq_matrix_stopped, count_vectorizer_stopped.get_feature_names())
        elif stemming:
            analyzer = CountVectorizer().build_analyzer()
            def stemmed_words(doc):
                return (tm.stem(w) for w in analyzer(doc))
            count_vectorizer_stemmed = CountVectorizer(min_df=1, analyzer=stemmed_words)
            term_freq_matrix_stemmed = count_vectorizer_stemmed.fit_transform(sampled_corpus.sample(corpus_size))
            return create_tfidf_matrix(term_freq_matrix_stemmed, count_vectorizer_stemmed.get_feature_names())
        else:
            count_vectorizer = CountVectorizer(min_df=1)
            term_freq_matrix = count_vectorizer.fit_transform(sampled_corpus.sample(corpus_size))
            return create_tfidf_matrix(term_freq_matrix, count_vectorizer.get_feature_names())
    interact(make_complaints_tdf_idf,
        corpus_size=widgets.IntSlider(min=1, max=10000, value=1000, continuous_update=False),
        stopwords=widgets.Checkbox(value=False, description='Remove stopwords', continuous_update=False),
        stemming=widgets.Checkbox(value=False, description='Apply stemming', continuous_update=False)
    )

def interact_plot_product_distribution():
    global complaints
    def plot_product_distribution(sample_size):
        sampled_complaints = complaints.sample(sample_size)
        fig = plt.figure(figsize=(8,6))
        sampled_complaints.groupby('Product')["Consumer complaint narrative"].count().plot.bar(ylim=0)
        plt.show()
    interact(plot_product_distribution, sample_size=widgets.IntSlider(min=10, max=len(complaints), continuous_update=False))

def interact_linearSVC():
    global complaints
    def make_classifier(corpus_size, stopwords, stemming):
        sampled_complaints = complaints.dropna(subset=['Consumer complaint narrative']).head(corpus_size)
        X_train, X_test, y_train, y_test = train_test_split(sampled_complaints['Consumer complaint narrative'],
                                                            sampled_complaints.Product, random_state=0)
        print("Our training data has {} rows".format(len(X_train)))
        print("Our test data has {} rows".format(len(X_test)))
        if stopwords and stemming:
            analyzer = CountVectorizer().build_analyzer()
            def stemmed_words(doc):
                return (tm.stem(w) for w in analyzer(doc))
            count_vectorizer_stemmed = CountVectorizer(min_df=1, analyzer=stemmed_words, stop_words=tm.stopwords)
        elif stemming:
            analyzer = CountVectorizer().build_analyzer()
            def stemmed_words(doc):
                return (tm.stem(w) for w in analyzer(doc))
            count_vectorizer_stemmed = CountVectorizer(min_df=1, analyzer=stemmed_words)
        if stopwords:
            count_vectorizer = CountVectorizer(stop_words=tm.stopwords)
        else:
            count_vectorizer = CountVectorizer(stop_words=None)
        X_train_counts = count_vectorizer.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        model = LinearSVC()
        classifier = model.fit(X_train_tfidf, y_train)
        vec = count_vectorizer.transform(X_test)
        predictions = classifier.predict(vec)
        results = pd.DataFrame({
            "Complaint narrative": X_test,
            "Actual category": y_test,
            "Predicted category": predictions
        })
        return results
    interact(make_classifier,
        corpus_size=widgets.IntSlider(min=60, max=len(complaints.dropna(subset=['Consumer complaint narrative'])), value=1000, continuous_update=False),
        stopwords=widgets.Checkbox(value=False, description='Remove stopwords', continuous_update=False),
        stemming=widgets.Checkbox(value=False, description='Apply stemming', continuous_update=False)
    )

def interact_linearSVC_cross_validation():
    global complaints
    def make_classifier(corpus_size, stopwords, stemming):
        sampled_complaints = complaints.dropna(subset=['Consumer complaint narrative']).head(corpus_size)
        X_train, X_test, y_train, y_test = train_test_split(sampled_complaints['Consumer complaint narrative'],
                                                            sampled_complaints.Product, random_state=0)
        print("Our training data has {} rows".format(len(X_train)))
        print("Our test data has {} rows".format(len(X_test)))
        if stopwords and stemming:
            analyzer = CountVectorizer().build_analyzer()
            def stemmed_words(doc):
                return (tm.stem(w) for w in analyzer(doc))
            count_vectorizer_stemmed = CountVectorizer(min_df=1, analyzer=stemmed_words, stop_words=tm.stopwords)
        elif stemming:
            analyzer = CountVectorizer().build_analyzer()
            def stemmed_words(doc):
                return (tm.stem(w) for w in analyzer(doc))
            count_vectorizer_stemmed = CountVectorizer(min_df=1, analyzer=stemmed_words)
        if stopwords:
            count_vectorizer = CountVectorizer(stop_words=tm.stopwords)
        else:
            count_vectorizer = CountVectorizer(stop_words=None)
        X_train_counts = count_vectorizer.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        model = LinearSVC()
        classifier = model.fit(X_train_tfidf, y_train)
        vec = count_vectorizer.transform(X_test)
        predictions = classifier.predict(vec)
        tdf_vectorizer = TfidfVectorizer(min_df=1)
        scores = cross_val_score(classifier, tdf_vectorizer.fit_transform(X_test), predictions, scoring='accuracy', cv=5)
        return scores
    interact(make_classifier,
        corpus_size=widgets.IntSlider(min=60, max=len(complaints.dropna(subset=['Consumer complaint narrative'])), value=1000, continuous_update=False),
        stopwords=widgets.Checkbox(value=False, description='Remove stopwords', continuous_update=False),
        stemming=widgets.Checkbox(value=False, description='Apply stemming', continuous_update=False)
    )


def plot_interact_linearSVC_cross_val_comparison():
    global complaints
    def make_classifier(corpus_size):
        sampled_complaints = complaints.dropna(subset=['Consumer complaint narrative']).head(corpus_size)
        X_train, X_test, y_train, y_test = train_test_split(sampled_complaints['Consumer complaint narrative'],
                                                            sampled_complaints.Product, random_state=0)
        print("Our training data has {} rows".format(len(X_train)))
        print("Our test data has {} rows".format(len(X_test)))

        analyzer = CountVectorizer().build_analyzer()
        def stemmed_words(doc):
            return (tm.stem(w) for w in analyzer(doc))

        count_vectorizer_stemmed = CountVectorizer(min_df=1, analyzer=stemmed_words)
        count_vectorizer_stopped = CountVectorizer(stop_words=tm.stopwords)
        count_vectorizer = CountVectorizer(stop_words=None)

        def get_scores(count_vectorizer):
            X_train_counts = count_vectorizer.fit_transform(X_train)
            tfidf_transformer = TfidfTransformer()
            X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
            model = LinearSVC()
            classifier = model.fit(X_train_tfidf, y_train)
            vec = count_vectorizer.transform(X_test)
            predictions = classifier.predict(vec)
            tdf_vectorizer = TfidfVectorizer(min_df=1)
            scores = cross_val_score(classifier, tdf_vectorizer.fit_transform(X_test), predictions, scoring='accuracy', cv=5)
            return scores

        no_pprocess_scores = get_scores(count_vectorizer)
        stopped_scores = get_scores(count_vectorizer_stopped)
        stemmed_scores = get_scores(count_vectorizer_stemmed)
        d = {}
        labels = ('No processing', 'Stopwords', 'Stemming')
        scores = (no_pprocess_scores, stopped_scores, stemmed_scores)
        for x, y in zip(labels, scores):
            d[x] = y
        cross_val_scores = pd.DataFrame(d)
        _ = cross_val_scores.boxplot().set_title(
            "Cross validation scores trained on {} records".format(corpus_size))
        return cross_val_scores.describe()
    interact(make_classifier,
        corpus_size=widgets.IntSlider(min=60, max=len(complaints.dropna(subset=['Consumer complaint narrative'])), value=1000, continuous_update=False)
    )

print("Social Media and Digital Methods Lab 3a initialized... OK!")
