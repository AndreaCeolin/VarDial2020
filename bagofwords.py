from os import listdir, makedirs
from os.path import isfile, join, splitext, exists
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import lil_matrix

def mnb_word(train, eval, alpha):
    # Vectorize training
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform([sentence for lang,sentence in train])
    y_train = [lang for lang,sentence in train]

    print('Matrix calculated. Training data: ')
    print('Rows x: ' + str(X_train.shape[0]))
    print('Columns x: ' + str(X_train.shape[1]))
    print('Labels y: ' + str(len(y_train)))

    # Train a Naive Bayes classifier
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)

    print('Classifier trained.')

    # Vectorize Evaluation

    X_eval = vectorizer.transform([sentence for lang, sentence in eval])
    y_eval = [lang for lang,sentence in eval]

    print('Vectorization completed! Now predicting the labels....')

    # Predict
    ypred = model.predict(X_eval)
    accuracyNB = f1_score(y_eval, ypred, average='macro')

    # Calculate F-score globally and print F-score per category
    print('F1_score:')
    print(accuracyNB)

    print('F1_score per category:')
    print(f1_score(y_eval, ypred, average=None))

    return accuracyNB

def mnb_word_tfidf(train, eval, alpha):
    # Vectorize training
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform([sentence for lang,sentence in train])
    y_train = [lang for lang,sentence in train]

    print('Matrix calculated. Training data: ')
    print('Rows x: ' + str(X_train.shape[0]))
    print('Columns x: ' + str(X_train.shape[1]))
    print('Labels y: ' + str(len(y_train)))

    # Train a Naive Bayes classifier
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)

    print('Classifier trained.')

    # Vectorize Evaluation

    X_eval = vectorizer.transform([sentence for lang, sentence in eval])
    y_eval = [lang for lang,sentence in eval]

    print('Vectorization completed! Now predicting the labels....')

    # Predict
    ypred = model.predict(X_eval)
    accuracyNB = f1_score(y_eval, ypred, average='macro')

    # Calculate F-score globally and print F-score per category
    print('F1_score:')
    print(accuracyNB)

    print('F1_score per category:')
    print(f1_score(y_eval, ypred, average=None))

    return accuracyNB

def svm_word(train, eval, alpha):
    # Vectorize training
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform([sentence for lang, sentence in train])
    y_train = [lang for lang, sentence in train]

    print('Matrix calculated. Training data: ')
    print('Rows x: ' + str(X_train.shape[0]))
    print('Columns x: ' + str(X_train.shape[1]))
    print('Labels y: ' + str(len(y_train)))

    # Train a SVM classifier
    model = LinearSVC(C=alpha, class_weight='balanced')
    model.fit(X_train, y_train)

    print('Classifier trained.')

    # Vectorize Evaluation

    X_eval = vectorizer.transform([sentence for lang, sentence in eval])
    y_eval = [lang for lang, sentence in eval]

    print('Vectorization completed! Now predicting the labels....')

    # Predict
    ypred = model.predict(X_eval)
    accuracyNB = f1_score(y_eval, ypred, average='macro')

    # Calculate F-score globally and print F-score per category
    print('F1_score:')
    print(accuracyNB)

    print('F1_score per category:')
    print(f1_score(y_eval, ypred, average=None))

    return accuracyNB

def svm_word_tfidf(train, eval, alpha):
    # Vectorize training
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform([sentence for lang, sentence in train])
    y_train = [lang for lang, sentence in train]

    print('Matrix calculated. Training data: ')
    print('Rows x: ' + str(X_train.shape[0]))
    print('Columns x: ' + str(X_train.shape[1]))
    print('Labels y: ' + str(len(y_train)))

    # Train a SVM classifier
    model = LinearSVC(C=alpha, class_weight='balanced')
    model.fit(X_train, y_train)

    print('Classifier trained.')

    # Vectorize Evaluation

    X_eval = vectorizer.transform([sentence for lang, sentence in eval])
    y_eval = [lang for lang, sentence in eval]

    print('Vectorization completed! Now predicting the labels....')

    # Predict
    ypred = model.predict(X_eval)
    accuracyNB = f1_score(y_eval, ypred, average='macro')

    # Calculate F-score globally and print F-score per category
    print('F1_score:')
    print(accuracyNB)

    print('F1_score per category:')
    print(f1_score(y_eval, ypred, average=None))

    return accuracyNB


