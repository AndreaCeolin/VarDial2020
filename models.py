from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

def mnb_word(train, eval, alpha):
    # Vectorize training set
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
    # Vectorize training set
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
    #We were able to improve the F1 score by merging all the sentences of a language in a single long sentence,
    #using a dictionary

    X_train_dic = defaultdict(list)
    for lang, sentence in train:
        X_train_dic[lang].append(sentence)

    #Vectorize training set
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform([' '.join(sentence) for sentence in X_train_dic.values()])
    y_train = [lang for lang in X_train_dic.keys()]

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
    # Vectorize training set
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

def mnb_char(train, eval, alpha):
    # Vectorize training set
    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(5,8))
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

def mnb_char_tfidf(train, eval, alpha):
    # Vectorize training set
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(5,8))
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

def svm_char(train, eval, alpha):
    #We were able to improve the F1 score by merging all the sentences of a language in a single long sentence,
    #using a dictionary

    X_train_dic = defaultdict(list)
    for lang, sentence in train:
        X_train_dic[lang].append(sentence)

    #Vectorize training set
    vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(6, 8))
    X_train = vectorizer.fit_transform([' '.join(sentence) for sentence in X_train_dic.values()])
    y_train = [lang for lang in X_train_dic.keys()]

    print('Matrix calculated. Training data: ')
    print('Rows x: ' + str(X_train.shape[0]))
    print('Columns x: ' + str(X_train.shape[1]))
    print('Labels y: ' + str(len(y_train)))

    # Train a SVM classifier
    model = LinearSVC(C=alpha)
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

def svm_char_tfidf(train, eval, alpha):
    # Vectorize training set
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(6,8))
    X_train = vectorizer.fit_transform([sentence for lang, sentence in train])
    y_train = [lang for lang, sentence in train]

    print('Matrix calculated. Training data: ')
    print('Rows x: ' + str(X_train.shape[0]))
    print('Columns x: ' + str(X_train.shape[1]))
    print('Labels y: ' + str(len(y_train)))

    # Train a SVM classifier
    model = LinearSVC(C=alpha)
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




