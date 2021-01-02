##author: Andrea Ceolin
##date: December 2017


import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from scipy.sparse import csr_matrix, lil_matrix
from collections import Counter, defaultdict
import os
import string

#This function extract character ngrams from sentences

def ngrams_extract(sentence, n):
    ngrams = []
    for value in n:
        for word in sentence:
            padded_word = word
            for _ in range(value-1):
                padded_word = '#' + word
                padded_word = padded_word + '#'
            size = len(padded_word)
            for index, letter in enumerate(padded_word):
                if index <= size - value:
                    ngrams.append(padded_word[index:index+value])
    return ngrams

print(ngrams_extract("Hello world!", [4]))
print(ngrams_extract("Hello world!", [2,3,4]))

################################################################

def mnb_char(train, test, n, threshold, alpha):

    #create a dictionary with language as key, and all words used in the training data for that
    #language as a list
    X_train_dic = defaultdict(list)
    for lang,sentence in train:
        X_train_dic[lang].extend(sentence.split())

    #Feature matrix. Count # of occurrences of words. Notice that this matrix creates a big vector
    #for each language, putting together the content of all sentences, and treating it as a long single sentence.
    #This is good to speed up training

    ngrams_all = Counter()
    for key, long_sentence in X_train_dic.items():
        for ngram in ngrams_extract(long_sentence, n):
            ngrams_all[ngram] +=1

    ngrams_size = len(ngrams_all)
    print('Total # of ngrams: ' + str(ngrams_size))

    #This part allows us to filter out high frequency ngrams

    ngrams = dict()
    for ngram, count in ngrams_all.items():
        if count<threshold:
            ngrams[ngram] = count

    ngrams_size = len(ngrams)
    print('Total # of ngrams (filtered): ' + str(ngrams_size))

    #Keep an index for each ngram, so you can fill the vector later
    ngram_index = {}
    i=0
    for ngram in ngrams:
        ngram_index[ngram] = int(i)
        i+=1


    #Build the training vectors

    X_train = lil_matrix((len(X_train_dic), ngrams_size))
    y_train = []


    i=0
    for lang, long_sentence in X_train_dic.items():
        #this is the word vector
        #this dictionary is used to count the words
        ngrams_counts = Counter(ngrams_extract(long_sentence,n))
        #fill the word vector with the counts for each word
        for ngram, count in ngrams_counts.items():
            if ngram in ngrams:
                X_train[i, ngram_index[ngram]] = count
        #add the word vector to the matrix for the language, keep track of the language name
        i+=1
        y_train.append(lang)


    print('Matrix calculated. Training data: ')
    print('Rows x: ' + str(X_train.shape[0]))
    print('Columns x: ' + str(X_train.shape[1]))
    print('Labels y: ' + str(len(y_train)))

    #Train a Naive Bayes classifier

    model = MultinomialNB(alpha=alpha)
    model.fit(X_train,y_train)

    print('Classifier trained. Now I am working on the test dataset..')

    #Now let's work the test sentences. You need to create a wordvector for each sentence to train the classifier

    #Initiate sparse matrix
    y_test = [lang for lang, sentence in test]
    test_len = len(y_test)

    X_test = lil_matrix((test_len, ngrams_size))

    i=0

    for lang,sentence in test:
        #Now we create the wordvector using the words in the sentence. We start by counting.
        ngrams_counts = Counter(ngrams_extract(sentence.split(),n))
        #fill the wordvector with counts
        for ngram, count in ngrams_counts.items():
            if ngram in ngrams:
                X_test[i, ngram_index[ngram]] += count
        i+=1
        if i % 500 == 0:
            print('Vectorization. Sentence ' + str(i) + '/' + str(test_len) + ' (' + str(int(i/test_len*100)) + '%)')
    print('Vectorization completed! Now predicting the labels....')


    ypred = model.predict(X_test)

    # Calculate F-score globally and print F-score per category
    accuracyNB = f1_score(y_test, ypred, average='macro')
    print('Done! F1_score global: ' + str(accuracyNB))

    print('F1_score per category:')
    print(f1_score(y_test, ypred, average=None))

    return accuracyNB

def mnb_char_tfidf(train, test, n, threshold, alpha):

    #create a dictionary with language as key, and all words used in the training data for that
    #language as a list
    X_train_dic = defaultdict(list)
    for lang,sentence in train:
        X_train_dic[lang].extend(sentence.split())

    #Feature matrix. Count # of occurrences of words. Notice that this matrix creates a big vector
    #for each language, putting together the content of all sentences, and treating it as a long single sentence.
    #This is good to speed up training

    ngrams_all = Counter()
    for key, long_sentence in X_train_dic.items():
        for ngram in ngrams_extract(long_sentence, n):
            ngrams_all[ngram] +=1

    ngrams_size = len(ngrams_all)
    print('Total # of ngrams: ' + str(ngrams_size))


    #This part allows us to filter out high frequency ngrams

    ngrams = dict()
    for ngram, count in ngrams_all.items():
        if count<threshold:
            ngrams[ngram] = count

    ngrams_size = len(ngrams)
    print('Total # of ngrams (filtered): ' + str(ngrams_size))

    #Keep an index for each ngram, so you can fill the vector later
    ngram_index = {}
    i=0
    for ngram in ngrams:
        ngram_index[ngram] = int(i)
        i+=1



    #Build the training vectors

    X_train = lil_matrix((len(train), ngrams_size))
    y_train = []


    i=0
    for lang, long_sentence in train:
        #this is the word vector
        #this dictionary is used to count the words
        ngrams_counts = Counter(ngrams_extract(long_sentence,n))
        #fill the word vector with the counts for each word
        for ngram, count in ngrams_counts.items():
            if ngram in ngrams:
                X_train[i, ngram_index[ngram]] = count
        #add the word vector to the matrix for the language, keep track of the language name
        i+=1
        y_train.append(lang)



    print('Matrix calculated. Training data: ')
    print('Rows x: ' + str(X_train.shape[0]))
    print('Columns x: ' + str(X_train.shape[1]))
    print('Labels y: ' + str(len(y_train)))


    #Transform the training matrix using TF-IDF

    tfidf = TfidfTransformer(sublinear_tf=True)
    X_train = tfidf.fit_transform(X_train)

    #Train a Naive Bayes classifier

    model = MultinomialNB(alpha=alpha)
    model.fit(X_train,y_train)

    print('Classifier trained. Now I am working on the test dataset..')

    #Now let's work the test sentences. You need to create a wordvector for each sentence to train the classifier

    #Initiate sparse matrix
    y_test = [lang for lang, sentence in test]
    test_len = len(y_test)

    X_test = lil_matrix((test_len, ngrams_size))

    i=0

    for lang,sentence in test:
        #Now we create the wordvector using the words in the sentence. We start by counting.
        ngrams_counts = Counter(ngrams_extract(sentence.split(),n))
        #fill the wordvector with counts
        for ngram, count in ngrams_counts.items():
            if ngram in ngrams:
                X_test[i, ngram_index[ngram]] += count
        i+=1
        if i % 500 == 0:
            print('Vectorization. Sentence ' + str(i) + '/' + str(test_len) + ' (' + str(int(i/test_len*100)) + '%)')
    print('Vectorization completed! Now predicting the labels....')

    X_test = tfidf.transform(X_test)
    ypred = model.predict(X_test)

    # Calculate F-score globally and print F-score per category
    accuracyNB = f1_score(y_test, ypred, average='macro')
    print('Done! F1_score global: ' + str(accuracyNB))

    print('F1_score per category:')
    print(f1_score(y_test, ypred, average=None))

    return accuracyNB

#################################################################################################

def svm_char(train, test, n, threshold, alpha):
    # create a dictionary with language as key, and all words used in the training data for that
    # language as a list
    X_train_dic = defaultdict(list)
    for lang, sentence in train:
        X_train_dic[lang].extend(sentence.split())

    # Feature matrix. Count # of occurrences of words. Notice that this matrix creates a big vector
    # for each language, putting together the content of all sentences, and treating it as a long single sentence.
    # This is good to speed up training

    ngrams_all = Counter()
    for key, long_sentence in X_train_dic.items():
        for ngram in ngrams_extract(long_sentence, n):
            ngrams_all[ngram] += 1

    ngrams_size = len(ngrams_all)
    print('Total # of ngrams: ' + str(ngrams_size))

    # This part allows us to filter out high frequency ngrams

    ngrams = dict()
    for ngram, count in ngrams_all.items():
        if count < threshold:
            ngrams[ngram] = count

    ngrams_size = len(ngrams)
    print('Total # of ngrams (filtered): ' + str(ngrams_size))

    # Keep an index for each ngram, so you can fill the vector later
    ngram_index = {}
    i = 0
    for ngram in ngrams:
        ngram_index[ngram] = int(i)
        i += 1

    # Build the training vectors

    X_train = lil_matrix((len(X_train_dic), ngrams_size))
    y_train = []

    i = 0
    for lang, long_sentence in X_train_dic.items():
        # this is the word vector
        # this dictionary is used to count the words
        ngrams_counts = Counter(ngrams_extract(long_sentence, n))
        # fill the word vector with the counts for each word
        for ngram, count in ngrams_counts.items():
            if ngram in ngrams:
                X_train[i, ngram_index[ngram]] = count
        # add the word vector to the matrix for the language, keep track of the language name
        i += 1
        y_train.append(lang)

    print('Matrix calculated. Training data: ')
    print('Rows x: ' + str(X_train.shape[0]))
    print('Columns x: ' + str(X_train.shape[1]))
    print('Labels y: ' + str(len(y_train)))

    # Train a Naive Bayes classifier

    model = LinearSVC(C=alpha, class_weight='balanced')
    model.fit(X_train, y_train)

    print('Classifier trained. Now I am working on the test dataset..')

    # Now let's work the test sentences. You need to create a wordvector for each sentence to train the classifier

    # Initiate sparse matrix
    y_test = [lang for lang, sentence in test]
    test_len = len(y_test)

    X_test = lil_matrix((test_len, ngrams_size))

    i = 0

    for lang, sentence in test:
        # Now we create the wordvector using the words in the sentence. We start by counting.
        ngrams_counts = Counter(ngrams_extract(sentence.split(), n))
        # fill the wordvector with counts
        for ngram, count in ngrams_counts.items():
            if ngram in ngrams:
                X_test[i, ngram_index[ngram]] += count
        i += 1
        if i % 500 == 0:
            print(
                'Vectorization. Sentence ' + str(i) + '/' + str(test_len) + ' (' + str(int(i / test_len * 100)) + '%)')
    print('Vectorization completed! Now predicting the labels....')

    ypred = model.predict(X_test)

    # Calculate F-score globally and print F-score per category
    accuracyNB = f1_score(y_test, ypred, average='macro')
    print('Done! F1_score global: ' + str(accuracyNB))

    print('F1_score per category:')
    print(f1_score(y_test, ypred, average=None))

    return accuracyNB

def svm_char_tfidf(train, test, n, threshold, alpha):
    # create a dictionary with language as key, and all words used in the training data for that
    # language as a list
    X_train_dic = defaultdict(list)
    for lang, sentence in train:
        X_train_dic[lang].extend(sentence.split())

    # Feature matrix. Count # of occurrences of words. Notice that this matrix creates a big vector
    # for each language, putting together the content of all sentences, and treating it as a long single sentence.
    # This is good to speed up training

    ngrams_all = Counter()
    for key, long_sentence in X_train_dic.items():
        for ngram in ngrams_extract(long_sentence, n):
            ngrams_all[ngram] += 1

    ngrams_size = len(ngrams_all)
    print('Total # of ngrams: ' + str(ngrams_size))

    # This part allows us to filter out high frequency ngrams

    ngrams = dict()
    for ngram, count in ngrams_all.items():
        if count < threshold:
            ngrams[ngram] = count

    ngrams_size = len(ngrams)
    print('Total # of ngrams (filtered): ' + str(ngrams_size))

    # Keep an index for each ngram, so you can fill the vector later
    ngram_index = {}
    i = 0
    for ngram in ngrams:
        ngram_index[ngram] = int(i)
        i += 1

    # Build the training vectors

    X_train = lil_matrix((len(train), ngrams_size))
    y_train = []

    i = 0
    for lang, long_sentence in train:
        # this is the word vector
        # this dictionary is used to count the words
        ngrams_counts = Counter(ngrams_extract(long_sentence, n))
        # fill the word vector with the counts for each word
        for ngram, count in ngrams_counts.items():
            if ngram in ngrams:
                X_train[i, ngram_index[ngram]] = count
        # add the word vector to the matrix for the language, keep track of the language name
        i += 1
        y_train.append(lang)

    print('Matrix calculated. Training data: ')
    print('Rows x: ' + str(X_train.shape[0]))
    print('Columns x: ' + str(X_train.shape[1]))
    print('Labels y: ' + str(len(y_train)))

    # Transform the training matrix using TF-IDF

    tfidf = TfidfTransformer(sublinear_tf=True)
    X_train = tfidf.fit_transform(X_train)

    # Train a Naive Bayes classifier

    model = LinearSVC(C=alpha, class_weight='balanced')
    model.fit(X_train, y_train)

    print('Classifier trained. Now I am working on the test dataset..')

    # Now let's work the test sentences. You need to create a wordvector for each sentence to train the classifier

    # Initiate sparse matrix
    y_test = [lang for lang, sentence in test]
    test_len = len(y_test)

    X_test = lil_matrix((test_len, ngrams_size))

    i = 0

    for lang, sentence in test:
        # Now we create the wordvector using the words in the sentence. We start by counting.
        ngrams_counts = Counter(ngrams_extract(sentence.split(), n))
        # fill the wordvector with counts
        for ngram, count in ngrams_counts.items():
            if ngram in ngrams:
                X_test[i, ngram_index[ngram]] += count
        i += 1
        if i % 500 == 0:
            print(
                'Vectorization. Sentence ' + str(i) + '/' + str(test_len) + ' (' + str(int(i / test_len * 100)) + '%)')
    print('Vectorization completed! Now predicting the labels....')

    X_test = tfidf.transform(X_test)
    ypred = model.predict(X_test)

    # Calculate F-score globally and print F-score per category
    accuracyNB = f1_score(y_test, ypred, average='macro')
    print('Done! F1_score global: ' + str(accuracyNB))

    print('F1_score per category:')
    print(f1_score(y_test, ypred, average=None))

    return accuracyNB


