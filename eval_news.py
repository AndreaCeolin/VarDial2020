import string
from models import mnb_word, mnb_word_tfidf, svm_word, svm_word_tfidf, mnb_char, mnb_char_tfidf, svm_char, svm_char_tfidf

'''Load Data'''
training = open('data/train.txt').readlines()
dev1 = open('data/dev-source.txt').readlines()

training_sets = [training]

evaluation = open('data/dev-source.txt').readlines()


'''Prepare training set'''
trainDialectLabels = []
trainSamples = []
for training_set in training_sets:
    for line in training_set:
        sample, label = line.split('\t')
        trainDialectLabels.append(label.rstrip().rstrip('\u202c'))
        trainSamples.append(sample.replace('$NE$', '').translate(line.maketrans('', '', string.punctuation+'0123456789')))

'''Prepare evaluation set'''
validationDialectLabels = []
validationSamples = []
for line in evaluation:
    sample, label = line.split('\t')
    validationDialectLabels.append(label.rstrip().rstrip('\u202c'))
    validationSamples.append(sample.replace('$NE$', '').replace('FOTO', '').replace('VIDEO','').replace('LIVE','').translate(line.maketrans('', '', string.punctuation+'|-0123456789”„…')))


'''Data matrices'''
train_list = [(lang, sentence) for lang, sentence in zip(trainDialectLabels, trainSamples)]
eval_list = [(lang, sentence) for lang, sentence in zip(validationDialectLabels, validationSamples)]

print(
'mnb_word:'  + str(mnb_word(train_list,  eval_list, 0.0001)) + '\n',
'mnb_word_tfidf:'  + str(mnb_word_tfidf(train_list, eval_list, 0.0001)) + '\n',
'mnb_char:' + str(mnb_char(train_list, eval_list, 0.0001)) + '\n',
'mnb_char_tfidf :' + str(mnb_char_tfidf(train_list, eval_list, 0.0001)) + '\n',
'svm_word:'  + str(svm_word(train_list, eval_list, 2)) + '\n',
'svm_word_tfidf:'  + str(svm_word_tfidf(train_list, eval_list, 2)) + '\n',
'svm_char:' + str(svm_char(train_list, eval_list, 1)) + '\n',
'svm_char_tfidf:' + str(svm_char_tfidf(train_list, eval_list, 1)))


