import os
import sys
import pandas as pd
import numpy as np
import numpy.random as npr
import glob
from nltk.tokenize import WordPunctTokenizer
from nltk import FreqDist
import matplotlib.pyplot as plt

np.seterr(invalid='ignore')


def part1_load(folder1, folder2, n = 1):
    '''This function takes two dirs and return a dataframe with each article per row and in columns filename, class(dir)
    and the absolute frequency of words more frequent than n(n=1000) are shown'''
    corpus1 = loader(folder1)
    corpus2 = loader(folder2)

    all_words_pool = []
    # a list of all the words in the corpus put together
    for key, value in corpus1.items():
        all_words_pool += value
    for key, value in corpus2.items():
        all_words_pool += value

    fd = FreqDist(all_words_pool)
    # build a distribution frequency of the words
    frequent_words = []
    for word, count in fd.items():
        if word.isalpha() and count > n:  # n=100
            # if the word(token) consists only letters and occurs more than 1000 times in the corpus
            frequent_words.append(word)

    data = []
    # list of tuple to be written in dataframe
    for key, value in corpus1.items():
        # iterate thourgh all the files in corpus1 and build a list with [filename, dirname and count0,count1,count2...]
        count = []
        filename, dirname = key.split("_from_")

        for word in frequent_words:
            count.append(value.count(word))
        list_to_add = ([filename, dirname] + count)
        data.append(list_to_add)
    for key, value in corpus2.items():
        # iterate thourgh all the files in corpus1 and build a list with [filename, dirname and count0,count1,count2...]
        count = []
        filename, dirname = key.split("_from_")
        for word in frequent_words:
            count.append(value.count(word))
        list_to_add = ([filename, dirname] + count)
        data.append(list_to_add)

    pd.DataFrame(data).fillna(0).to_numpy()

    df = pd.DataFrame(data, columns=(['Filename', 'Dirname'] + frequent_words))

    # create a dataframe with head(title) accordinly
    pd.DataFrame(df).fillna(0).to_numpy()
    return df


def part2_vis(df):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)

    # CHANGE WHAT YOU WANT HERE
    # -------from here----------

    sort_by_class = df.sort_values("Dirname")
    classes = set(sort_by_class["Dirname"])
    sorted_df = {}
    # create a dictionary with {class: dataframe of the class}
    for classname in classes:
        sorted_df[classname] = (df[df["Dirname"] == classname]).to_numpy()

    sum_rows = {}
    # plot the most frequent 5 words (m=5) for readability
    for k, y in sorted_df.items():
        sum_row = y[:, 2:7]
        sum_rows['{}'.format(k)] = (np.sum(sum_row, axis=0))
    index = sort_by_class.columns[2:7]

    df_to_plot = pd.DataFrame(sum_rows, index=index)
    ax = df_to_plot.plot.bar()


def part3_tfidf(df):
    # DO NOT CHANGE
    assert isinstance(df, pd.DataFrame)

    # CHANGE WHAT YOU WANT HERE
    # -------from here----------
    columns = df.columns
    np_raw = df.iloc[:, 2:].to_numpy()
    # a 2-d np maxtrix only contains the count data
    np_extra = df.iloc[:, :2].to_numpy()
    # a 2-d np maxtrix contains the files name and class information

    # calculation of tp
    tf = np.zeros(np_raw.shape)
    # a default matrix filled with 0 to receive the tf values

    for i in range(np_raw.shape[0]):
        # go through each row(file)
        row_sum = np.sum(np_raw[i])
        # number of total words in that file
        for j in range(np_raw.shape[1]):
            # term frequency = word-count/total number of words in that file
            tf[i][j] = (np_raw[i][j]) / row_sum
            # update the tf matrix

    # calculation of idf
    idf = np.zeros(np_raw.shape)
    # a default matrix filled with 0 to receive the tf values

    transposed_np = np_raw.T
    # tranpose the np_raw matrix to iterate over the column(rows/words)

    for i in range(transposed_np.shape[0]):
        # go through all the words(row)
        count = list(transposed_np[i] > 0).count(True)
        # number of files containing the word
        idf_value = np.log((transposed_np.shape[0] * transposed_np.shape[1]) / count)
        # idf = log(N/nt) i.e. log(total number of files/ number of files containing word t)
        for j in range(transposed_np.shape[1]):
            idf[j, i] = idf_value
            # update the idf matraix

    tf_idf = tf * idf
    # tf_idf matrix

    data = np.hstack((np_extra, tf_idf))
    # glue the matrix with extra information together
    data = pd.DataFrame(data).fillna(0)
    # fill the invalid cells with 0 as required in part1
    data.columns = columns

    return data


# ADD WHATEVER YOU NEED HERE, INCLUDING BONUS CODE.

def loader(dir):
    '''load all files in a dir in to a dictionary {filename: [text file with words as tokens]}
    '''
    wpt = WordPunctTokenizer()
    dir_corpus = {}
    for filename in glob.glob('{}/*.txt'.format(dir)):
        with open(filename, "r") as file:
            word_list = []
            for line in file.readlines():
                word_list += [word.lower() for word in wpt.tokenize(line)]
            dir_corpus[os.path.basename(filename) + "_from_" + os.path.basename(dir)] = word_list

    return dir_corpus


# bonus part

def bonus_part():
    from sklearn.naive_bayes import MultinomialNB
    print('load MultinomialNB classifier from sklearn')
    print("(1)Training with fq data")
    data = part1_load('crude', 'grain').to_numpy()
    np.random.shuffle(data)
    # shuffle the data is the output from part 3 if-idf matrix
    X = data[:, 2:]
    # characters
    Y = data[:, 1]
    # class information
    Xtrain = X[:-100, ]
    # slicing
    Ytrain = Y[:-100, ]
    # slicing
    Xtest = X[-100:, ]
    # slicing
    Ytest = Y[-100:, ]
    # slicing

    model = MultinomialNB()
    print('training model...')
    model.fit(Xtrain, Ytrain)
    # training
    print('training done')
    print("----PERFORMANCE REPORT-----")
    print("Classification rate for NB with fq data", model.score(Xtest, Ytest))
    # testing

    data = part3_tfidf(part1_load('crude', 'grain')).to_numpy()
    np.random.shuffle(data)
    # shuffle the data is the output from part 3 if-idf matrix
    X = data[:, 2:]
    # characters
    Y = data[:, 1]
    # class information
    Xtrain = X[:-100, ]
    # slicing
    Ytrain = Y[:-100, ]
    # slicing
    Xtest = X[-100:, ]
    # slicing
    Ytest = Y[-100:, ]
    # slicing

    model = MultinomialNB()
    print('training model...')
    model.fit(Xtrain, Ytrain)
    # training
    print('training done')
    print("----PERFORMANCE REPORT-----")
    print("Classification rate for NB with td-idf data", model.score(Xtest, Ytest))
    # testing

