#Manage Imports
import re
import csv
from nltk.corpus import stopwords  
from nltk import stem
from nltk import tokenize
import nltk
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import json
import time
nltk.download('stopwords')

# Load data from file
def loadData(file, rowToFilter1, rowToFilter2, delimiterToFilter):
    reviews = []
    reviews_ids = []
    with open(file) as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiterToFilter)
        for row in reader:
            reviews.append(row[rowToFilter1])
            reviews_ids.append(row[rowToFilter2])
    print("Data were load from CSV File")
    return reviews, reviews_ids

# Remove stop word tokens by using NLTKs English stop word library
def removeStopwords(stringset):
    stop_words = set(stopwords.words("english"))
    filtered_set = []
    for word in stringset:
        if word.lower() not in stop_words:
            filtered_set.append(word)
    return filtered_set

# Split up reviews into tokens by using NLTKs regular expression tokenizer with a custom regex statement
def splitByRegEx(reviews):
    unfiltered_tokens = []
    regex_statement = '[a-zA-Záéíóú]+'
    tokenizer = tokenize.RegexpTokenizer(regex_statement)

    for review in reviews:
        unfiltered_tokens.append(tokenizer.tokenize(review))
    print("Split by RegEx")
    return unfiltered_tokens

# Remove stop word tokens
def removeStopWordsToken(unfilteredTokens):
    filtered_tokens = []
    for review in unfilteredTokens:
        filtered_tokens.append(removeStopwords(review))
    print("Stopwords removed")
    return filtered_tokens

# Stem the remaining words
def stemming(filteredTokens):
    stemmed_tokens = []
    stemmer = stem.snowball.EnglishStemmer()
    for review in filteredTokens:
        stemmed_tokens_temp = []
        for token in review:
            stemmed_tokens_temp.append(stemmer.stem(token))
        stemmed_tokens.append(stemmed_tokens_temp)
    print("Stemming was successful")
    return stemmed_tokens

# Merge common multi word expressions into one token
def mergeCommonAbbr(stemmedTokens):
    merged_tokens = []
    dictionary = [
        ('e', 'g')
    ]
    tokenizer = tokenize.MWETokenizer(dictionary, separator='_')
    for review in stemmedTokens:
        merged_tokens.append(tokenizer.tokenize(review))
    print("Merged the common Word Expressions")
    return merged_tokens

#Count the total Number of Tokens
def countTokens(allTokens):
    token_count = 0
    for review in allTokens:
        if len(review)>0:
            for token in review:
                token_count+=1
    print("Number of all tokens: ", token_count)

#Create List of all tokens and IDs
def createTokenList(merged_tokens, reviews_ids):
    for i in range(0, len(merged_tokens)):
        merged_tokens[i].insert(0, reviews_ids[i])
    print("Token List is created")
    return merged_tokens

#recreate the sentences with the preprocessed words
def recreateSentence(merged_tokens):
    tokens = []
    id = []
    stemmed_sentences = []
    i = 1
    for review in merged_tokens:
        tmp = ""
        review.pop(0)

        for word in review:
            tmp += str(word)+" "
            tokens.append(word)
            id.append(i)
            i += 1
        stemmed_sentences.append(tmp)

    tokens = set(tokens)
    id = set(id)
    print("Sentences were recreated succesful")
    return tokens, id, stemmed_sentences

#Save the created Vectors
def saveData(dimensions, vectors, countVectors, connections, countConnections, word2int):
    #save all data
    path = "vectors_"+str(dimensions)+"d.csv"
    with open(path, mode='w', newline='') as csv_file:
        for x in range(countVectors):
            tmp_str = ""
            if(dimensions == 2):
                tmp_str += vectors.iat[x,0] + "," + str(vectors.iat[x,1]) + "," + str(vectors.iat[x,2])
            if(dimensions == 3):
                tmp_str += vectors.iat[x,0] + "," + str(vectors.iat[x,1]) + "," + str(vectors.iat[x,2]) + "," + str(vectors.iat[x,3])
            csv_file.write(tmp_str)
            csv_file.write('\n')
            
    print("Saved the Vectors")
    
    path = "neighbours_"+str(dimensions)+"d.txt"
    with open(path,'w') as file:
        for x in range(countConnections):
            tmp_str = ""
            tmp_str += connections.iat[x,0] + "," + connections.iat[x,1]
            file.write(tmp_str)
            file.write('\n')
            
    print("Saved the word neighbours")

    path = "word2int_"+str(dimensions)+"d.txt"
    with open(path,'w') as file:
        file.write(json.dumps(word2int))
                
    print("Saved the one hot encoding")

# function to convert numbers to one hot vectors
def to_one_hot_encoding(data_point_index, dimension):
    one_hot_encoding = np.zeros(dimension)
    one_hot_encoding[data_point_index] = 1
    return one_hot_encoding

#create Vectors by using a NN
def createVectors(review_text, dimensions, WINDOW_SIZE, words):
    print()
    print("-----------------")
    print("Let's create some Word Vectors by using a NN!")
    word2int = {}

    for i,word in enumerate(words):
        word2int[word] = i
    
    sentences = []
    for sentence in review_text:
        sentences.append(sentence.split())
        
    data = []
    for sentence in sentences:
        for idx, word in enumerate(sentence):
            for neighbor in sentence[max(idx - WINDOW_SIZE, 0) : min(idx + WINDOW_SIZE, len(sentence)) + 1] : 
                if neighbor != word:
                    data.append([word, neighbor])
    
    df = pd.DataFrame(data, columns = ['input', 'label'])
    
    #start the training for the neuronal network
    ONE_HOT_DIM = len(words)
    
    X = [] # input word
    Y = [] # target word
    
    for x, y in zip(df['input'], df['label']):
        X.append(to_one_hot_encoding(word2int[ x ], ONE_HOT_DIM))
        Y.append(to_one_hot_encoding(word2int[ y ], ONE_HOT_DIM))
    
    # convert them to numpy arrays
    X_train = np.asarray(X)
    Y_train = np.asarray(Y)
    
    # making placeholders for X_train and Y_train
    x = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))
    y_label = tf.placeholder(tf.float32, shape=(None, ONE_HOT_DIM))
    
    # word embedding will be 2 dimension for 2d visualization
    EMBEDDING_DIM =  dimensions
    
    # hidden layer: which represents word vector eventually
    W1 = tf.Variable(tf.random_normal([ONE_HOT_DIM, EMBEDDING_DIM]))
    b1 = tf.Variable(tf.random_normal([1])) #bias
    hidden_layer = tf.add(tf.matmul(x,W1), b1)
    
    # output layer
    W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, ONE_HOT_DIM]))
    b2 = tf.Variable(tf.random_normal([1]))
    prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_layer, W2), b2))
    
    # loss function: cross entropy
    loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), axis=[1]))
    
    # training operation
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    
    import time
    start = time.time()
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init) 
    
    iteration = 25000
    for i in range(iteration):
        # input is X_train which is one hot encoded word
        # label is Y_train which is one hot encoded neighbor word
        sess.run(train_op, feed_dict={x: X_train, y_label: Y_train})
        if i % 100 == 0:
            percent = (i/iteration)*100
            print(str (dimensions) + "D: " + str(round(percent,1)) + "%")
            
    end = time.time()
    time = (end - start)/60
    print("Vectors are created. It took " + str(time) + " minutes.")
    print()
    
    # Now the hidden layer (W1 + b1) is actually the word look up table
    vectors = sess.run(W1 + b1)
    
    #save the vectors in a data frame
    if(dimensions == 2):
        w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2'])
        w2v_df['word'] = words
        w2v_df = w2v_df[[ 'word', 'x1', 'x2']]
        w2v_df
    if(dimensions == 3):
        w2v_df = pd.DataFrame(vectors, columns = ['x1', 'x2', 'x3'])
        w2v_df['word'] = words
        w2v_df = w2v_df[[ 'word', 'x1', 'x2', 'x3']]
        w2v_df
    
    fig, ax = plt.subplots()

    #Plot a 2D-Model
    if(dimensions == 2):
        for word, x1, x2 in zip(w2v_df['word'], w2v_df['x1'], w2v_df['x2']):
            ax.annotate(word, (x1,x2 ))
        
        PADDING = 1.0
        x_axis_min = np.amin(vectors, axis=0)[0] - PADDING
        y_axis_min = np.amin(vectors, axis=0)[1] - PADDING
        x_axis_max = np.amax(vectors, axis=0)[0] + PADDING
        y_axis_max = np.amax(vectors, axis=0)[1] + PADDING
         
        plt.xlim(x_axis_min,x_axis_max)
        plt.ylim(y_axis_min,y_axis_max)
        plt.rcParams["figure.figsize"] = (20,20)
        
        plt.show()

    #save the Data into files
    saveData(dimensions, w2v_df, len(w2v_df.index), df, len(df.index), word2int)

#run the process for Word2Vec
def main():
    print("")
    print("")
    print("--------------------------------------")
    print("Let us try out Word2Vec!")
    reviews, reviews_ids = loadData('yelp_short.csv', 'text', 'review_id', ',')
    unfiltered_tokens = splitByRegEx(reviews)
    filtered_tokens = removeStopWordsToken(unfiltered_tokens)
    stemmed_tokens = stemming(filtered_tokens)
    merged_tokens = mergeCommonAbbr(stemmed_tokens)
    countTokens(merged_tokens)
    merged_tokens = createTokenList(merged_tokens, reviews_ids)
    tokens, id, stemmed_sentences = recreateSentence(merged_tokens)
    createVectors(stemmed_sentences, 2, 2, tokens)

if __name__ == "__main__":
    main()