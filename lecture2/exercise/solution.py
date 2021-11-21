#Manage the Imports
from gensim.models.doc2vec import Doc2Vec, TaggedDocument, Word2Vec
import csv
import re
import nltk
nltk.download('punkt')
from nltk import stem
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt


# Load data from file
def loadData(file, rowToFilter1, rowToFilter2, delimiterToFilter, maxTweets):
    tweets = []
    tweet_ids = []
    tweet_rating = []
    count = 0
    with open(file, 'r', encoding='mac_roman', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiterToFilter)
        for row in reader:
            tweets.append(row[rowToFilter1])
            tweet_ids.append(count)
            tweet_rating.append(row[rowToFilter2])
            count += 1
            if count > maxTweets:
                break
    print("Data were load from CSV File")
    return tweets, tweet_ids, tweet_rating

#Create a List of tweet IDs
def createIdList(tweet_ids):
    with open('tweetIds.txt', mode='w', newline='') as f:
        for tweet_id in tweet_ids:
            f.write(str(tweet_id) + '\n')

#Create a List of tweet Sentiments
def createSentimentList(tweets_sentiments):
    with open('tweetRatings.txt', mode='w', newline='') as f:
        for tweet_star in tweets_sentiments:
            f.write(tweet_star + '\n')

# Remove stop word tokens by using NLTKs English stop word library
def removeStopwords(stringset):
    stop_words = set(stopwords.words("english"))
    filtered_set = []
    for word in stringset:
        if word.lower() not in stop_words:
            filtered_set.append(word)
    return filtered_set

# Split up tweets into tokens by using NLTKs regular expression tokenizer with a custom regex statement
def splitByRegEx(tweets):
    unfiltered_tokens = []
    regex_statement = '[a-zA-Záéíóú]+'
    tokenizer = tokenize.RegexpTokenizer(regex_statement)

    for tweet in tweets:
        unfiltered_tokens.append(tokenizer.tokenize(tweet))
    print("Split by RegEx")
    return unfiltered_tokens

# Remove stop word tokens
def removeStopWordsToken(unfilteredTokens):
    filtered_tokens = []
    for tweet in unfilteredTokens:
        filtered_tokens.append(removeStopwords(tweet))
    print("Stopwords removed")
    return filtered_tokens

# Stem the remaining words
def stemming(filteredTokens):
    stemmed_tokens = []
    stemmer = stem.snowball.EnglishStemmer()
    for tweet in filteredTokens:
        stemmed_tokens_temp = []
        for token in tweet:
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
    for tweet in stemmedTokens:
        merged_tokens.append(tokenizer.tokenize(tweet))
    print("Merged the common Word Expressions")
    return merged_tokens

#Count the total Number of Tokens
def countTokens(allTokens):
    token_count = 0
    for tweet in allTokens:
        if len(tweet)>0:
            for token in tweet:
                token_count+=1
    print("Number of all tokens: ", token_count)

#Create List of all tokens and IDs
def createTokenList(merged_tokens, tweet_ids):
    for i in range(0, len(merged_tokens)):
        merged_tokens[i].insert(0, tweet_ids[i])
    print("Token List is created")
    return merged_tokens

#recreate the sentences with the preprocessed words and give it an ID
def recreateSentence(merged_tokens):
    stemmed_sentences = []
    id = []
    i = 1
    for tweet in merged_tokens:
        tmp = ""
        tweet.pop(0)
        for word in tweet:
            tmp += str(word)+" "
            id.append(i)
            i += 1
        stemmed_sentences.append(tmp)

    #give every document an id
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(stemmed_sentences)]
    return tagged_data, stemmed_sentences

# Save preprocessed data to .csv file, using ',' as delimiter between 2 tokens and quoting each token with '
def saveSentences(merged_sentences):
    with open('preprocessedTweets.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar="\'", quoting=csv.QUOTE_ALL)
        for tweet in merged_sentences:
            csv_writer.writerow(tweet)

    print("Wrote dataset to ", csv_file.name)

#Create Doc2Vec Model and train it
def trainDoc2Vec(tagged_data, stemmed_sentences, method, tweet_rating):
    #train the doc2vec model
    max_epochs = 15
    vec_size = 2
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1,
                    dm =method)
    
    model.build_vocab(tagged_data)

    import time
    start = time.time()

    for epoch in range(max_epochs):
        if epoch % 100 == 0:
            percent = (epoch/max_epochs)*100
            print(str(round(percent,1)) + "%")
        model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha

    end = time.time()
    time = (end - start)/60
    print("Vectors are created. It took " + str(time) + " minutes.")
    print()

    #get the definition names from the list
    tweets = []
    with open("tweetIds.txt") as file:
        tweets = file.read().splitlines()

    #get the vectors out of the model, connect it with the definition names and safe the data
    vectors = []
    for i in range(len(stemmed_sentences)):
        vector = model.dv[str(i)]
        vec_tmp = tweets[i] + "," + str(vector[0]) + "," + str(vector[1]) + "," + str(tweet_rating[i])
        vectors.append(vec_tmp)

    filename = "doc2vec_dbow_2d.csv" if method == 0 else "doc2vec_dm_2d.csv"
    with open(filename, 'w') as file:
        file.write("tweet_id,x_vector,y_vector,sentiment")
        file.write('\n')
        for vector in vectors:
            file.write(vector)
            file.write('\n')

def trainWord2Vec(merged_tokens):
    import time
    start = time.time()

    model = Word2Vec(merged_tokens, min_count=1,vector_size=2,workers=4, epochs=1000)

    end = time.time()
    time = (end - start)/60
    print("Vectors are created. It took " + str(time) + " minutes.")

    #Get all unique tokens
    allToken = []
    for i in range(len(merged_tokens)):
        for token in merged_tokens[i]:
            allToken.append(token)
    
    #get the vectors for the words
    vectors = []
    count = 0
    for word in set(allToken):
        vector = model.wv[word]
        vec_tmp = str(count) + "," + str(vector[0]) + "," + str(vector[1]) + "," + word 
        vectors.append(vec_tmp)

    #save the vectors for the words
    with open("Word2Vec2d.csv", 'w') as file:
        file.write("id,x_vector,y_vector,word")
        file.write('\n')
        for vector in vectors:
            file.write(vector)
            file.write('\n')

def kMeansDoc2Vec(filename):
    df = pd.read_csv(filename, sep=",", usecols=[1,2,3])
    df_sentiment = pd.read_csv(filename, sep=",", usecols=[3])
    print(df.head(2))
    print("-----------")
    print("How are the tweet sentiments distributed over the corpus?")
    print(pd.value_counts(df_sentiment.values.ravel()))

    #Change the string values into Integers
    df['sentiment'] = df['sentiment'].replace(['Extremely Negative','Negative', 'Neutral', 'Positive', 'Extremely Positive'],[0,1,2,3,4])
    print(df.head(2))

    #Use the K-Means estimator algorithm
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1,12)).fit(df)
    visualizer.show()

    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0).fit(df)

    #Plot the cluster with Numbers as label
    sns.scatterplot(data=df, x="x_vector", y="y_vector", hue=kmeans.labels_)
    for i in range(df.shape[0]):
        plt.text(x=df.x_vector[i]+0.1,y=df.y_vector[i]+0.1,s=df.sentiment[i], 
            fontdict=dict(color='red',size=10),
            bbox=dict(facecolor='yellow',alpha=0.5))
    plt.show()

    #Plot the cluster with centroids in it
    sns.scatterplot(data=df, x="x_vector", y="y_vector", hue=kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
                marker="X", c="r", s=80, label="centroids")
    plt.legend()
    plt.show()

def dbscan(filename, eps):
    #prepare: read vector data
    data_df = pd.read_csv(filename, sep=",", usecols=[0,1,2])
    vectors_arr = data_df[data_df.columns[1:]].values

    # Compute DBSCAN
    db = DBSCAN(eps=eps).fit(vectors_arr)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Plot result

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Transparent black used for noise.
            col = [0, 0, 0, 0]

        class_member_mask = (labels == k)

        xy = vectors_arr[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k')

        xy = vectors_arr[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k')

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

def kMeansWord2Vec(filename):
    df = pd.read_csv(filename, sep=",", usecols=[1,2])
    df_words = pd.read_csv(filename, sep=",", usecols=[3])
    print(df.head(2))

    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1,12)).fit(df)
    visualizer.show()

    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0).fit(df)

    sns.scatterplot(data=df, x="x_vector", y="y_vector", hue=kmeans.labels_)
    for i in range(df.shape[0]):
        plt.text(x=df.x_vector[i]+0.1,y=df.y_vector[i]+0.1,s=df_words.word[i], 
            fontdict=dict(color='red',size=10),
            bbox=dict(facecolor='yellow',alpha=0.5))
    plt.show()

    sns.scatterplot(data=df, x="x_vector", y="y_vector", hue=kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
                marker="X", c="r", s=80, label="centroids")
    plt.legend()
    plt.show()

def svm(filename):
    from sklearn import svm
    
    df = pd.read_csv(filename, sep=",", usecols=[1,2,3])
    df['sentiment'] = df['sentiment'].replace(['Extremely Negative','Negative', 'Neutral', 'Positive', 'Extremely Positive'],[0,1,2,3,4])

    X = df[['x_vector','y_vector']]
    y = df['sentiment']

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X.values, y.values) 

    # Plot Decision Region using mlxtend's awesome plotting function
    plot_decision_regions(X=X.values, 
                        y=y.values,
                        clf=clf, 
                        legend=2)

    # Update plot object with X/Y axis labels and Figure Title
    plt.xlabel(X.columns[0], size=14)
    plt.ylabel(X.columns[1], size=14)
    plt.title('SVM Decision Region Boundary', size=16)

    plt.show()

def main():
    tweets, tweet_ids, tweet_rating = loadData('Corona_NLP_train.csv', 'OriginalTweet', 'Sentiment', ',', 500)
    createIdList(tweet_ids)
    createSentimentList(tweet_rating)
    unfiltered_tokens = splitByRegEx(tweets)
    filtered_tokens = removeStopWordsToken(unfiltered_tokens)
    stemmed_tokens = stemming(filtered_tokens)
    merged_tokens = mergeCommonAbbr(stemmed_tokens)
    countTokens(merged_tokens)
    print("First: Word2Vec")
    trainWord2Vec(merged_tokens)
    kMeansWord2Vec('Word2Vec2d.csv')
    dbscan('Word2Vec2d.csv', 0.05)
    merged_tokens = createTokenList(merged_tokens, tweet_ids)
    saveSentences(merged_tokens)
    tagged_data, stemmed_sentences = recreateSentence(merged_tokens)
    print("Now Doc2Vec")
    print("Distributed Bag of Words")
    trainDoc2Vec(tagged_data, stemmed_sentences, 0, tweet_rating) #0 for distributed Bag of Words, 1 for distributed Memory
    kMeansDoc2Vec('doc2vec_dbow_2d.csv')
    dbscan('doc2vec_dbow_2d.csv', 0.1)    
    svm('doc2vec_dbow_2d.csv')
    print("Distributed Memory")
    trainDoc2Vec(tagged_data, stemmed_sentences, 1, tweet_rating)
    kMeansDoc2Vec('doc2vec_dm_2d.csv')
    dbscan('doc2vec_dm_2d.csv', 0.2)
    svm('doc2vec_dm_2d.csv')

if __name__ == "__main__":
    main()