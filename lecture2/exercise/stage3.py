# Manage the Imports
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

# Load the Data
def loadData(file, rowToFilter1, rowToFilter2, delimiterToFilter, maxTweets):
    # Load the train CSV file with the tweets and (optional) give a maximum to load
    # Return tweets, tweet IDs and tweet sentiment

# Create a list of IDs and of Sentiments
def createIdList(tweet_ids):
    # Write the IDs of the tweets to a txt file

def createSentimentList(tweets_sentiments):
    # Write all Sentiments to a txt file

# Do the preprocessing by:
#   Well, it worked last time, so why not doing it quite similar to this one?

# Train a Word2Vec Model and save the vectors 
def trainWord2Vec(merged_tokens):
    # Use the Word2Vec Gensim Model
    # Hand in all merged tokens and let it run
    model = Word2Vec(merged_tokens, min_count=1,vector_size=2,workers=4, epochs=1000)
    # Get all unique tokens
    # Save for each token the vector to a csv file with ID and the word itself
    with open("Word2Vec2d.csv", 'w') as file:
        file.write("id,x_vector,y_vector,word")
        file.write('\n')
        for vector in vectors:
            file.write(vector)
            file.write('\n')

def trainDoc2Vec(tagged_data, stemmed_sentences, method, tweet_rating):
    # Use the Doc2Vec Gensim Model
    # Hand in all tagged data and stemmed sentences, then let it run
    max_epochs = 15
    vec_size = 2
    alpha = 0.025

    model = Doc2Vec(vector_size=vec_size,
                    alpha=alpha, 
                    min_alpha=0.00025,
                    min_count=1,
                    dm =method)
    
    model.build_vocab(tagged_data)

    model.train(tagged_data,
                    total_examples=model.corpus_count,
                    epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    # Get for every ID the vectors out of the model
    # Get for every ID the vectors out of the model and save them to a csv file with ID and the sentiment
    filename = "doc2vec_dbow_2d.csv" if method == 0 else "doc2vec_dm_2d.csv"
    with open(filename, 'w') as file:
        file.write("tweet_id,x_vector,y_vector,sentiment")
        file.write('\n')
        for vector in vectors:
            file.write(vector)
            file.write('\n')

# Implement a K-Means cluster algorithm
def kMeansWord2Vec(filename):
    # Read the vectors CSV file for Word2Vec
    # Use the algorithm from the lecture
    # Plot it

def kMeansDoc2Vec(filename):
    # Read the vectors CSV file for Word2Vec
    # Replace the sentiment by numbers, e.g. 0 to 4
    # Use the algorithm from the lecture
    # Plot it

# Implement DBSCAN
def dbscan(filename, eps):
    # Read the CSV file
    # Use the algorithm from the lecture
    # Plot it

# Implement Support Vector Machine
def svm(filename):
    # Read the vectors CSV file for Word2Vec
    # Replace the sentiment by numbers, e.g. 0 to 4
    # Use the algorithm from the lecture
    # Plot it

# Run all functions in the right order
def main():
    # Run all functions required

if __name__ == "__main__":
    main()