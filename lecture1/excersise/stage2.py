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
def loadData():
    #Open the CSV File
    #Read all rows in it
    #Return the Review Texts

# Remove stop word tokens by using NLTKs English stop word library
def removeStopwords():
    #Load the stopwords from nltk.corpus
    # filter for the stopwords in the review texts
    # return filtered set

# Split up reviews into tokens by using NLTKs regular expression tokenizer with a custom regex statement
def splitByRegEx():
    #Use RegEx [a-zA-Záéíóú]+ to split tokens by nltk RegexpTokenizer 
    #Loop every review and apply tokenizer
    #Return unfiltered, but splitted tokens

# Remove stop word tokens
def removeStopWordsToken():
    #loop through all unfiltered Tokens
    #use removeStopwords method 
    #return all stopword-filtered tokens

# Stem the remaining words
def stemming():
    #Use the english snowball stemmer from nltk.stem
    #loop through all filtered tokens and stem them
    #return all stemmed tokens

# Merge common multi word expressions into one token
def mergeCommonAbbr():
    #Define words, that occur together often (e.g.), but that will most probably be split up (to 'e' and 'g')
    #use the MWETokenizerfrom nltk
    #loop each review and search for those words
    #return all merged tokens

#Count the total Number of Tokens
def countTokens():
    #loop all tokens and count the tokens included

#Create List of all tokens and IDs
def createTokenList():
    #run through the merged tokens and create the token list with the ids
    #insert in the review ids into the list of tokens
    #return the list

#recreate the sentences with the preprocessed words
def recreateSentence():
    #run through all tokens and add them as a sentence to a list
    #use the set() method for the tokens and the id
    #return tokjens, ids and stemmed_sentences

#Save the created Vectors
def saveData():
    #save the word vectors to a file
    #save the word neighbours to a file
    #save the one hot encoding to a file

# function to convert numbers to one hot vectors
def to_one_hot_encoding():
    #create a one hot encoding

#create Vectors by using a NN
def createVectors():
    #we will create a NN to do Word2Vec
    
    #loop through all words and enumerate them
    
    #split all review texts by " "

    #loop all sentences and each word in it. Create the word neighbours in the set window size and append all neighbour combinations to an array 
    
    #create an empty Pandas Data Frame for input and label
    
    #one hot encoding equals the total words
    
    #create input and target arrays and fill them with the one hot encoding words
    
    # convert the arrays to numpy arrays
    
    # making placeholders for X_train and Y_train

    # create the tf variables for the hidden layer in the NN
    
    # create the tf variables for the output layer in the NN
    
    # create the loss function with cross entropy
    
    # start the training operation with GradientDescentOptimizer and 25.000 iterations
    
    #read the vectors from the hidden layer, as they are our vectors
    
    #save the vectors in a data frame

    #Plot a 2D-Model

    #save the Data into files

#run the process for Word2Vec
def main():
    #run the required methods from above

if __name__ == "__main__":
    main()