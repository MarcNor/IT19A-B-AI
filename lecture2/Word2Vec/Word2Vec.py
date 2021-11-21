#Manage the Imports
from gensim.models import Word2Vec
import csv
import re
import nltk
nltk.download('punkt')
from nltk import stem
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


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

#Create Word2Vec Model and train it
def trainWord2Vec(merged_tokens):
    import time
    start = time.time()

    model = Word2Vec(merged_tokens, min_count=1,vector_size=2,workers=4, epochs=10)

    end = time.time()
    time = (end - start)/60
    print("Vectors are created. It took " + str(time) + " minutes.")
    print()
    
    similar_words = model.wv.most_similar('man')	
    print()
    print("Similar words to man: ")
    print(similar_words)

    #Get all unique tokens
    allToken = []
    for i in range(len(merged_tokens)):
        for token in merged_tokens[i]:
            allToken.append(token)
    
    #get the vectors for the words
    vectors = []
    for word in set(allToken):
        vector = model.wv[word]
        vec_tmp = str(vector[0]) + "," + str(vector[1]) + "," + word 
        vectors.append(vec_tmp)

    #save the vectors for the words
    with open("Word2Vec2d.csv", 'w') as file:
        file.write("x_vector,y_vector,word")
        file.write('\n')
        for vector in vectors:
            file.write(vector)
            file.write('\n')

def main():
    print("")
    print("")
    print("--------------------------------------")
    print("Let us try out Word2Vec in Gensim!")
    reviews, reviews_ids = loadData('yelp_short.csv', 'text', 'review_id', ',')
    unfiltered_tokens = splitByRegEx(reviews)
    filtered_tokens = removeStopWordsToken(unfiltered_tokens)
    stemmed_tokens = stemming(filtered_tokens)
    merged_tokens = mergeCommonAbbr(stemmed_tokens)
    countTokens(merged_tokens)
    trainWord2Vec(merged_tokens)

if __name__ == "__main__":
    main()