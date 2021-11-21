#Manage the Imports
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import csv
import re
import nltk
nltk.download('punkt')
from nltk import stem
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Load data from file
def loadData(file, rowToFilter1, rowToFilter2, rowToFilter3, delimiterToFilter):
    reviews = []
    reviews_ids = []
    review_stars = []
    with open(file) as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiterToFilter)
        for row in reader:
            reviews.append(row[rowToFilter1])
            reviews_ids.append(row[rowToFilter2])
            review_stars.append(row[rowToFilter3])
    print("Data were load from CSV File")
    return reviews, reviews_ids, review_stars

#Create a List of Review IDs
def createIdList(reviews_ids):
    with open('reviewIds.txt', mode='w', newline='') as f:
        for reviews_id in reviews_ids:
            f.write(reviews_id + '\n')

#Create a List of Review Stars
def createStarList(reviews_stars):
    with open('reviewStars.txt', mode='w', newline='') as f:
        for reviews_star in reviews_stars:
            f.write(reviews_star + '\n')

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
    #print(merged_tokens)
    return merged_tokens

#recreate the sentences with the preprocessed words and give it an ID
def recreateSentence(merged_tokens):
    stemmed_sentences = []
    id = []
    i = 1
    for review in merged_tokens:
        tmp = ""
        review.pop(0)
        for word in review:
            tmp += str(word)+" "
            id.append(i)
            i += 1
        stemmed_sentences.append(tmp)

    #give every document an id
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(stemmed_sentences)]
    return tagged_data, stemmed_sentences

# Save preprocessed data to .csv file, using ',' as delimiter between 2 tokens and quoting each token with '
def saveSentences(merged_sentences):
    with open('preprocessedReviews.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar="\'", quoting=csv.QUOTE_ALL)
        for definition in merged_sentences:
            csv_writer.writerow(definition)

    print("Wrote dataset to ", csv_file.name)

#Create Doc2Vec Model and train it
def trainDoc2Vec(tagged_data, stemmed_sentences, method, review_stars):
    #train the doc2vec model
    max_epochs = 25000
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

    modelname = "doc2vecDBOW.model" if method == 0 else "doc2vecDM.model"
    model.save(modelname)
    print("Model Saved to " + modelname)

    #get the definition names from the list
    reviews = []
    with open("reviewIds.txt") as file:
        reviews = file.read().splitlines()

    #get the vectors out of the model, connect it with the definition names and safe the data
    vectors = []
    for i in range(len(stemmed_sentences)):
        vector = model.dv[str(i)]
        vec_tmp = reviews[i] + "," + str(vector[0]) + "," + str(vector[1]) + "," + str(review_stars[i])
        vectors.append(vec_tmp)

    filename = "doc2vec_dbow_2d.csv" if method == 0 else "doc2vec_dm_2d.csv"
    with open(filename, 'w') as file:
        file.write("review_id,x_vector,y_vector,stars")
        file.write('\n')
        for vector in vectors:
            file.write(vector)
            file.write('\n')

def main():
    print("")
    print("")
    print("--------------------------------------")
    print("Let us try out Doc2Vec!")
    reviews, reviews_ids, review_stars = loadData('yelp_short.csv', 'text', 'review_id', 'stars', ',')
    createIdList(reviews_ids)
    createStarList(review_stars)
    unfiltered_tokens = splitByRegEx(reviews)
    filtered_tokens = removeStopWordsToken(unfiltered_tokens)
    stemmed_tokens = stemming(filtered_tokens)
    merged_tokens = mergeCommonAbbr(stemmed_tokens)
    countTokens(merged_tokens)
    merged_tokens = createTokenList(merged_tokens, reviews_ids)
    saveSentences(merged_tokens)
    tagged_data, stemmed_sentences = recreateSentence(merged_tokens)
    trainDoc2Vec(tagged_data, stemmed_sentences, 0, review_stars) #0 for distributed Bag of Words, 1 for distributed Memory
    trainDoc2Vec(tagged_data, stemmed_sentences, 1, review_stars)

if __name__ == "__main__":
    main()