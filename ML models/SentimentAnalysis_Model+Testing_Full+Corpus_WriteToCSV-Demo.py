
# coding: utf-8

# In[1]:

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import twitter


# In[2]:

# use the training data to test the model

def createTestData(TestFile):
    import csv
    testData=[]
    with open(TestFile,'r') as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        for row in lineReader:
            testData.append({"text":row[0],"label":None})
    return testData


# In[3]:

# TestFile="C:\Users\songsu\Desktop\Sentiment Analysis\TestFile.csv"


# In[4]:

# testData=createTestData(TestFile)


# In[5]:

# testData[0:9]


# In[6]:

def createTrainingCorpus(corpusFile):
    import csv
    trainingData=[]
    with open(corpusFile,'r') as csvfile:
        lineReader = csv.reader(csvfile,delimiter=',',quotechar="\"")
        for row in lineReader:
            trainingData.append({"text":row[0],"label":row[1]})
    return trainingData


# In[7]:

corpusFile="C:\Users\songsu\Desktop\Training Data\Compiled.csv"
trainingData=createTrainingCorpus(corpusFile)


# In[8]:

# trainingData[0:9]


# In[9]:

corpusFileSVM="C:\Users\songsu\Desktop\Training Data\Compiled_SVM.csv"
trainingDataSVM=createTrainingCorpus(corpusFileSVM)


# In[10]:

trainingDataSVM[0:9]


# In[11]:

import re
from nltk.tokenize import word_tokenize
from string import punctuation 
from nltk.corpus import stopwords 


# In[12]:

class PreProcessTweets:
    def __init__(self):
        self._stopwords=set(stopwords.words('english')+list(punctuation)+['AT_USER','URL'])
        
    def processTweets(self,list_of_tweets):
        # The list of tweets is a list of dictionaries which should have the keys, "text" and "label"
        processedTweets=[]
        # This list will be a list of tuples. Each tuple is a tweet which is a list of words and its label
        for tweet in list_of_tweets:
            processedTweets.append((self._processTweet(tweet["text"]),tweet["label"]))
        return processedTweets
 

    def _processTweet(self,tweet):
        # 1. Convert to lower case
        tweet=tweet.lower()
        # 2. Replace links with the word URL 
        tweet=re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)     
        # 3. Replace @username with "AT_USER"
        tweet=re.sub('@[^\s]+','AT_USER',tweet)
        # 4. Replace #word with word 
        tweet=re.sub(r'#([^\s]+)',r'\1',tweet)
        # You can do further cleanup as well if you like, replace 
        # repetitions of characters, for ex: change "huuuuungry" to "hungry"
        tweet=word_tokenize(tweet.decode('utf-8'))
        # This tokenizes the tweet into a list of words 
        # Let's now return this list minus any stopwords 
        # Remove the line breakers and carriage returns. The string '\n' represents newlines.
        # And \r represents carriage returns 
        return [word for word in tweet if word not in self._stopwords]
    


# In[13]:

tweetProcessor=PreProcessTweets()


# In[14]:

ppTrainingData=tweetProcessor.processTweets(trainingData)


# In[15]:

ppTrainingDataSVM=tweetProcessor.processTweets(trainingDataSVM)


# In[16]:

import nltk 
# Naive Bayes Classifier - We'll use NLTK's built in classifier to perform the classification

# First build a vocabulary 
def buildVocabulary(ppTrainingData):
    all_words=[]
    for (words,sentiment) in ppTrainingData:
        all_words.extend(words)
    # This will give us a list in which all the words in all the tweets are present
    # These have to be de-duped. Each word occurs in this list as many times as it 
    # appears in the corpus 
    wordlist=nltk.FreqDist(all_words)
    # This will create a dictionary with each word and its frequency
    word_features=wordlist.keys()
    # This will return the unique list of words in the corpus 
    return word_features

# NLTK has an apply_features function that takes in a user-defined function to extract features 
# from training data. We want to define our extract features function to take each tweet in 
# The training data and represent it with the presence or absence of a word in the vocabulary 

def extract_features(tweet):
    tweet_words=set(tweet)
    features={}
    for word in word_features:
        features['contains(%s)' % word]=(word in tweet_words)
        # This will give us a dictionary , with keys like 'contains word1' and 'contains word2'
        # and values as True or False 
    return features 


# In[17]:

# Now we can extract the features and train the classifier 
word_features = buildVocabulary(ppTrainingData)
trainingFeatures=nltk.classify.apply_features(extract_features,ppTrainingData)
# apply_features will take the extract_features function we defined above, and apply it it 
# each element of ppTrainingData. It automatically identifies that each of those elements 
# is actually a tuple , so it takes the first element of the tuple to be the text and 
# second element to be the label, and applies the function only on the text 


# In[ ]:

NBayesClassifier=nltk.NaiveBayesClassifier.train(trainingFeatures)
# We now have a classifier that has been trained using Naive Bayes


# In[ ]:

# Support Vector Machines 
from nltk.corpus import sentiwordnet as swn
import numpy as np 
from sklearn.feature_extraction.text import CountVectorizer 

# We have to change the form of the data slightly. SKLearn has a CountVectorizer object. 
# It will take in documents and directly return a term-document matrix with the frequencies of 
# a word in the document. It builds the vocabulary by itself. We will give the trainingData 
# and the labels separately to the SVM classifier and not as tuples. 
# Another thing to take care of, if you built the Naive Bayes for more than 2 classes, 
# SVM can only classify into 2 classes - it is a binary classifier. 

svmTrainingData=[' '.join(tweet[0]) for tweet in ppTrainingDataSVM]
# Creates sentences out of the lists of words 

vectorizer=CountVectorizer(min_df=1)
X=vectorizer.fit_transform(svmTrainingData).toarray()
# We now have a term document matrix 
vocabulary=vectorizer.get_feature_names()

# Now for the twist we are adding to SVM. We'll use sentiwordnet to add some weights to these 
# features 

swn_weights=[]

for word in vocabulary:
    try:
        # Put this code in a try block as all the words may not be there in sentiwordnet (esp. Proper
        # nouns). Look for the synsets of that word in sentiwordnet 
        synset=list(swn.senti_synsets(word))
        # use the first synset only to compute the score, as this represents the most common 
        # usage of that word 
        common_meaning =synset[0]
        # If the pos_Score is greater, use that as the weight, if neg_score is greater, use -neg_score
        # as the weight 
        if common_meaning.pos_score()>common_meaning.neg_score():
            weight=common_meaning.pos_score()
        elif common_meaning.pos_score()<common_meaning.neg_score():
            weight=-common_meaning.neg_score()
        else: 
            weight=0
    except: 
        weight=0
    swn_weights.append(weight)


# Let's now multiply each array in our original matrix with these weights 
# Initialize a list

swn_X=[]
for row in X: 
    swn_X.append(np.multiply(row,np.array(swn_weights)))
# Convert the list to a numpy array 
swn_X=np.vstack(swn_X)


# We have our documents ready. Let's get the labels ready too. 
# Lets map positive to 1 and negative to 2 so that everything is nicely represented as arrays 
labels_to_array={"positive":1,"negative":2}
labels=[labels_to_array[tweet[1]] for tweet in ppTrainingDataSVM]
y=np.array(labels)

# Let's now build our SVM classifier 
from sklearn.svm import SVC 
SVMClassifier=SVC()
SVMClassifier.fit(swn_X,y)


# In[ ]:

# if NBResultLabels.count('positive')>NBResultLabels.count('negative'):
#    print "NB Result Positive Sentiment" + str(100*NBResultLabels.count('positive')/len(NBResultLabels))+"%"
#else: 
#    print "NB Result Negative Sentiment" + str(100*NBResultLabels.count('negative')/len(NBResultLabels))+"%"
    
    
    
    
#if SVMResultLabels.count(1)>SVMResultLabels.count(2):
#    print "SVM Result Positive Sentiment" + str(100*SVMResultLabels.count(1)/len(SVMResultLabels))+"%"
#else: 
#    print "SVM Result Negative Sentiment" + str(100*SVMResultLabels.count(2)/len(SVMResultLabels))+"%"
  


# In[ ]:

api = twitter.Api(consumer_key='Ecllhp6hUpjeFz14Sp84fOTmK',
                 consumer_secret='wWMGD0db7pSbC39TuDFxQ6z84GaMPbck3T04w1JlSTKx19zopy',
                 access_token_key='2455417567-YyIf76YAbN6sEydm4Z2EhXyuwyYtMCf6kTg70Ak',
                 access_token_secret='E0kz81jDnLh2wICvXlAtPVhtwyA6e5JOoCljP3qYu94bO')

# To see if this worked, use the command below, it will print out a bunch of details about your user account
# and that's how you know you're all set to use the API
print(api.VerifyCredentials())


# In[ ]:

def createTestData(search_string):
    try:
        tweets_fetched=api.GetSearch(search_string, count=100,lang='en')
        # This will return a list with twitter.Status objects. These have attributes for 
        # text, hashtags etc of the tweet that you are fetching. 
        # The full documentation again, you can see by typing pydoc twitter.Status at the 
        # command prompt of your terminal 
        print "Great! We fetched "+str(len(tweets_fetched))+" tweets with the term "+search_string+"!!"
        # We will fetch only the text for each of the tweets, and since these don't have labels yet, 
        # we will keep the label empty 
        return [{"text":status.text,"label":None} for status in tweets_fetched]
    except:
        print "Sorry there was an error!"
        return None
    
search_string=input("Hi there! What are we searching for today?")
testData=createTestData(search_string)


# In[ ]:

#get rid of the breaklines and carriage return
for a in testData:
    a['text']=a['text'].replace('\n', '')


# In[ ]:

for a in testData:
    a['text']=a['text'].replace('\r', '')


# In[ ]:

ppTestData=tweetProcessor.processTweets(testData)


# In[ ]:

# First Naive Bayes 
NBResultLabels=[NBayesClassifier.classify(extract_features(tweet[0])) for tweet in ppTestData]



# Now SVM 
SVMResultLabels=[]
for tweet in ppTestData:
    tweet_sentence=' '.join(tweet[0])
    svmFeatures=np.multiply(vectorizer.transform([tweet_sentence]).toarray(),np.array(swn_weights))
    SVMResultLabels.append(SVMClassifier.predict(svmFeatures)[0])
    # predict() returns  a list of numpy arrays, get the first element of the first array 
    # there is only 1 element and array


# In[ ]:

# Write the result to csvfile to be imported to Spotfire
def WriteResultToCSV(tweetsData,ResultLabels,Searchterm, ResultLabelFile):
    import csv
    with open(ResultLabelFile,'wb') as csvfile:
        linewriter=csv.writer(csvfile,delimiter=',',quotechar="\"")
        for row in range(0,len(tweetsData)):
            try:
                #linewriter.writerow(Searchterm,tweetsData[row]['text'], ResultLabels[row])
                linewriter.writerow([Searchterm, ResultLabels[row],tweetsData[row]['text']])
            except Exception:
                print "error"
    return None
    


# In[ ]:

ResultLabelFile='C:/Users/songsu/Desktop/Spotfire_Sentiment Analysis/Result.csv'


# In[ ]:

ResultLabelFileSVM='C:/Users/songsu/Desktop/Spotfire_Sentiment Analysis/ResultSVM.csv'


# In[ ]:

Result=WriteResultToCSV(testData,NBResultLabels,search_string,ResultLabelFile)


# In[ ]:

ResultSVM=WriteResultToCSV(testData,SVMResultLabels,search_string,ResultLabelFileSVM)

