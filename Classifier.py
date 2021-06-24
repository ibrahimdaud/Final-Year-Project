# Imports

import os
import sys
import argparse
from joblib import dump

import numpy as np
import pandas as pd

import time
import datetime
import wordcloud
import matplotlib.pyplot as plt
import seaborn as sns

import re, string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

##############################################################################

def calculate_frequencies(file_contents):
    
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    uninteresting_words = ["the","for", "a", "to", "if", "is","not","on", "it", "of", "and", "or", "an", 
                           "as", "in", "i", "me", "my", "we", "our", "ours", "you", "your", "yours", "he", 
                           "she", "him", "his", "her", "hers", "its", "they", "them", "their", "what", 
                           "which", "who", "whom", "this", "that", "am", "are", "was", "were", "be", "been", 
                           "being", "have", "has", "had", "do", "does", "did", "but", "at", "by", "with", 
                           "from", "here", "when", "where", "how", "all", "any", "both", "each", "few", 
                           "more", "some", "such", "no", "nor", "too", "very", "can", "will", "just", 
                           "than"]
    new_content = ""
    frequencies = {}
    for ch in file_contents:
        if ch not in punctuations:
            new_content += ch
    temp_file = new_content.split(" ")
    while "" in temp_file:
        temp_file.remove("")
    for word in temp_file:
        if word.lower() not in uninteresting_words:
            if word in frequencies:
                frequencies[word] += 1
            else:
                frequencies[word] = 1
    
    # generating a Word Cloud
    cloud = wordcloud.WordCloud()
    cloud.generate_from_frequencies(frequencies)
    return cloud.to_array()

# function for pre-processing tweets
def pre_processing(tweet):
    
    # lower casing
    tweet = tweet.lower()
    # removes numbers
    tweet = re.sub(r'\d+', '', tweet)

    # removes whitespace
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    tweet = tweet.strip()

    # stop words removal
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(tweet)
    tweet = [i for i in tokens if not i in stop_words]

    return tweet

# function to clean tweets in dataset by removing punctuation, stop words
def clean_it(df):

    df_label = []
    df_tweet = []

    for ind, row in df.iterrows():

        tweet = row['tweet']
        tweet = pre_processing(tweet)
        
        if len(tweet) > 1:
            separator = ' '
            tweet = separator.join(tweet)
             
            df_tweet.append(tweet)
            df_label.append(row['label'])
            
    # cleaned dataset
    
    df_cleaned = pd.DataFrame(list(zip(df_tweet, df_label)), columns = ["tweet", "label"])

    return df_cleaned



def main(argv):
    
    # User provides three parameters (- h, help function)
    # Error checking to see if parameters have been provided
    
    ts = datetime.date.today()
    
    print('\nTime started at: ' + str(ts))
    
    parser = argparse.ArgumentParser(description = 'Select your three parameters:')
            
    parser.add_argument('model_name', 
                        help = "Please provide a name for the model.")
    
    parser.add_argument('feature_extraction', action = "store", choices = ['TFIDF', 'BOW'], 
                        help = "Please select an option for feature extraction.")
    
    parser.add_argument('under_sampling', action = "store", choices = ['True', 'False'], 
                        help = "Please select an option for under sampling.")

    results = parser.parse_args()
    
    
    
    start_time = time.time()
    
    # Locating dataset
    for dirname, _, filenames in os.walk('dataset'):
        for filename in filenames:
            if filename == "hate_speech_test.csv":
                test_df = pd.read_csv(os.path.join(dirname, filename))
            elif filename == "hate_speech_train.csv":
                train_df = pd.read_csv(os.path.join(dirname, filename))
    
    hate_speech_tweet_list = train_df[train_df["label"] == 1]["tweet"].tolist()
    hate_speech_tweet_list = hate_speech_tweet_list[0:100]
    hate_speech_tweet = ''
    hate_speech_tweet = hate_speech_tweet.join(hate_speech_tweet_list)

    # Display Word Cloud visualisation for hate speech
    myimage = calculate_frequencies(hate_speech_tweet)
    plt.imshow(myimage, interpolation = 'nearest')
    plt.axis('off')
    
    # Saves visualisation to a folder
    plt.savefig('visualisations/word cloud (Hate speech)_' + results.model_name + '_' + str(ts) +
                '.png')
    plt.clf()

    non_hate_speech_tweet_list = train_df[train_df["label"] == 0]["tweet"].tolist()
    non_hate_speech_tweet_list = non_hate_speech_tweet_list[0:100]
    non_hate_speech_tweet = ''
    non_hate_speech_tweet = non_hate_speech_tweet.join(hate_speech_tweet_list)

    # Display your Word Cloud image for non-hate speech
    myimage = calculate_frequencies(non_hate_speech_tweet)
    plt.imshow(myimage, interpolation = 'nearest')
    plt.axis('off')
    
    # Saves visualisation to a folder
    plt.savefig('visualisations/word cloud (Non-hate speech)_-' + results.model_name + '_' + str(ts) +
                '.png')
    plt.clf()

    plt.style.use('fivethirtyeight')

    tot = train_df.shape[0]
    num_hate_speech = train_df[train_df.label == 1].shape[0]

    # Display pie chart visualisation
    
    slices = [num_hate_speech / tot, (tot - num_hate_speech) / tot]
    labeling = ['Hate speech', 'Non-hate speech']
    explode = [0.3, 0]
    plt.pie(slices, explode = explode, shadow = True, autopct = '%1.1f%%', 
            labels = labeling, wedgeprops = {'edgecolor':'black'})
    plt.title('Class distribution')
    plt.tight_layout()
    
    # Saves visualisation to a folder
    plt.savefig('visualisations/Pie chart_' + results.model_name + '_' + str(ts) + '.png')
    plt.clf()
    
    # Cleaning training and testing data via clean_it function

    train_df = clean_it(train_df)
    test_df = clean_it(test_df)

    elapsed_time = time.time() - start_time
    print('\nPre-processing complete!')
    print("Time elapsed: ", elapsed_time, "seconds")    
    
    
    # Training begins
    start_time = time.time()
    print('\nTraining has started...')
    
    # 
    y = train_df["label"]
    
    
    # inputted from the user
    if results.feature_extraction == 'BOW':
        # BOW vectorisation
        # disregard terms that appear often (more than 25% of the documents)
        # disregard terms that appear rarely (less than 5 documents)
        vectorizer = CountVectorizer(max_df = 0.25, min_df = 5)
        # Fitting the training data
        train_vectors = vectorizer.fit_transform(train_df["tweet"])
        # Fitting the testing data
        test_vectors = vectorizer.transform(test_df["tweet"])
        # Saves vectorizer to models folder        
        dump(vectorizer, 'models/' + results.model_name + '-' + 'BOW.vec') 
    else:
        # BOW vectorisation
        vectorizer = TfidfVectorizer(max_df = 0.25, min_df = 5)
        # Fitting the training data
        train_vectors = vectorizer.fit_transform(train_df["tweet"])
        # Fitting the testing data
        test_vectors = vectorizer.transform(test_df["tweet"])
        
        # Saves vectorizer to models folder
        dump(vectorizer, 'models/' + results.model_name+ '-' + 'TFIDF.vec') 

    # If user selects undersampling
    if results.under_sampling:
        under = RandomUnderSampler()
        train_vectors, y = under.fit_resample(train_vectors, y)
        

    # Applying Logistic Regression to the training data
    
    # Hyperparameters are finetuned: max_iter, multi_class
    clf = LogisticRegression(max_iter = 1000, multi_class = "ovr", n_jobs = -1, solver = "saga")
    clf.fit(train_vectors, y)
    
    # Saves models to models folder
    dump(clf, 'models/' + results.model_name + '-' + results.feature_extraction + ".mod") 
    predictions = clf.predict(test_vectors)
    
    # Generating Confusion matrix 
    labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    categories = ['Non-hate speech', 'Hate speech']
    cf_matrix = confusion_matrix(test_df["label"], predictions)
     
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}"for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    heatmap = sns.heatmap(cf_matrix, annot = labels, fmt = '', cmap = 'Blues')
    fig = heatmap.get_figure()
    
    # Saves Confusion matrix visualisation to folder
    fig.savefig('visualisations/Confusion matrix_' + results.model_name + '_' + results.feature_extraction 
                + '_' + str(ts)+'.png')

    elapsed_time = time.time() - start_time
    print("Time elapsed: ", elapsed_time, "seconds")
    
    # print Accuracy score
    print("\nAccuracy:", accuracy_score(test_df["label"], predictions))
    
    # print Error rate
    print("Error rate:", 1 - accuracy_score(test_df["label"], predictions))
    
    # print Classification Report
    print('\nClassification Report for hate speech dataset \n')
    print(classification_report(test_df["label"], predictions, target_names = categories))
    
    print("Confusion matrix for hate speech dataset \n")
    
    cf_matrix_df = pd.DataFrame(cf_matrix, 
                                columns = ['Predicted Negative', 'Predicted Positive'],
                                index = ['Actual Negative', 'Actual Positive'])
    print(cf_matrix_df)

# Display
  
if __name__ == "__main__":
   main(sys.argv[1:])
