import os
from os.path import join
from joblib import load
import pandas as pd
import time
import re
import tweepy 
import tkinter
import wordcloud
from tkinter.ttk import *

def calculate_frequencies(file_contents):
    
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    uninteresting_words = ["the","for", "a", "to", "if", "is","not","on", "it", "of", "and", "or", "an", "as",
                           "in", "i", "me", "my", "we", "our", "ours", "you", "your", "yours", "he", "she", 
                           "him", "his", "her", "hers", "its", "they", "them", "their", "what", "which", "who", 
                           "whom", "this", "that", "am", "are", "was", "were", "be", "been", "being", "have", 
                           "has", "had", "do", "does", "did", "but", "at", "by", "with", "from", "here", "when",
                           "where", "how", "all", "any", "both", "each", "few", "more", "some", "such", "no", 
                           "nor", "too", "very", "can", "will", "just", "than"]
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
    
    # wordcloud
    cloud = wordcloud.WordCloud()
    cloud.generate_from_frequencies(frequencies)
    return cloud.to_array()

# Function for pre-processing the tweets
def processTweet(tweet):

    # Convert to lower case
    tweet = tweet.lower()
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    # Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # trim
    tweet = tweet.strip('\'"')
     
    return tweet

def get_tweets_data(api,keyword,count):

    try:
        # Creation of query method using parameters
        tweets = tweepy.Cursor(api.search, q = keyword, result_type = "recent", lang = 'en').items(count)
 
        # Pulling information from tweets iterable object
        tweets_list = [[tweet.created_at, tweet.id, tweet.text] for tweet in tweets]
 
        # Creation of dataframe from tweets list
        tweets_df = pd.DataFrame(tweets_list, columns = ['date_time', 'tweet_id', 'original_tweet'])

        tweets_df['cleaned_tweet'] = tweets_df.apply (lambda row: processTweet(row['original_tweet']), 
                                                      axis = 1)
        
        return tweets_df
     
    except BaseException as e:
        print('failed on_status,', str(e))
        time.sleep(3)

class MainWin():
    def __init__(self, master):
        self.master = master
        # set title
        self.master.title("Hate speech detection GUI")
        # validate function
        # vcmd = self.master.register(self.validate)
        self.userid = ""
        # entry for userid
        self.entry = tkinter.Entry(self.master, text = "Search for tweet")
        # entry for number of tweets
        self.limit = tkinter.Entry(self.master, text = "Enter limit of tweets")
        # button to start
        self.start_button = tkinter.Button(self.master, text = "Classify", command = self.gen_tag)
        # labels
        label1 = tkinter.Label(self.master, text = "Search for tweets on Twitter: ")
        label2 = tkinter.Label(self.master, text = "Number of tweets: ")

        self.single_tweet = tkinter.Entry(self.master, text = "Enter single line of text: ", width = 30)
        single_tweet_label = tkinter.Label(self.master, text = "Single piece of text to classify: ")
        self.hate_speech_label = tkinter.Label(self.master, text = "", fg = 'red')
        self.non_hate_speech_label = tkinter.Label(self.master, text = "", fg = 'green')

        labelTop = tkinter.Label(self.master, text = "Choose trained model:")

        import glob
        # only_files= glob.glob("models/*.mod")
        file_names = [os.path.basename(x) for x in glob.glob("models/*.mod")]
        print(file_names)
        
        # Error checking if models cannot be found
        if len(file_names)== 0:
            file_names.append('No Trained model found')
        else:
            self.start_button.grid(row=5, column=1)

        self.comboExample = tkinter.ttk.Combobox(self.master, values = file_names, state = "readonly")
        self.comboExample.current(0)

        # arrange the UI components
        # instruction.grid(row=0, columnspan=3)
        self.entry.grid(row=1, column=1)
        self.limit.grid(row=2, column=1)
        self.single_tweet.grid(row=3,column=1)
        self.comboExample.grid(row=4, column=1)
        
        label1.grid(row=1, column=0)
        label2.grid(row=2, column=0)
        single_tweet_label.grid(row=3,column=0)
        labelTop.grid(row=4, column=0)


        ## 
        self.hate_speech_label.grid(row=7, column=0)
        self.non_hate_speech_label.grid(row=8, column=0)

    def update(self):
        search_text = self.entry.get()
        if not search_text:
            return False
        else:
            return True

    def check_limit(self):
        search_text = self.limit.get()
         
        if not search_text:
            return False
        else:

            if str.isdigit(search_text):
                return True
            else:
                return False
        
    def gen_tag(self): # function for when button is clicked

        single_text = self.single_tweet.get()
        if single_text:
            # locate model in folder
            cv_path = 'models/' + self.comboExample.get()
            clf = load(cv_path)
            cv_path = cv_path.replace('.mod','.vec')

            # changed from count_vectorizer
            vectorizer = load(cv_path)
            tweets_test_vectors = vectorizer.transform([single_text])

            predictions = clf.predict(tweets_test_vectors)
            # pred_text = ''
            
            
            # Display prediction to user
            if predictions == 0:
                self.hate_speech_label['text'] = ""
                self.non_hate_speech_label['text'] = 'This piece of text is predicted as non-hate speech'
            else:
                self.non_hate_speech_label['text'] = ""
                self.hate_speech_label ['text'] = 'This piece of text is predicted as hate speech'
          
        else:
            # Error checking for empty text field
            if not self.update():
                err = tkinter.Label(
                    self.master, text = "Please provide search text", fg = "red")
                err.grid(columnspan=3)
                return False

            # Error checking for empty text field
            if not self.check_limit():
                err = tkinter.Label(
                    self.master, text = "Please provide a valid limit for tweets as an integer", 
                    fg = "red")
                err.grid(columnspan=3)
                return False

            # Twitter API keys
            consumer_key = "OrpK4U401ohJuyXaRvDjhMER1"
            consumer_secret = "cNYj3uIXV0LDQkTEToFmvKcyW7Z22jDGewurddcyLvalVvp52B"
            access_token = "1311757193335762946-ldOHNe3LLtOu0obhDjlG5qJXu6pQ0j"
            access_token_secret = "Lnagq5vC9NWUUnjmfXRL42aE1WjBmgZaPt8wJqGvVgbH3"
            
            # Tweepy library
            
            # Authentication
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            
            api = tweepy.API(auth, wait_on_rate_limit = True)
                       
            df = get_tweets_data(api,self.entry.get(),int(self.limit.get()))

            tweets_test_data = df['cleaned_tweet'].tolist()

            cv_path = 'models/' + self.comboExample.get()
            clf = load(cv_path)
            cv_path = cv_path.replace('.mod','.vec')

            vectorizer = load(cv_path)
            tweets_test_vectors = vectorizer.transform(tweets_test_data)

            predictions = clf.predict(tweets_test_vectors)
            df['prediction'] = predictions

            hate_speech_tweet_list=df[df["prediction"] == 1]["cleaned_tweet"].tolist()

            hate_speech_tweet = ''
            hate_speech_tweet = hate_speech_tweet.join(hate_speech_tweet_list)

            # Save to Excel file
            fname = self.comboExample.get()
            fname = fname.replace('.mod','.xlsx')
            
            # Save to twitter_hate_speech_detection folder
            df.to_excel("twitter_hate_speech_detection/" + fname)
            
            # Notify user that hate speech detection is complete
            print('Processing done. Results are stored in twitter_classification folder')
  
        return True

GUI_ROOT = tkinter.Tk()
GUI_ROOT.geometry('500x500')
CLASSIFIER_GUI = MainWin(GUI_ROOT)

GUI_ROOT.mainloop()
