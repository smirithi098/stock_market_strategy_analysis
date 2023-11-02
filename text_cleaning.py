#%% import libraries
import string
import re

import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from heapq import nlargest

#%% Import the data

transcript_df = pd.read_csv("S:/Dissertation 2023/Stock market analysis/"
                            "stock_market_strategy_analysis/data_files/transcript.csv")


#%% Function to clean the text

def clean_transcript(text):
    text = re.sub(r'(\xa0|\n)', ' ', text)
    return text

#%% Call the function to get rid of new lines and special characters

transformed_scripts = []
for script in transcript_df['transcript']:
    transformed_scripts.append(clean_transcript(script))

#%% Function to generate frequency table for all words in all sentences of the paragraph

def create_freq_table(words_list, upper_thresh, lower_thresh):
    # get the stop words for english language
    stop_words = set(stopwords.words("english"))

    # initialize the object for stemmer class
    stemmer = PorterStemmer()

    # empty dictionary to store the word count of every word in the text
    frequency_table = {}

    """Stem the words and check if they are in the stop words. 
    If not, increase the count of that word in the frequency table"""
    for word in words_list:
        for w in word:
            w = stemmer.stem(w)

            if w in stop_words:
                continue
            if w in frequency_table:
                frequency_table[w] += 1
            else:
                frequency_table[w] = 1

    # get the maximum frequency value from the entire text
    max_frequency = float(max(frequency_table.values()))

    freq_copy = frequency_table.copy()

    for key in freq_copy.keys():
        frequency_table[key] = frequency_table[key]/max_frequency

        if frequency_table[key] >= upper_thresh or frequency_table[key] <= lower_thresh:
            del frequency_table[key]

    return frequency_table

#%% Function to get the ranking of individual sentences in the text

def get_ranking(text_rank, num):
    return nlargest(num, text_rank, key=text_rank.get)

#%% Function to generate the summary of the text

def create_summary(text, n):
    sentences = sent_tokenize(text)

    words = [word_tokenize(sent.lower()) for sent in sentences]

    frequency = create_freq_table(words, 0.7, 0.1)
    rank = {}

    for i, sentence in enumerate(words):
        for word in sentence:
            if word in frequency:
                rank[i] += frequency[word]

    sent_ranking = get_ranking(rank, n)

    return [sentences[j] for j in sent_ranking]

#%% Call the summary function

summaries = []

for script in transformed_scripts:
    summaries.append(create_summary(script, 1))
