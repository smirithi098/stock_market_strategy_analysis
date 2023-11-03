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

def create_freq_table(text):

    # tokenize the text into sentences
    all_sentences = sent_tokenize(text)
    no_of_sentences = len(all_sentences)
    print(f"Number of sentences in text: {no_of_sentences}")

    # empty dictionary for frequency matrix
    frequency_matrix = {}

    # get the stop words for english language
    stop_words = set(stopwords.words("english"))

    # initialize the object for stemmer class
    stemmer = PorterStemmer()

    """Stem the words and check if they are in the stop words. 
    If not, increase the count of that word in the frequency table"""
    for i, sentence in enumerate(all_sentences, 1):
        # empty dictionary to store the word count of every word in the text
        frequency_table = {}

        # tokenize the sentence into words
        words = word_tokenize(sentence)

        for word in words:
            word = stemmer.stem(word.lower())

            if word in stop_words:
                continue

            if word in frequency_table:
                frequency_table[word] += 1
            else:
                frequency_table[word] = 1

        frequency_matrix[i] = frequency_table

    return frequency_matrix

#%% Create the Term frequency matrix (TF)

def create_tf_matrix(freq_mat):
    term_frequency_matrix = {}

    for sentence, freq_table in freq_mat.items():
        term_frequency_table = {}

        total_words_in_sentence = len(freq_table)

        for word, count in freq_table.items():
            term_frequency_table[word] = count / total_words_in_sentence

        term_frequency_matrix[sentence] = term_frequency_table

    return term_frequency_matrix

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
