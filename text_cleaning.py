#%% import libraries
import math
import string
import re
from collections import Counter

import numpy as np
import pandas as pd

# from nltk.corpus import stopwords
# from nltk import sent_tokenize, word_tokenize, PorterStemmer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

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

#%% Function to get the summary

def get_summary(text, n):
    # load the spacy model
    summarizer = spacy.load('en_core_web_sm')

    document = summarizer(text)

    # tokenize the document into words
    words = [word.text for word in document]

    frequency_table = {}

    # get the count of each word in the document
    for word in document:
        word = word.text
        if word.lower() not in list(STOP_WORDS) and word.lower() not in string.punctuation:
            if word not in frequency_table.keys():
                frequency_table[word] = 1
            else:
                frequency_table[word] += 1

    # get the maximum count from the table
    max_freq = max(frequency_table.values())

    # normalize the word frequency

    for word in frequency_table.keys():
        frequency_table[word] = frequency_table[word] / max_freq

    # get the score of each sentence depending on how often a word occurs in that sentence
    all_sentences = [sent for sent in document.sents]

    sentence_scores = {}

    for sentence in all_sentences:
        for word in sentence:
            word = word.text.lower()
            if word in frequency_table.keys():
                if sentence not in sentence_scores.keys():
                    sentence_scores[sentence] = frequency_table[word]
                else:
                    sentence_scores[sentence] = frequency_table[word]

    no_of_sentences = int(len(all_sentences)*n)

    # get the summary with sentences with maximum score

    summary = nlargest(n=no_of_sentences, iterable=sentence_scores, key=sentence_scores.get)

    output = [word.text for word in summary]

    final_summary = " ".join(output)

    return final_summary

#%% Call the function for all transcripts
def summarization():

    summary_percentage = {transformed_scripts[0]: 0.075, transformed_scripts[1]: 0.19, transformed_scripts[2]: 0.4,
                          transformed_scripts[3]: 0.12}

    for key, value in summary_percentage:
        result = get_summary(key, value)

        print("*"*30)
        print(f"length of original text: {len(key)}")
        print("*"*30)
        print(f"Summary of the given text:\n{result}")
        print("*"*30)
        print(f"length of summary text: {len(result)}")


