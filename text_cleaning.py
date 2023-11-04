#%% import libraries
import math
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

#%% I. Function to generate frequency table for all words in all sentences of the paragraph

def create_freq_table(all_sentences):

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

#%% II. Create the Term frequency matrix (TF)

def create_tf_matrix(freq_mat):
    term_frequency_matrix = {}

    for sentence, freq_table in freq_mat.items():
        term_frequency_table = {}

        total_words_in_sentence = len(freq_table)

        for word, count in freq_table.items():
            term_frequency_table[word] = count / total_words_in_sentence

        term_frequency_matrix[sentence] = term_frequency_table

    return term_frequency_matrix

#%% III. Function to get number of sentences with a word

def get_sentences_with_word(freq_mat):
    word_in_sentence_table = {}

    for sentence, freq_table in freq_mat.items():
        for word in freq_table.keys():
            if word in word_in_sentence_table:
                word_in_sentence_table[word] += 1
            else:
                word_in_sentence_table = 1

    return word_in_sentence_table

#%% IV. Function to get the IDF matrix (Inverse document frequency)

def create_idf_matrix(freq_mat, num_documents_per_word, total_sentences):
    idf_matrix = {}

    for sentence, freq_table in freq_mat.items():
        idf_table = {}

        for word in freq_table.keys():
            idf_table[word] = math.log10(total_sentences / float(num_documents_per_word[word]))

        idf_matrix[sentence] = idf_table

    return idf_matrix

#%% V. function to create the tf-idf matrix

def create_tf_idf_matrix(tf_mat, idf_mat):
    tf_idf_matrix = {}

    for (tf_sent, tf_table), (idf_sent, idf_table) in zip(tf_mat.items(), idf_mat.items()):
        tf_idf_table = {}

        for (tf_word, tf_count), (idf_word, idf_count) in zip(tf_table.items(), idf_table.items()):
            tf_idf_table[tf_word] = float(tf_count * idf_count)

        tf_idf_matrix[tf_sent] = tf_idf_table

    return tf_idf_matrix

#%% VI. Function to rank the sentences based on the word frequency in every sentence

def rank_sentence(tf_idf_mat):
    sentence_rank = {}

    for sentence, freq_table in tf_idf_mat.items():
        total = 0

        total_words_in_sentence = len(freq_table)

        for word, count in freq_table.items():
            total += count

        sentence_rank[sentence] = total / total_words_in_sentence

    return sentence_rank

#%% VII. Function to get the average score to identify threshold

def get_average_rank(ranking):
    sum_rank = 0

    for i in ranking:
        sum_rank += ranking[i]

    average_rank = (sum_rank / len(ranking))

    return average_rank

#%% VIII. Function to create the summary fo the paragraph

def create_summary(sentences, sent_rank, threshold):
    count = 0
    summary = ""

    for index, sentence in enumerate(sentences, 1):
        if index in sent_rank and sent_rank[index] >= threshold:
            summary += " " + sentence
            count += 1

    return summary
#%% Function to call all above steps to get the summary

def get_summary(text):
    # 1. Tokenize the text into sentences
    all_sentences = sent_tokenize(text)
    no_of_sentences = len(all_sentences)
    print(f"Number of sentences in text: {no_of_sentences}")

    # 2. Create the frequency matrix for every word in each sentence
    frequency_matrix = create_freq_table(all_sentences=all_sentences)
    print(f"Frequency matrix\n{frequency_matrix}")

    # 3. Term frequency matrix - occurence of a word in a sentence / total words in a sentence
    tf_matrix = create_tf_matrix(freq_mat=frequency_matrix)
    print(f"Term frequency matrix\n{tf_matrix}")

    # 4. Get the count of sentences containing a particular word
    no_of_sentences_with_word = get_sentences_with_word(freq_mat=frequency_matrix)
    print(f"Sentences with a particular word count\n{no_of_sentences_with_word}")

    # 5. Inverse document frequency matrix
    idf_matrix = create_idf_matrix(freq_mat=frequency_matrix, num_documents_per_word=no_of_sentences_with_word,
                                   total_sentences=no_of_sentences)
    print(f"Idf matrix\n{idf_matrix}")

    # 6. TF-IDF Matrix
    tf_idf_matrix = create_tf_idf_matrix(tf_mat=tf_matrix, idf_mat=idf_matrix)
    print(f"TF-IDF matrix\n{tf_idf_matrix}")

    # 7. Rank the sentences
    sentence_ranking = rank_sentence(tf_idf_mat=tf_idf_matrix)
    print(f"Ranking of sentences:\n{sentence_ranking}")

    # 8. get the threshold value
    cut_off = get_average_rank(ranking=sentence_ranking)
    print(f"Threshold value to select important sentences: {cut_off}")

    # 9. Generate the summary of the given text
    final_summary = create_summary(sentences=all_sentences, sent_rank=sentence_ranking, threshold=(1.25 * cut_off))
    print(f"Summary of the text is:\n{final_summary}")


