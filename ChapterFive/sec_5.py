"""
emotion_preprocess
https://storage.googleapis.com/laurencemoroney-blog.appspot.com/binary-emotion.csv
 """

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup
import string
import csv


if __name__ == '__main__':
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                 "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
                 "do",
                 "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                 "having",
                 "he", "hed", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how",
                 "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "it", "its", "itself",
                 "lets", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",
                 "ought",
                 "our", "ours", "ourselves", "out", "over", "own", "same", "she", "shed", "shell", "shes", "should",
                 "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then",
                 "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through",
                 "to", "too", "under", "until", "up", "very", "was", "we", "wed", "well", "were", "weve", "were",
                 "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
                 "whys", "with", "would", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
                 "yourselves"]

    table = str.maketrans('', '', string.punctuation)
    print(type(table))
    sentences = []
    labels = []
    with open('E:\projects\AI_ML_Coders\ChapterFive/binary-emotion.csv', encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            labels.append(int(row[0]))
            sentence = row[1].lower()
            #print(sentence)
            sentence = sentence.replace(",", " , ")
            sentence = sentence.replace(".", " . ")
            sentence = sentence.replace("-", " - ")
            sentence = sentence.replace("/", " / ")
            soup = BeautifulSoup(sentence)
            sentence = soup.get_text()
            words = sentence.split()
            filtered_sentence = ""
            for word in words:
                word = word.translate(table)
                if word not in stopwords:
                    filtered_sentence = filtered_sentence + word + " "
            sentences.append(filtered_sentence)

    print(len(labels))
    #print(labels)
    print(len(sentences))
    #print(sentences)
    """ training & testing dataset """
    training_size = 28000
    training_sentences = sentences[0:training_size]
    testing_sentences = sentences[training_size:]
    training_labels = labels[0:training_size]
    testing_labels = labels[training_size:]

    """ encoding sentence by tokenizer"""

    vocab_size = 20000
    embedding_dim = 32
    max_length = 10
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)

    word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    print(training_sequences[0])
    print(training_padded[0])
    print(type(word_index))
    print(word_index)