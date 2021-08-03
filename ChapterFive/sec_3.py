""" Removing Stopwords and Cleaning Text """
from bs4 import BeautifulSoup
import string
# 1- The first is to strip out HTML tags.
def stripOutHTML():
    sentences = "<HTML><body>Hi Ahmed </body></HTML>"
    soup = BeautifulSoup(sentences, features="lxml")
    sentence = soup.get_text()
    print(sentence)


# 2- A common way to remove stopwords
def removeStopwords():
    sentences = []
    sentence = "Today is a sunny day"
    stopwords = ["a", "about", "above","yours", "yourself", "yourselves"]
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word +" "
    sentences.append(filtered_sentence)
    print(sentences)

# 3-  Stripping out punctuation

def removePunctuation():
    sentences = []
    table = str.maketrans('', '', string.punctuation)
    sentence = "Today is a sunny day $ # % ^ & * "
    stopwords = ["a", "about", "above", "yours", "yourself", "yourselves"]
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    sentences.append(filtered_sentence)
    print(sentences)



if __name__ == '__main__':
    stripOutHTML()
    removeStopwords()
    removePunctuation()