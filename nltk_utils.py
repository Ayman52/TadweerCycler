import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenzied_sentence,all_words):
    tokenzied_sentence = [stem(w) for w in tokenzied_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for index, w in enumerate(all_words):
        if w in tokenzied_sentence:
            bag[index] = 1.0
    return bag



