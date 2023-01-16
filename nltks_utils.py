import nltk
import numpy as np

nltk.download("punkt")
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer() # create stemmer


def tokenize(sentence): # create method function inside a class
    return nltk.word_tokenize(sentence)
# implement in one line you have to download package punkt otherwise error
# the punkt package includes a pre-trained tokenizer
# example string g = "How long does shipping take?"
# g = tokenize(g) print(g) = ['How', 'long', 'does', 'shipping', 'take', '?']

def stem(word): # method for stemming
    return stemmer.stem(word.lower())
# also implement in one line but have to import a stemmer from nltk
# convert word to stemmer and lowercase
# all pre-processing techniques utility functions
# words = ["organize", "organizes", "organizing"]
# stemmed_words = [stem(w) for w in words print(stemmed_words) = 'organ', 'organ', 'organ'



def bag_of_words(tokenized_sentence, all_words): # method bag of words to cover whole nltk package
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thanks", "cool"]
    bog = [ 0, 1, 0, 1, 0, 0, 0]
    """
    # implement bag of words function
    # tokenized sentence and all words collected based ont he patterns we looked up from json file
    # we look at each word in sentence and if it is available in words array then we return 1 otherwise 0

    tokenized_sentence = [stem(w) for w in tokenized_sentence]
# gets tokenized sentence in the training pipeline we will apply the stemming for all words array
    # same for tokenized sentence call  stemmer for each word in tokenized sentence

    bag = np.zeros(len(all_words), dtype=np.float32)
    # create bag and initialize it with zero for each word
    # we have all words and create and array with same size but only with zeroes via numpy
    # import numpy as np
    #  np.zeroes with size of length of the words allwords and define datatype numpy float 32 important
    for index, w in enumerate(all_words):
        # loop through allwords gives us index and current word
        if w in tokenized_sentence:
            # check if word is in our tokenized sentence
            bag[index] = 1.0 # if yes then 1
    return bag

# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thanks", "cool"]
# bog bag of words = bag_of_words the bag of words function sentence first and word second
# bog = bag_of_words(sentence, words) = will get same array = bog [0, 1, 0, 1, 0, 0, 0]
