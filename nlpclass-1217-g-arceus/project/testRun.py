import re
import string
import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import one_hot

import os
import matplotlib.pyplot as plt
import seaborn as sns

import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#GRNN Embedding Imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

testData = ""
trainData = ""
validationData = ""
vocabularySize = ""
sentenceLength = ""

def main():
    #with open('/content/drive/MyDrive/Colab Notebooks/EmoNet Data/test.txt')
    global testData 
    global trainData 
    global validationData 
    global vocabularySize
    global sentenceLength

    testData = pd.read_csv("./data/emotion_dataset_test.txt", header=None, sep=";", 
                            names=["Comment","Emotion"], encoding="utf-8")
    trainData = pd.read_csv("./data/emotion_dataset_train.txt", header=None, sep=";",
                            names=["Comment","Emotion"], encoding="utf-8")
    validationData = pd.read_csv("./data/emotion_dataset_val.txt", header=None, sep=";",
                                names=["Comment","Emotion"], encoding="utf-8")

    #Verifying that the data has been successfully imported
    print("Train : ", trainData.shape)
    print("Test : ", testData.shape)
    print("Validation : ", validationData.shape)

    #Scikitlearn's label encoder
    lb = LabelEncoder()
    trainData["Emotion"] = lb.fit_transform(trainData["Emotion"])
    testData["Emotion"] = lb.fit_transform(testData["Emotion"])
    validationData["Emotion"] = lb.fit_transform(validationData["Emotion"])

    vocabularySize = 15000
    sentenceLength = 150

    nltk.download('stopwords')
    
    runLSTMModel()


def textPrepareOHCEncoding(data, column):
    print(data.shape)
    stemmer = PorterStemmer()
    corpus = []
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    for text in data[column]:
      #Taking the text data that begin with characters and processing them for encoding
      text = re.sub("[^a-zA-Z]", " ", text)
      text = text.lower()
      text = text.split()

      #Reducing the inflection of words to their roots
      #text = [stemmer.stem(word) for word in text if word not in stopwords]
      for word in text:
          if word not in stopwords:
            stemmer.stem(word)

      text = " ".join(text)

      corpus.append(text)
    
    #Encoding begins here (OHC)
    oneHotWord = [one_hot(input_text=word, n=vocabularySize) for word in corpus]
    embeddedDoc = pad_sequences(sequences=oneHotWord, maxlen=sentenceLength, padding="pre")

    print(data.shape)
    return embeddedDoc

def runLSTMModel():
    x_train= textPrepareOHCEncoding(trainData, "Comment")
    x_validate=textPrepareOHCEncoding(validationData, "Comment")
    x_test=textPrepareOHCEncoding(testData, "Comment")

    y_train=trainData["Emotion"]
    y_validate=validationData["Emotion"]
    y_test=testData["Emotion"]

    enc = OneHotEncoder()
    y_train = np.array(y_train)
    y_train = enc.fit_transform(y_train.reshape(-1,1)).toarray()

    y_test = np.array(y_test)
    y_validate = np.array(y_validate)

    y_test = enc.fit_transform(y_test.reshape(-1,1)).toarray()
    y_validate = enc.fit_transform(y_validate.reshape(-1,1)).toarray()
    model = Sequential()
    model.add(Embedding(input_dim=vocabularySize, output_dim=150, input_length=sentenceLength))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="sigmoid"))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation="softmax"))

    model.compile(optimizer="Adam", loss = "categorical_crossentropy", metrics=["accuracy"])

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
    mc = ModelCheckpoint('./model.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)

    hist = model.fit(x_train, y_train, epochs = 25, batch_size = 64, validation_data=(x_validate, y_validate),verbose = 1, callbacks= [es, mc])

if __name__ == "__main__":
    main()