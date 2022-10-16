import os

import numpy as np
import pandas as pd
import re
import string

from scipy.sparse import csr_matrix
from scipy import sparse

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from emonet.dataset import Dataset
from emonet.m_abstract import Model
from emonet.utils import tokenize_tweet

import re

class tfidfGRU(Model):
    def __init__(self,
                 name,
                 vocab_limit = 2000,
                 stem = False,
                 lower = True,
                 max_seq_len = 200,
                 tokenizer = "nltk", **kwargs):
        super().__init__(name)
        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self. ohEncoder = None 
        self.prep_stem = stem
        self.prep_tokenizer = tokenizer
        self.prep_lower = lower
        self.prep_vocab_limit = vocab_limit
        self.prep_max_seq_len = max_seq_len

    def preprocess(self, ds: Dataset):
        #Process the current data...
        ##tokenize the data
        ##remove stop words
        ##stem words
        ##lowercase the text (might already be done)
        x = ds.x.apply(lambda t: tokenize_tweet(
            t, tokenizer = self.prep_tokenizer, remove_stop_words = True,
            stem = self.prep_stem, lower = self.prep_lower 
        ))

        for sentence in x:
            sentence= re.sub(r'\s+', ' ', sentence, flags=re.I)
            sentence = re.sub(r'^b\s+', '', sentence)
            sentence = re.sub(r'\W', ' ', sentence)
            sentence = sentence.lower()

        y = np.array(ds.y).reshape(-1,1)

        if self.ohEncoder is None:
            self.ohEncoder = OneHotEncoder().fit(y)

        y = self.ohEncoder.transform(y).toarray()

        #If the model has no tokenizer yet, instantiate one. 
        #Fit the tokenizer on the data.
        
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words = self.prep_vocab_limit)
            self.tokenizer.fit_on_texts(x)

        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", 
                                            ngram_range=(1,2), 
                                            analyzer='word', 
                                            min_df=5, 
                                            max_df=0.7, 
                                            use_idf=True,
                                            norm = None)
        x = self.vectorizer.fit_transform(x).toarray()
        
        return x, y

    #Get the vocab size as set in the parameters.
    def get_vocab_size(self):
        #return len(self.tokenizer.word_index) + 1
        return len(self.vectorizer.vocabulary_) + 1 

    def build(self, train: Dataset, valid: Dataset):

        #Passing entire dataset as a parameter but only returning the data not the labels
        x_train, y_train = self.preprocess(train)
        x_valid, y_valid = self.preprocess(valid)

        self.model = keras.Sequential()
        self.model.add(layers.Embedding(input_dim = self.get_vocab_size(), output_dim = 150))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.GRU(128))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(64, activation = "sigmoid"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(6, activation = "softmax"))

        self.model.compile(optimizer = "Adam", loss = "categorical_crossentropy",
                        metrics = ["categorical_accuracy", 
                                    tf.keras.metrics.AUC(name="auc")])

        self.model.fit(x_train, y_train,
                    batch_size = 128, epochs = 50,
                    validation_data = (x_valid, y_valid),
                    callbacks = [tf.keras.callbacks.EarlyStopping(
                        monitor = "val_loss", patience = 15, restore_best_weights = True
                    )])
         
    #Function that's used when we want to evaluate the model on some test data.
    def eval(self, ds: Dataset, verbose = -1):
        #Preprocess the data we will generate labels for.
        x,y = self.preprocess(ds)
        #Evaluate the model with the test labels y.
        m = self.model.evaluate(x, y)
        m = {n:v for n,v in zip(self.model.metrics_names, m)}
        #Create a dictionary where the contents are the metrics from the test set.
        all_metrics = {
            ds.name+"_acc": m["categorical_accuracy"],
            ds.name+"_auc": m["auc"],
        }

        #Print out update to console if verbose is on.
        if verbose > 0:
            print("   evaluation", " ".join(['{}: {:.4f}'.format(k, v) for k, v in all_metrics.items()]))
        
        print(all_metrics)

        #Return the metrics.
        return all_metrics
