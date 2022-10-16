import os

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from emonet.dataset import Dataset
from emonet.m_abstract import Model
from emonet.utils import tokenize_tweet

import re

class GRUGloVe(Model):
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
            t, tokenizer = self.prep_tokenizer, remove_stop_words = False,
            stem = self.prep_stem, lower = self.prep_lower 
        ))
        y = np.array(ds.y).reshape(-1,1)

        if self.ohEncoder is None:
            self.ohEncoder = OneHotEncoder().fit(y)

        y = self.ohEncoder.transform(y).toarray()

        #If the model has no tokenizer yet, instantiate one. 
        #Fit the tokenizer on the data.
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words = self.prep_vocab_limit)
            self.tokenizer.fit_on_texts(x)

        #Transform the text to numeric sequences.
        x = self.tokenizer.texts_to_sequences(x)
        #Return the padded representations.
        return pad_sequences(x, maxlen = self.prep_max_seq_len, padding = "post"), y

    #Get the glove embedding matrix
    def get_glove_embedding_matrix(self, embeddings_index):  
        num_tokens = self.get_vocab_size()
        embedding_dim = 100

        # Assign word vectors to vocabulary
        embedding_matrix = np.zeros((num_tokens, embedding_dim))
        for word, i in self.tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    #Get the vocab size as set in the parameters.
    def get_vocab_size(self):
        return len(self.tokenizer.word_index) + 1

    def build(self, train: Dataset, valid: Dataset, embeddings_index: Dataset):
        #Passing entire dataset as a parameter but only returning the data not the labels
        x_train, y_train = self.preprocess(train)
        x_valid, y_valid = self.preprocess(valid)
        embedding_matrix = self.get_glove_embedding_matrix(embeddings_index)

        self.model = keras.Sequential()
        self.model.add(layers.Embedding(input_dim = self.get_vocab_size(), output_dim = 100,
                                        input_length = self.prep_max_seq_len,
                                        trainable=False,
                                        weights=[embedding_matrix]))
        self.model.add(layers.GRU(128, return_sequences=False))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(64, activation = "sigmoid"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(6, activation = "softmax"))

        self.model.compile(optimizer = "Adam", loss = "categorical_crossentropy",
                        metrics = ["categorical_accuracy", 
                                    tf.keras.metrics.AUC(name="auc"), 
                                    # tf.keras.metrics.Precision(name="precision"), 
                                    # tf.keras.metrics.Recall(name="recall")
                                    ])
        
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
            #ds.name+"_pre": m["precision"],
            #ds.name+"_rec": m["recall"]
        }

        #Print out update to console if verbose is on.
        if verbose > 0:
            print("   evaluation", " ".join(['{}: {:.4f}'.format(k, v) for k, v in all_metrics.items()]))
        
        print(all_metrics)

        #Return the metrics.
        return all_metrics
