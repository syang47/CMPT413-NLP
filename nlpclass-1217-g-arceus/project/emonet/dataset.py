import os
import numpy as np
from typing import NamedTuple, List, Tuple
import pandas as pd

emotion_labels = {
    "sadness": 0,
    "fear": 1,
    "love": 2,
    "joy": 3,
    "anger": 4,
    "surprise": 5
}

class Dataset:
    def __init__(self, name, x, y):
        self.name = name
        self.name : str
        self.x = x
        self.x: pd.DataFrame
        self.y = y
        self.y: pd.DataFrame

def get_dataset():
    def prep_dataset(df, name):
        return Dataset(name, df.text, df.emotion.apply(lambda e: emotion_labels[e]))
    
    def prep_gloveEmb():
        embeddings_index = {}
        with open("data/glove.6B.100d.txt", encoding="utf-8") as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs
        return embeddings_index

    train = pd.read_csv("data/emotion_dataset_train.txt", header = None, sep = ";", index_col = None)
    train.columns = ["text","emotion"]
    val = pd.read_csv("data/emotion_dataset_val.txt", header = None, sep = ";", index_col = None)
    val.columns = ["text","emotion"]
    test = pd.read_csv("data/emotion_dataset_test.txt", header = None, sep = ";", index_col = None)
    test.columns = ["text","emotion"]
    embeddings_index = prep_gloveEmb()


    return [prep_dataset(train, "emotions_train"),
            prep_dataset(val, "emotions_validation"),
            prep_dataset(test, "emotions_test"),
            embeddings_index]