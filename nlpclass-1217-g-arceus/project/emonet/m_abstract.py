from abc import ABCMeta, abstractmethod

from emonet.dataset import Dataset
from emonet.utils import tokenize_tweet

class Model(metaclass=ABCMeta):
    def __init__(self, name):
        self.name = name
        
    @abstractmethod
    def preprocess(self, ds: Dataset):
        pass

    @abstractmethod
    def build(self, train: Dataset, valid: Dataset, embeddings_index: Dataset):
        pass

    @abstractmethod
    def eval(self, ds: Dataset, verbose=-1):
        pass

    def get_model_folder(self):
        print(os.path.join('models', self.__class__.__name__, self.name))
        return os.path.join(
            'models', self.__class__.__name__, self.name)