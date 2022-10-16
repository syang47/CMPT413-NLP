from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize.casual import TweetTokenizer
from typing import List

ps = PorterStemmer()
tkn_nltk = TweetTokenizer(
    strip_handles = True, reduce_len = True
)

stop_words = set(stopwords.words("english"))

tokenizers = {
    "nltk": lambda t: tkn_nltk.tokenize(t)
}

def tokenize_tweet(tweet, tokenizer = "nltk", remove_stop_words = False,
                   stem = False, lower = True, as_str = True):
    tweet = str(tweet)
    if lower:
        tweet = tweet.lower()
    if remove_stop_words:
        tweet = [w for w in tweet if w not in stop_words]
    if stem:
        tweet = [ps.stem(w) for w in tweet]
    if as_str:
        return " ".join(tweet)
    else:
        return tweet
    
