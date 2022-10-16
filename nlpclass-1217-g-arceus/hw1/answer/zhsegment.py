import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10

class Entry:
    def __init__(self, word, start_position, log_probability, back_pointer):
        self.word = word
        self.start_position = start_position
        self.log_probability = log_probability
        self.back_pointer = back_pointer
    def __eq__ (self, entry):
        if( self.word == entry.word) and (self.start_position == entry.start_position) and (self.log_probability == entry.log_probability):
            return True
        if not isinstance(entry, type(self)):
            return False
        else:
            return False
    def __lt__(self, entry):   # for < instances in heap
        return self.log_probability > entry.log_probability
    def __gt__(self, entry):   # for > instances in heap
        return self.log_probability <= entry.log_probability

class Segment:

    def __init__(self, bigram_Pw, unigram_Pw):
        self.bigram_Pw = bigram_Pw
        self.unigram_Pw = unigram_Pw

    def segment(self, text):
        ## Initialize the heap, seg, chart array ##
        heap, segmentation = [], []
        chart = {}
        # for each word that matches input at pos 0, insert entry into heap
        
        for i in range(len(text)):
            word = text[0:i+1]
            chart[i] = Entry(None, None, None, None)
            if word in self.unigram_Pw or len(word) < 6:
#                 print("forloop1")
                heapq.heappush(heap, Entry(word, 0, log10(self.unigram_Pw(word)), None))    ## heapq.heappush pushes the entry item to the heap queue
        
        ## Iteratively fill in chart[i] for all i ##
        while len(heap) != 0:
#             print("whileloop1")
            entry = heapq.heappop(heap)     # top entry in the heap
            endindex = entry.start_position + len(entry.word) - 1
            if chart[endindex].back_pointer is not None:        
#                 preventry = chart[endindex]
                preventry = chart[endindex].back_pointer
                if entry.log_probability > preventry.log_probability:
                    chart[endindex] = entry
                else:
                    continue    ## we have already found a good segmentation until endindex ##
            else:
                chart[endindex] = entry
                
            for j in range(endindex+1, len(text)):
#                 print("forloop2")
                newword = text[endindex+1 : j+1]
                
#                 print(entry.word+newword)
#                 print(entry.word+" "+newword)
                if newword in self.unigram_Pw or len(newword) < 6:
                    word_pairs = entry.word + newword
                    if word_pairs in self.bigram_Pw and entry.word in self.unigram_Pw:
                        newentry = Entry(newword, endindex+1, entry.log_probability+log10(self.bigram_Pw(word_pairs)/self.unigram_Pw(newword)), entry)
                    else:
#                         newentry = Entry(newword, endindex+1, entry.log_probability+log10(self.unigram_Pw(newword)), entry)
                        newentry = Entry(newword, endindex+1, entry.log_probability+log10(self.unigram_Pw(newword)), entry)
                    
                    if newentry not in heap:
                        heapq.heappush(heap, newentry)
        
        ## Get the best segmentation ##
        finalindex = len(text)
        finalentry = chart[finalindex-1]
        while finalentry is not None:
            segmentation.insert(0, finalentry.word)
            finalentry = finalentry.back_pointer

        return segmentation

    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)
    

#### Support functions (p. 224)

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.missingfn = missingfn or (lambda k, N: 1. / (N * 1100 ** len(k))) 
    def __call__(self, key): 
        if key in self: return self[key]/self.N  
        else: return self.missingfn(key, self.N)

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    unigram_Pw = Pdist(data=datafile(opts.counts1w))
    bigram_Pw = Pdist(data=datafile(opts.counts2w))
    segmenter = Segment(bigram_Pw, unigram_Pw)
    with open(opts.input) as f:
        for line in f:
            print(" ".join(segmenter.segment(line.strip())))
