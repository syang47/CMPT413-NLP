#Dataset handling.
import pandas as pd

#Does something
from typing import List

#Iterative and multi-threading tools.
from tqdm import tqdm
from functools import partial
from concurrent.futures import as_completed, ProcessPoolExecutor
from itertools import product

#Import models and dataset utility.
from emonet import Model
from emonet.dataset import Dataset, get_dataset

#Get model class names
model_classes = {
    cls.__name__: cls for cls in Model.__subclasses__()
}

#Runs the experiment based on passed class name.
def run(params, model_name, verbose = 1, result_with_param = True):
    model_cls = model_classes[model_name]
    
    #Instantiate train, validate, and test set.
    train, valid, test, embeddings_index = get_dataset()

    #Instantiate results dictionary. Keys are based on parameters.
    results = {}

    #Still need to figure out this naming convention shit
    valid.name = "emotions_val"
    # print(**params)
    model = model_cls(name = "emotions", **params)
    
    #Force model to be class type.
    model: Model
    model.build(train, valid, embeddings_index)
    #model.build(train, valid)

    #Get validation metrics
    val_metrics = model.eval(valid, verbose = verbose)
    # print(val_metrics)
    results.update(val_metrics)

    #Get test set metrics
    metrics = model.eval(test, verbose = verbose)
    results.update(metrics)
    
    #Return the results for the current model - parameter experiment.
    if result_with_param:
        results.update(params)
    return results

#Runs the Emonet GRU Model
def run_base_gru():
    #Parameter dictionary.
    d = {
        "vocab_limit": [80000],
        "stem": [True],
        "lower": [True],
        "max_seq_len":[30],
    }
    #Hyperparameter space including all combinations of features.
    hp_space = [dict(zip(d,v)) for v in product(*d.values())]
    #Compile all the results.
    all_results = []
    #Allow for possible threads to run multiple parameter set-ups at once.
    executor = ProcessPoolExecutor(max_workers = 1) #Defaults to number of CPUs available.
    print("Submitting {} jobs to be completed".format(len(hp_space)))
    #Compile all jobs - a job is a possible parameter set to train the model on.
    jobs = [executor.submit(run, p, "BaseGRU", -1) for p in hp_space]
    print("Getting the results")
    #Loop through jobs and save them to a respective csv file.
    for job in tqdm(as_completed(jobs), total = len(jobs)):
        all_results.append(job.result())
        print(pd.DataFrame(all_results))
        pd.DataFrame(all_results).to_csv("results/EMOnet.csv")

#Runs a vanilla GRU with passed parameters. 
def run_base_gru():
    #Parameter dictionary.
    d = {
        "vocab_limit": [100],
        "stem": [True],
        "lower": [True],
        "max_seq_len":[100, 200],
    }
    #Hyperparameter space including all combinations of features.
    hp_space = [dict(zip(d,v)) for v in product(*d.values())]
    #Compile all the results.
    all_results = []
    #Allow for possible threads to run multiple parameter set-ups at once.
    executor = ProcessPoolExecutor(max_workers = 1) #Defaults to number of CPUs available.
    print("Submitting {} jobs to be completed".format(len(hp_space)))
    #Compile all jobs - a job is a possible parameter set to train the model on.
    jobs = [executor.submit(run, p, "BaseGRU", -1) for p in hp_space]
    print("Getting the results")
    #Loop through jobs and save them to a respective csv file.
    for job in tqdm(as_completed(jobs), total = len(jobs)):
        all_results.append(job.result())
        print(pd.DataFrame(all_results))
        pd.DataFrame(all_results).to_csv("results/base_gru.csv")

#Runs GRU with glove embedding
def run_gru_glove():
    #Parameter dictionary.
    d = {
        "vocab_limit": [100],
        "stem": [True, False],
        "lower": [True],
    }
    #Hyperparameter space including all combinations of features.
    hp_space = [dict(zip(d,v)) for v in product(*d.values())]
    #Compile all the results.
    all_results = []
    #Allow for possible threads to run multiple parameter set-ups at once.
    executor = ProcessPoolExecutor(max_workers = 1) #Defaults to number of CPUs available.
    print("Submitting {} jobs to be completed".format(len(hp_space)))
    #Compile all jobs - a job is a possible parameter set to train the model on.
    jobs = [executor.submit(run, p, "GRUGloVe", -1) for p in hp_space]
    print("Getting the results")
    #Loop through jobs and save them to a respective csv file.
    for job in tqdm(as_completed(jobs), total = len(jobs)):
        all_results.append(job.result())
        print(pd.DataFrame(all_results))
        pd.DataFrame(all_results).to_csv("results/gru_glove.csv")

#Runs a vanilla LSTM with passed parameters. 
def run_base_lstm():
    #Parameter dictionary.
    d = {
        "vocab_limit": [100],
        "stem": [True],
        "lower": [True],
        "max_seq_len": [100,200],
    }
    #Hyperparameter space including all combinations of features.
    hp_space = [dict(zip(d,v)) for v in product(*d.values())]
    #Compile all the results.
    all_results = []
    #Allow for possible threads to run multiple parameter set-ups at once.
    executor = ProcessPoolExecutor(max_workers = 1) #Defaults to number of CPUs available.
    print("Submitting {} jobs to be completed".format(len(hp_space)))
    #Compile all jobs - a job is a possible parameter set to train the model on.
    jobs = [executor.submit(run, p, "BaseLSTM", -1) for p in hp_space]
    print("Getting the results")
    #Loop through jobs and save them to a respective csv file.
    for job in tqdm(as_completed(jobs), total = len(jobs)):
        all_results.append(job.result())
        print(pd.DataFrame(all_results))
        pd.DataFrame(all_results).to_csv("results/base_lstm.csv")

#Runs a bidirectional LSTM with passed parameters. 
def run_bi_lstm():
    #Parameter dictionary.
    d = {
        "vocab_limit": [1000],
        "stem": [True],
        "lower": [True],
        "max_seq_len": [100,200],
    }
    #Hyperparameter space including all combinations of features.
    hp_space = [dict(zip(d,v)) for v in product(*d.values())]
    #Compile all the results.
    all_results = []
    #Allow for possible threads to run multiple parameter set-ups at once.
    executor = ProcessPoolExecutor(max_workers = 1) #Defaults to number of CPUs available.
    print("Submitting {} jobs to be completed".format(len(hp_space)))
    #Compile all jobs - a job is a possible parameter set to train the model on.
    jobs = [executor.submit(run, p, "BiLSTM", -1) for p in hp_space]
    print("Getting the results")
    #Loop through jobs and save them to a respective csv file.
    for job in tqdm(as_completed(jobs), total = len(jobs)):
        all_results.append(job.result())
        print(pd.DataFrame(all_results))
        pd.DataFrame(all_results).to_csv("results/bi_lstm.csv")

#Runs an n-gram count vectorized GRU with passed parameters. 
def run_count_vector_gru():
    #Parameter dictionary.
    d = {
        "vocab_limit": [100],
        "stem": [True],
        "lower": [True],
        "max_seq_len": [100,200],
    }
    #Hyperparameter space including all combinations of features.
    hp_space = [dict(zip(d,v)) for v in product(*d.values())]
    #Compile all the results.
    all_results = []
    #Allow for possible threads to run multiple parameter set-ups at once.
    executor = ProcessPoolExecutor(max_workers = 1) #Defaults to number of CPUs available.
    print("Submitting {} jobs to be completed".format(len(hp_space)))
    #Compile all jobs - a job is a possible parameter set to train the model on.
    jobs = [executor.submit(run, p, "countVectorGRU", -1) for p in hp_space]
    print("Getting the results")
    #Loop through jobs and save them to a respective csv file.
    for job in tqdm(as_completed(jobs), total = len(jobs)):
        all_results.append(job.result())
        print(pd.DataFrame(all_results))
        pd.DataFrame(all_results).to_csv("results/countVectorGRU.csv")

#Runs a TFIDF Vectorized GRU with parameters passed
def run_tfidf_gru():
    #Parameter dictionary.
    d = {
        "vocab_limit": [100],
        "stem": [True],
        "lower": [True],
        "max_seq_len": [100,200],
    }
    #Hyperparameter space including all combinations of features.
    hp_space = [dict(zip(d,v)) for v in product(*d.values())]
    #Compile all the results.
    all_results = []
    #Allow for possible threads to run multiple parameter set-ups at once.
    executor = ProcessPoolExecutor(max_workers = 1) #Defaults to number of CPUs available.
    print("Submitting {} jobs to be completed".format(len(hp_space)))
    #Compile all jobs - a job is a possible parameter set to train the model on.
    jobs = [executor.submit(run, p, "tfidfGRU", -1) for p in hp_space]
    print("Getting the results")
    #Loop through jobs and save them to a respective csv file.
    for job in tqdm(as_completed(jobs), total = len(jobs)):
        print("THE JOB RESULT IS:")
        print(type(job.result()))
        all_results.append(job.result())
        print(pd.DataFrame(all_results))
        pd.DataFrame(all_results).to_csv("results/tfidf_gru.csv")

#Runs a Bidirectional LSTM with glove embedding with passed parameters. 
def run_lstm_bi_glove():
    #Parameter dictionary.
    d = {
        "vocab_limit": [100,500,1000],
        "stem": [True, False],
        "lower": [True],
    }
    #Hyperparameter space including all combinations of features.
    hp_space = [dict(zip(d,v)) for v in product(*d.values())]
    #Compile all the results.
    all_results = []
    #Allow for possible threads to run multiple parameter set-ups at once.
    executor = ProcessPoolExecutor(max_workers = 1) #Defaults to number of CPUs available.
    print("Submitting {} jobs to be completed".format(len(hp_space)))
    #Compile all jobs - a job is a possible parameter set to train the model on.
    jobs = [executor.submit(run, p, "LstmBiGloVe", -1) for p in hp_space]
    print("Getting the results")
    #Loop through jobs and save them to a respective csv file.
    for job in tqdm(as_completed(jobs), total = len(jobs)):
        all_results.append(job.result())
        print(pd.DataFrame(all_results))
        pd.DataFrame(all_results).to_csv("results/lstm_bi_glove.csv")

#Runs a Bidirectional GRU with passed parameters. 
def run_bi_gru():
    #Parameter dictionary.
    d = {
        "vocab_limit": [1000],
        "stem": [True],
        "lower": [True],
        "max_seq_len": [200],
    }
    #Hyperparameter space including all combinations of features.
    hp_space = [dict(zip(d,v)) for v in product(*d.values())]
    #Compile all the results.
    all_results = []
    #Allow for possible threads to run multiple parameter set-ups at once.
    executor = ProcessPoolExecutor(max_workers = 1) #Defaults to number of CPUs available.
    print("Submitting {} jobs to be completed".format(len(hp_space)))
    #Compile all jobs - a job is a possible parameter set to train the model on.
    jobs = [executor.submit(run, p, "BiLSTM", -1) for p in hp_space]
    print("Getting the results")
    #Loop through jobs and save them to a respective csv file.
    for job in tqdm(as_completed(jobs), total = len(jobs)):
        all_results.append(job.result())
        print(pd.DataFrame(all_results))
        pd.DataFrame(all_results).to_csv("results/bi_gru.csv")

#Main function where the model to be ran can be changed.
if __name__ == "__main__":
    
    #Pick the model you want to run to run it
    #Might have to change line #39 whether or not the model needs an embedding index
    run_base_gru() 
    #run_base_lstm() 
    #run_count_vector_gru() 
    #run_tfidf_gru() 
    #run_bi_lstm()
    #run_lstm_bi_glove()
    #run_bi_gru() #run
