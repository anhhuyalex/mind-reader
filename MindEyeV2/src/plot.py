import pickle 
import glob
import numpy as np
import torch
import io
from collections import defaultdict 
import seaborn as sns
import matplotlib.pyplot as plt

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def plot(results_folder, exp_name, 
                    num_runs_analyze,
                    
                    palette = None
                   ):
    fs = glob.glob(f"{results_folder}/{exp_name}.pkl")
    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    avg_test_fwd_pct_correct = defaultdict(list)
    avg_test_bwd_pct_correct = defaultdict(list)
    def plot_histogram(x, data, hue, xlabel, ylabel):
        sns.set_theme(style="whitegrid")
        ax = sns.histplot( x=x, data=data, hue = hue, palette=palette)
        ax.set_title(exp_name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        # legend upper right 
        sns.move_legend(ax, "center right")

        plt.show()
    for f in np.random.choice(fs, min(num_runs_analyze, len(fs)), replace=False) :
        
        with open(f, 'rb') as handle:
            try:
                record = CPU_Unpickler(handle).load()
                
            except Exception as e: 
                print(e)
                print("problem !")
                continue
            # sort by key, return a list of tuples 
            t = sorted(record.metrics.train_losses.items() , key=lambda x: x[0])
            train_losses["train_losses"].append ( t[-1][1])
            train_losses["num_sessions"].append ( record.args.num_sessions) 
            t = sorted(record.metrics.test_losses.items() , key=lambda x: x[0])
            test_losses["test_losses"].append ( t[-1][1])
            test_losses["num_sessions"].append ( record.args.num_sessions)
            t = sorted(record.metrics.avg_test_fwd_pct_correct.items() , key=lambda x: x[0])
            avg_test_fwd_pct_correct["avg_test_fwd_pct_correct"].append ( t[-1][1])
            avg_test_fwd_pct_correct["num_sessions"].append ( record.args.num_sessions)
            t = sorted(record.metrics.avg_test_bwd_pct_correct.items() , key=lambda x: x[0])
            avg_test_bwd_pct_correct["avg_test_bwd_pct_correct"].append ( t[-1][1])
            avg_test_bwd_pct_correct["num_sessions"].append ( record.args.num_sessions)
    print (record.metrics.keys())
            
    # plot histogram of train losses 
    plot_histogram(x = "train_losses", data = train_losses, hue = "num_sessions", xlabel = "train_losses", ylabel = "frequency")
    # plot histogram of test losses
    plot_histogram(x = "test_losses", data = test_losses, hue = "num_sessions", xlabel = "test_losses", ylabel = "frequency")
    # plot histogram of avg_test_fwd_pct_correct
    plot_histogram(x = "avg_test_fwd_pct_correct", data = avg_test_fwd_pct_correct, hue = "num_sessions", xlabel = "avg_test_fwd_pct_correct", ylabel = "frequency")
    # plot histogram of avg_test_bwd_pct_correct
    plot_histogram(x = "avg_test_bwd_pct_correct", data = avg_test_bwd_pct_correct, hue = "num_sessions", xlabel = "avg_test_bwd_pct_correct", ylabel = "frequency")
    
def plot_learning_curve(results_folder, exp_name, 
                    num_runs_analyze,
                    
                    palette = None
                   ):
    fs = glob.glob(f"{results_folder}/{exp_name}.pkl")
    train_losses = defaultdict(list)
    test_losses = defaultdict(list)
    avg_test_fwd_pct_correct = defaultdict(list)
    avg_test_bwd_pct_correct = defaultdict(list)
    def plot_curve(x, y, data, xlabel, ylabel):
        sns.set_theme(style="whitegrid")
        ax = sns.lineplot( x=x, y=y, data=data, palette=palette)
        ax.set_title(exp_name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        # legend upper right 
        # sns.move_legend(ax, "center right")

        plt.show()
    for f in np.random.choice(fs, min(num_runs_analyze, len(fs)), replace=False) :
        
        with open(f, 'rb') as handle:
            try:
                record = CPU_Unpickler(handle).load()
                
            except Exception as e: 
                print(e)
                print("problem !")
                continue
            # sort by key, return a list of tuples 
            t = sorted(record.metrics.train_losses.items() , key=lambda x: x[0])
            train_losses["train_losses"].append ( t[-1][1])
            train_losses["num_sessions"].append ( record.args.num_sessions) 
            t = sorted(record.metrics.test_losses.items() , key=lambda x: x[0])
            test_losses["test_losses"].append ( t[-1][1])
            test_losses["num_sessions"].append ( record.args.num_sessions)
            t = sorted(record.metrics.avg_test_fwd_pct_correct.items() , key=lambda x: x[0])
            avg_test_fwd_pct_correct["avg_test_fwd_pct_correct"].append ( t[-1][1])
            avg_test_fwd_pct_correct["num_sessions"].append ( record.args.num_sessions)
            t = sorted(record.metrics.avg_test_bwd_pct_correct.items() , key=lambda x: x[0])
            avg_test_bwd_pct_correct["avg_test_bwd_pct_correct"].append ( t[-1][1])
            avg_test_bwd_pct_correct["num_sessions"].append ( record.args.num_sessions)
    print (record.metrics.keys())
            
    # plot curve of train losses 
    plot_curve( y = "train_losses", data = train_losses, 
            x = "num_sessions", xlabel = "num_sessions", ylabel = "train_losses")
    # plot curve of test losses
    plot_curve(y= "test_losses", 
                data = test_losses, x = "num_sessions", 
                xlabel = "num_sessions", ylabel = "test_losses")
    # plot curve of avg_test_fwd_pct_correct
    plot_curve(y= "avg_test_fwd_pct_correct", data = avg_test_fwd_pct_correct, 
            x = "num_sessions", ylabel = "avg_test_fwd_pct_correct",
             xlabel = "num_sessions")
    # plot curve of avg_test_bwd_pct_correct
    plot_curve(y= "avg_test_bwd_pct_correct", 
            data = avg_test_bwd_pct_correct, 
            x = "num_sessions", 
            ylabel = "avg_test_bwd_pct_correct", 
            xlabel = "num_sessions")
    