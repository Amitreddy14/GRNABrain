from sklearn.metrics import roc_curve, auc
from scipy import interp
import os
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
import numpy as np
import tensorflow as tf

from utils import *
import preprocessing
from tqdm import tqdm

def generate_candidate_grna(gan, rna, chromosome, start, end, a=400, view_length=23, num_seqs=4, plot=True):
    debug_print(['generating candidate grna for', chromosome, start, ':', end])
    
    chromosomes = [chromosome for _ in range(num_seqs)]
    starts = [start for _ in range(num_seqs)]
    ends = [end for _ in range(num_seqs)]
    
    if plot:
        fig, axis = plt.subplots(num_seqs, 1, figsize=(8, num_seqs * 2))
        axis[0].set_title('GRNA activity')


    X = np.zeros((num_seqs, view_length, 8))
    seq = None
