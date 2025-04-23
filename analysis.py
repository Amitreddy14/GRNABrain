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

    for n in range(num_seqs):
        seq = preprocessing.fetch_genomic_sequence(chromosome, start, end).lower()
        ohe_seq = np.concatenate([preprocessing.ohe_base(base) for base in seq], axis=0)
        epigenomic_signals = preprocessing.fetch_epigenomic_signals(chromosome, start, end)
        epigenomic_seq = np.concatenate([ohe_seq, epigenomic_signals], axis=1)
        
        X[n] = epigenomic_seq
        
    candidate_grna = gan.generate(X)
    filtered_candidate_grna = []
    candidate_grna_set = set()
    for i in range(candidate_grna.shape[0]):
        grna = preprocessing.str_bases(candidate_grna[i])
        if grna in candidate_grna_set:
            continue
        else:
            filtered_candidate_grna.append(candidate_grna[i])
            candidate_grna_set.add(grna)
    print(candidate_grna_set)
    filtered_candidate_grna = np.array(filtered_candidate_grna)

    debug_print(['generating candidate grna for', seq, ':'])
    debug_print(['      [correct grna]', preprocessing.str_bases(rna)])
    for grna in candidate_grna:
        debug_print(['     ', preprocessing.str_bases(grna)])
    
    activity_test(gan, filtered_candidate_grna, chromosomes, starts, ends, a, view_length, plot, filtered_candidate_grna.shape[0], True)

def activity_test(gan, rnas, chromosomes, starts, ends, a=400, view_length=23, plot=True, num_seqs=None, test_rna=False, return_sep=False, experimental_efficacies=None):
    if not num_seqs: num_seqs = len(rnas)
    
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    if plot:
        fig, axis = plt.subplots(num_seqs, 1, figsize=(8, num_seqs * 2))
        axis[0].set_title('GRNA activity')
    
    skipped = 0
    skip = []
    
    all_activity_scores = np.zeros(2 * a - 1)
    sep_activity_scores = []
    X_gen = np.zeros((len(rnas), view_length, 8))
    X = np.zeros((len(rnas), view_length + 2 * a, 8))

    for n in range(num_seqs):
        try:
            seq = preprocessing.fetch_genomic_sequence(chromosomes[n], starts[n] - a, ends[n] + a).lower()
            ohe_seq = np.concatenate([preprocessing.ohe_base(base) for base in seq], axis=0)
            epigenomic_signals = preprocessing.fetch_epigenomic_signals(chromosomes[n], starts[n] - a, ends[n] + a)
            epigenomic_seq = np.concatenate([ohe_seq, epigenomic_signals], axis=1)
            if np.isnan(epigenomic_seq).any():
                skip.append(n)
                continue
                
            X[n] = epigenomic_seq
            X_gen[n] = epigenomic_seq[a:a + view_length]
        except Exception as e:
            skip.append(n)
            continue
            
    if test_rna:
        real_Yi = np.array(rnas)
    else:

        pred_Yi = gan.generator(X_gen)
        pred = np.argmax(pred_Yi, axis=2)
        real = np.argmax(rnas, axis=2)
        real_Yi = gan.get_real_Yi(pred_Yi, pred, real)
        
    axis_index = 0

    for n, (rna, chromosome, start, end) in enumerate(zip(rnas, chromosomes, starts, ends)):
        if n >= num_seqs + skipped: continue
        if n in skip:
            skipped += 1
            continue
        preprocessing.debug_print(['running activity test on', preprocessing.str_bases(rna)])
        bind_site = start
        start -= a
        end += a
        
        activity_scores = []
        step = 1
        for i in range(0, end - start - view_length, step):
            if gan.discriminator.name == 'conv_discriminator' or gan.discriminator.name == 'critic_transformer_1':
                activity_score = gan.discriminator([
                    np.expand_dims(X[n][i:i+view_length], axis=0), 
                    np.expand_dims(real_Yi[n], axis=0)
                ])
                for _ in range(step):
                    activity_scores.append(activity_score[0][0].numpy())
            else:
                pass # finish
        activity_scores = np.array(activity_scores)
        moving_averages = activity_scores[23:]# moving_average(activity_scores, 100)
        all_activity_scores += np.array(activity_scores)
        sep_activity_scores.append(activity_scores)
        if plot:
            x = np.arange(start + view_length, end - view_length + 4)[:len(moving_averages)]
            axis[axis_index].plot(x, moving_averages, label=preprocessing.str_bases(rna))
            if experimental_efficacies is None:
                axis[axis_index].axvline(x=bind_site, color='orange', linestyle='dotted', label='bind site')
            else:
                axis[axis_index].scatter(x=bind_site, y=experimental_efficacies[n], color='red', label='experimental efficacy')
            axis[axis_index].legend()
        
        axis_index += 1
    
    if plot:
        plt.xlabel('genomic position')
        plt.ylabel('predicted activity')
        plt.show()
    
    if return_sep:
        return sep_activity_scores
    return all_activity_scores / axis_index