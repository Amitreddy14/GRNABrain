import csv
from tqdm import tqdm
import random
import copy
import numpy as np
import pandas as pd
import os

from Bio import SeqIO
import pyBigWig

from utils import *

REFERENCE_GENOME_PATH = 'data/GCF_000001405.26_GRCh38_genomic.fna' # https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.26/
GRNA_PATH = 'data/hg_guide_info.csv'  # https://www.ncbi.nlm.nih.gov/genome/guide/human/

GENOME_SEQUENCES = {}
def read_genome(path=REFERENCE_GENOME_PATH):
    debug_print(['loading genomic data from', path])
    for record in SeqIO.parse(path, 'fasta'):
        GENOME_SEQUENCES[record.id.split('.')[0]] = record.seq

def fetch_genomic_sequence(chromosome, start, end, a=0):
    chromosome_id = f'NC_0000{chromosome}'
    genome_sequence = GENOME_SEQUENCES.get(chromosome_id)
    if genome_sequence:
        return str(genome_sequence[start - a:end + a + 1]) # start - 1
    else:
        raise ValueError(f'chromosome {chromosome_id} not found in the genome file.') 

def fetch_epigenomic_signals(chromosome, start, end, a=0):
    signals = np.zeros((end - start + 1 + 2 * a, 4))
    
    chromosome = 'chr' + str(int(chromosome))
    
    h3k4me_file = pyBigWig.open(H3K4ME3_PATH)
    rrbs_file = pyBigWig.open(RRBS_PATH)
    dnase_file = pyBigWig.open(DNASE_PATH)
    ctcf_file = pyBigWig.open(CTCF_PATH)
    
    # print(chromosome, start - a, end  + a + 1)

    def set_signal(index, entries):
        if not entries:
            return
        for e in entries:
            read_start = e[0]
            read_end = e[1]
            string_vals = e[2].split('\t')
            val = 0 
            if string_vals[0].isnumeric():
                if float(string_vals[0]) > 0: val = 1
                else: val = 0 
                # val = float(string_vals[0]) / 1000
            else:
                if float(string_vals[1]) > 0: val = 1
                else: val = 0 
                # val = float(string_vals[1]) / 1000
            signals[read_start-start:read_end-start, index] = val  

    h3k4me_vals = np.array(h3k4me_file.values(chromosome, start - a, end + a + 1))
    signals[:, 2] = h3k4me_vals if not h3k4me_vals.any() == None else signals[:, 2]
    set_signal(1, rrbs_file.entries(chromosome, start - a, end + a + 1))
    dnase_vals = np.array(dnase_file.values(chromosome, start - a, end + a + 1))
    signals[:, 2] = dnase_vals if not dnase_vals.any() == None else signals[:, 2]
    ctcf_vals = np.array(ctcf_file.values(chromosome, start - a, end + a + 1))
    signals[:, 2] = ctcf_vals if not ctcf_vals.any() == None else signals[:, 2]    

    h3k4me_file.close()
    rrbs_file.close()
    dnase_file.close()
    ctcf_file.close()
    
    return signals       

def filter_bases_lists(bases1, bases2):
    debug_print(['filtering base sequences'])
    filter_bases1 = []
    filter_bases2 = []
    for base1, base2 in zip(bases1, bases2):
        set1 = set(list(base1.lower()))
        set2 = set(list(base2.lower()))
        if 'n' not in set1 and 'n' not in set2:
            filter_bases1.append(base1)
            filter_bases2.append(base2)

    return filter_bases1, filter_bases2  