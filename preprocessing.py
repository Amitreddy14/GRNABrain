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