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
