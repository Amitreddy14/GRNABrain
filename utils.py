import datetime
import numpy as np

def debug_print(statements=[], end='\n'):
    ct = datetime.datetime.now()
    print('[', str(ct)[:19], '] ', sep='', end='')
    for statement in statements:
        print(statement, end=' ')
    print(end=end)

def ohe_base(base):
    base = base.lower()
    ohe = np.zeros((1, 4))
    if base == 'a': ohe[0, 0] = 1
    if base == 'g': ohe[0, 1] = 1
    if base == 'c': ohe[0, 2] = 1
    if base == 't': ohe[0, 3] = 1
    return ohe    

def ohe_bases(bases_lists):
    debug_print(['one-hot encoding bases'])
    ohe = np.zeros((len(bases_lists), len(bases_lists[0]), 4))
    for i, bases_list in enumerate(bases_lists):
        for j, base in enumerate(bases_list):
            if j >= len(bases_lists[0]): continue
            ohe[i, j] = ohe_base(base)
    return ohe