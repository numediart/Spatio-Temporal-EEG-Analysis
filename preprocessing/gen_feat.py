import numpy as np
import gc
import os
import glob

from tqdm import tqdm
from scipy.stats import skew, kurtosis

def comp_entropy(sig):
    return 0.5*np.log(2*np.pi*np.exp(1)*np.var(sig))

path = 'path_to_numpy_eeg'

n_sample = 27192#for cao
feat_array = np.zeros((n_sample, 5, 30, 20))

par_tot = []
dim = 0
i = 0
for f in tqdm(glob.glob(path+'bands/eeg/*.npy')):
    eeg = np.load(f)

    for sig in eeg:
        for c in range(5):            
            var = np.var(sig[c], axis=1)
            for s in range(30):
                for k in range(20):
                    feat_array[i, c, s, k] = comp_entropy(sig[c,s][k*100:(k+1)*100])
        i += 1 

np.save('feat_path_to_save/b_de', feat_array)

