import glob 
import mne 
import os
import gc
import numpy as np 
from tqdm import tqdm

import matplotlib.pyplot as plt 

mne.set_log_level(verbose='CRITICAL')

path_save = 'bands'

path_dir = 'path_cao_preprocessed'

def comp_react_time(a):
    react_time = []
    instant = []
    for e in range(len(a)):
        if a[e,-1]==3:
            if a[e-1, -1] == 2 or a[e-1, -1] ==1:
                dt = a[e, 0] - a[e-1, 0]
                instant.append(a[e-1, 0])
                react_time.append(dt)
            else:
                print("error")
    return np.asarray(react_time), np.asarray(instant)


def gen_array(raw, instant, event_id = {'250': 1}, t_min=-1, t_max=3):
    picks =  mne.pick_types( raw.info, meg=True, eeg=True, stim=False, eog=True,
        include=[], exclude='bads')
    epoch = mne.Epochs(raw, instant, event_id, tmin=t_min, tmax=t_max, picks=picks,
        baseline=(None, 0), reject=None, preload=True)
    return epoch.get_data()

def time_array(instant):
    event = np.vstack((instant, np.zeros(instant.shape[0]), np.ones(instant.shape[0]))).astype(int)
    return np.transpose(event)

me = []
med = []
tot_dim = 0.0

deviation = np.load('deviation.npy', allow_pickle=True).all()

for f in tqdm(os.listdir(path_dir)):
    path = os.path.join(path_dir, f, '*.set')
    file = glob.glob(path)[0]
    f_name = file.split('/')[-1].split('.')[0]

    f_dev = deviation[f.split('_')[0]]

    d = mne.io.read_raw_eeglab(file).load_data()
    t = mne.io.read_raw_eeglab(file).load_data()
    a = mne.io.read_raw_eeglab(file).load_data()
    b = mne.io.read_raw_eeglab(file).load_data()
    g = mne.io.read_raw_eeglab(file).load_data()

    if f_dev>-0.5:
        d.filter(1+f_dev, 4+f_dev, fir_window='hann',filter_length='8s', n_jobs=10)
    else:
        d.filter(0.5, 4+f_dev, fir_window='hann',filter_length='8s', n_jobs=10)
    t.filter(4+f_dev, 8+f_dev, fir_window='hann',filter_length='8s', n_jobs=10)
    a.filter(8+f_dev, 14+f_dev, fir_window='hann',filter_length='8s', n_jobs=10)
    b.filter(14+f_dev, 31+f_dev, fir_window='hann',filter_length='8s', n_jobs=10)
    g.filter(31+f_dev, 50+f_dev,  fir_window='hann',filter_length='8s', n_jobs=10)

    react_time, t_id = comp_react_time(mne.events_from_annotations(d)[0])

    numpy_d = gen_array(d, time_array(t_id))
    numpy_t = gen_array(t, time_array(t_id))
    numpy_a = gen_array(a, time_array(t_id))
    numpy_b = gen_array(b, time_array(t_id))
    numpy_g = gen_array(g, time_array(t_id))

    '''
    print(numpy_d.shape)
    plt.plot(numpy_d[0,0], label='delta')
    plt.plot(numpy_t[0,0], label='theta')
    plt.plot(numpy_a[0,0], label='alpha')
    plt.plot(numpy_b[0,0], label='beta')
    plt.plot(numpy_g[0,0], label='gamma')
    plt.legend()
    plt.show()
    '''
    numpy_eeg = np.stack((numpy_d, numpy_t, numpy_a, numpy_b, numpy_g),axis=1)
    if numpy_eeg.shape[0] != react_time.shape[0]:
        print('error')
    del d, t, a, b, g   
    gc.collect()

    np.save(os.path.join(path_save, 'dev', f_name), np.asarray(numpy_eeg))

    del numpy_d, numpy_t, numpy_a, numpy_b, numpy_g, numpy_eeg
    gc.collect()
