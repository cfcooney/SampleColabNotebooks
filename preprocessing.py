import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pandas as pd 
import numpy as np 
import pickle
from scipy.signal import butter, lfilter
from scipy.signal import decimate as dec
from tensorflow.keras.utils import normalize 


def load_subject_eeg(data_folder):
    """ returns eeg data corresponding to words and vowels 
        given a subject identifier.
    """
    words_file = 'raw_array_ica.pickle'
    vowels_file = 'raw_array_vowels_ica.pickle'
    
    try:
        with open(data_folder + words_file, 'rb') as f:
            file = pickle.load(f)
    except:
        print("Not on PC! Attempting to load from laptop.")
        with open(data_folder1 + words_file, 'rb') as f:
            file = pickle.load(f)
            
    w_data = file['raw_array'][:][0]
    w_labels = file['labels']

    try:
        with open(data_folder + vowels_file, 'rb') as f:
            file = pickle.load(f)
    except:
        with open(data_folder1 + vowels_file, 'rb') as f:
            file = pickle.load(f)
    v_data = file['raw_array'][:][0]
    v_labels = file['labels']
    return w_data, v_data, w_labels, v_labels

def eeg_to_3d(data, epoch_size, n_events,n_chan):
    """
    function to return a 3D EEG data format from a 2D input.
    Parameters:
      data: 2D np.array of EEG
      epoch_size: number of samples per trial, int
      n_events: number of trials, int
      n_chan: number of channels, int
        
    Output:
      np.array of shape n_events * n_chans * n_samples
    """
    idx, a, x = ([] for i in range(3))
    [idx.append(i) for i in range(0,data.shape[1],epoch_size)]
    for j in data:
        [a.append([j[idx[k]:idx[k]+epoch_size]]) for k in range(len(idx))]
   
    
    return np.reshape(np.array(a),(n_events,n_chan,epoch_size))

def format_data(data, labels, data_type, epoch):
    """
    Returns data into format required for inputting to the CNNs.

    Parameters:
        data_type: str()
        subject_id: str()
        epoch: length of single trials, int
    """
    n_chan = len(data)
    data = eeg_to_3d(data, epoch, int(data.shape[1] / epoch), n_chan).astype(np.float32)
    labels = labels.astype(np.int64)  
    if data_type == 'words':
        labels[:] = [x - 6 for x in labels] # zero-index the labels
    else:
        labels[:] = [x - 1 for x in labels]
    return data, labels

def down_and_normal(x,d):
    """
    Function for downsampling the data, normalizing 
    and improving numerical stability.
    """
    x = dec(x, d) #downsampling
    fn = lambda a: a * 1e6 # improves numerical stability
    x = fn(x)
    x = normalize(x)
    return x.astype(np.float32)


def balanced_subsample(features, targets, random_state=12):
    """
    function for balancing datasets by randomly-sampling data
    according to length of smallest class set.
    """
    from sklearn.utils import resample
    unique, counts = np.unique(targets, return_counts=True)
    unique_classes = dict(zip(unique, counts))
    mnm = len(targets)
    for i in unique_classes:
        if unique_classes[i] < mnm:
            mnm = unique_classes[i]

    X_list, y_list = [],[]
    for unique in np.unique(targets):
        idx = np.where(targets == unique)
        X = features[idx]
        y = targets[idx]
        
        #X1, y1 = resample(X,y,n_samples=mnm, random_state=random_state)
        X_list.append(X[:mnm])
        y_list.append(y[:mnm])
    
    balanced_X = X_list[0]
    balanced_y = y_list[0]
    
    for i in range(1, len(X_list)):
        balanced_X = np.concatenate((balanced_X, X_list[i]))
        balanced_y = np.concatenate((balanced_y, y_list[i]))

    return balanced_X, balanced_y