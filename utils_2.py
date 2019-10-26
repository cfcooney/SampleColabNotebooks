import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import pandas as pd 
import numpy as np 
import pickle
from scipy.signal import butter, lfilter
from scipy.signal import decimate as dec

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	"""
	Returns signal bandpass filtered by Butterworth.
	"""
	b, a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = lfilter(b, a, data)
	return y


def current_acc(model_acc):
    """
    Returns the maximum validation accuracy from the 
    trained model
    """
    accs_list = []
    [accs_list.append(x) for x in model_acc]
    return np.min(np.array(accs_list))