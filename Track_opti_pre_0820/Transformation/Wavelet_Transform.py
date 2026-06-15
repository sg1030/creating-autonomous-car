import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from Track.track import Track
import pywt
import torch
import random

def Wavelet_Transform(target,wavelet,level):
    coeffs=pywt.wavedec(target,wavelet,level)
    return coeffs

def Wavelet_Inverse_Transform(wavelet,coeffs,length):
    Real_Signal=pywt.waverec(coeffs, wavelet)
    Real_Signal=Real_Signal[:length]

    return Real_Signal