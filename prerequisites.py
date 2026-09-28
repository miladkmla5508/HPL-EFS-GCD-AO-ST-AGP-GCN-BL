!pip install yfinance
!pip install scikit-learn
!pip install ta
!pip install mplfinance
!pip install openpyxl
!pip install ta-lib
!pip install statsmodels
!pip install tensorflow
!pip install keras-tuner
!pip install optuna
!pip install numba
!pip install torch
!pip install cvxpy
!pip install osqp
!pip install ecos
!pip install arch
!pip install pyvinecopulib
!pip install XlsxWriter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import mplfinance
import yfinance as yf
import sklearn
import ta
import openpyxl
import talib
import statsmodels
import statsmodels.api as sm

import tensorflow as tf
import keras_tuner
import optuna
import numba
import torch
import cvxpy as cp
import osqp
import ecos
import arch
import pyvinecopulib
import xlsxwriter

from scipy.stats import skew, kurtosis, norm
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from statsmodels.tsa.ar_model import AutoReg
from numba import njit, prange


print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Seaborn:", sns.__version__)
print("Matplotlib:", matplotlib.__version__)
print("mplfinance:", mplfinance.__version__)
print("yfinance:", yf.__version__)
print("scikit-learn:", sklearn.__version__)
print("openpyxl:", openpyxl.__version__)
print("TA-Lib:", talib.__version__)
print("statsmodels:", statsmodels.__version__)
print("TensorFlow:", tf.__version__)
print("Keras Tuner:", keras_tuner.__version__)
print("Optuna:", optuna.__version__)
print("Numba:", numba.__version__)
print("PyTorch:", torch.__version__)
print("CVXPY:", cp.__version__)
print("OSQP:", osqp.__version__)
print("ECOS:", ecos.__version__)
print("arch:", arch.__version__)
print("pyvinecopulib:", pyvinecopulib.__version__)
print("XlsxWriter:", xlsxwriter.__version__)
