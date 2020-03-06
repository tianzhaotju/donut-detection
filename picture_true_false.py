import numpy as np
from donut import complete_timestamp, standardize_kpi,Donut,DonutTrainer, DonutPredictor
import pandas as pd
import tensorflow as tf
import keras as K
from tfsnippet.modules import Sequential
from tensorflow.contrib import  rnn
from utilis1 import getNext_batch_single, gen_train_data_normal
from tfsnippet.utils import (VarScopeObject,
                             reopen_variable_scope,
                             get_default_session_or_error)
from tfsnippet.utils import get_variables_as_dict, VariableSaver
from matplotlib import pyplot as plt
import csv

# Read the raw data.
data = pd.read_csv("phase2_train.csv")
# values1, labels1 = np.array(data["value"]),np.array(data["label"])

# timestamp1 = []
# for i in range(len(values1)):
#     timestamp1.append(i)
batchsize  = 64
values = []
timestamp = []
labels = []
N = 1440*72
for i in range(1200):
    values.append(data['value'][i])
    labels.append(data['label'][i])
    timestamp.append(i)
import random
for i in range(1800,1880):
    values.append(data['value'][i]+random.uniform(-0.08,0.14))
    labels.append(data['label'][i])
    timestamp.append(i)
for i in range(1880,1960):
    values.append(data['value'][i]+random.uniform(0.08,0.12))
    labels.append(data['label'][i])
    timestamp.append(i)
for i in range(1960,2040):
    values.append(data['value'][i]+random.uniform(0,0.1))
    labels.append(data['label'][i])
    timestamp.append(i)

plt.figure()
plt.plot(range(0,1200),values[0:1200])
plt.plot(range(1200,1440),values[1200:],color='r')
plt.savefig('false.eps',format='eps')
plt.show()

exit()