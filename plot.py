import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

data = pd.read_csv('donut_data1.csv')
value = data['value'].values
label = data['label'].values
score = data['score'].values

value = np.array(value)
label = np.array(label)
score = np.array(score)


for i in range(10):
    print(i)
    strike = 1440 * 3
    win = 1440 * 3
    plt.figure()
    plt.plot(range(i * strike, i * strike + win), label[i * strike:i * strike + win])
    plt.plot(range(i * strike, i * strike + win), value[i * strike:i * strike + win])
    plt.plot(range(i * strike, i * strike + win), score[i * strike:i * strike + win])
    plt.legend(["label","value","score"])
    plt.show()

