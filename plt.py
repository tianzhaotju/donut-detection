import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

data = pd.read_csv('donut_data2.csv')
value = data['value'].values
label = data['label'].values
score = data['score'].values

value = np.array(value)
label = np.array(label)
score = np.array(score)


for i in range(30):
    print(i)
    strike = 1440
    win = 1440
    plt.figure()

    plt.plot(range(i * strike, i * strike + win), value[i * strike:i * strike + win])
    plt.plot(range(i * strike, i * strike + win), score[i * strike:i * strike + win])
    plt.plot(range(i * strike, i * strike + win), label[i * strike:i * strike + win])
    plt.legend(["value","score","label"])
    plt.show()

# writer = csv.writer(open('donut_data.csv', 'w', newline=''))
# for i in range(1440,1440*3):
#     writer.writerow([i,value[i],label[i],score[i]])
#