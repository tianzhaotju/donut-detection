import  pandas as pd
import matplotlib.pyplot as plt
from utilis1 import  get_recall,get_precision,get_AUC
import csv
import numpy as np
end  = 72
thres = 0.4
data = pd.read_csv("donut_detection_"+str(thres * 100)+"result2.csv")
values = data['value']
labels = data['label']
score = data['score']
score_save =[]
score_save = score.copy()
score[score<22] = 0
score[score>=22] =1
# for i in range(0,len(values)-1440,1440):
#     plt.plot(values[i:i+1440])
#     plt.plot(labels[i:i+1440])
#     plt.plot(score[i:i+1440]+0.2)
#     plt.show()
recall_all =[]
precision_all = []
F_all =[]


writer = csv.writer(open('A_donut_detection_' + str(thres * 100) + '.csv', 'w+', newline=''))

for i in range(23,24,1):
    score = []
    score = score_save.copy()
    score[score < i] = 0
    score[score >= i] = 1
    delay, recall = get_recall(labels, score, acceptance_delay=15)
    print('delay:', delay  )
    precision, _, _ = get_precision(labels, score, acceptance_delay=15)
    auc = get_AUC(labels, score, acceptance_delay=15)
    recall_all.append(recall)
    precision_all.append(precision)
    F_score = (2*precision*recall)/(precision+recall)
    F_all.append(F_score)
    writer.writerow([i,recall,precision,F_score,auc])

print(recall_all)
print(precision_all)
print(F_all)
plt.plot(range(0,80,1), recall_all)
plt.show()
plt.plot(range(0,80,1), precision_all)
plt.show()
plt.plot(range(0,80,1), F_all)
plt.show()


thres = 0.6
data = pd.read_csv("donut_detection_"+str(thres * 100)+"result2.csv")
values = data['value']
labels = data['label']
score = data['score']
score_save =[]
score_save = score.copy()
score[score<22] = 0
score[score>=22] =1
# for i in range(0,len(values)-1440,1440):
#     plt.plot(values[i:i+1440])
#     plt.plot(labels[i:i+1440])
#     plt.plot(score[i:i+1440]+0.2)
#     plt.show()
recall_all =[]
precision_all = []
F_all =[]


writer = csv.writer(open('A_donut_detection_' + str(thres * 100) + '.csv', 'w+', newline=''))

for i in range(0,80,1):
    score = []
    score = score_save.copy()
    score[score < i] = 0
    score[score >= i] = 1
    recall = get_recall(labels, score, acceptance_delay=15)
    precision, _, _ = get_precision(labels, score, acceptance_delay=15)
    auc = get_AUC(labels, score, acceptance_delay=15)
    recall_all.append(recall)
    precision_all.append(precision)
    F_score = (2*precision*recall)/(precision+recall)
    F_all.append(F_score)
    writer.writerow([i,recall,precision,F_score,auc])

print(recall_all)
print(precision_all)
print(F_all)
plt.plot(range(0,80,1), recall_all)
plt.show()
plt.plot(range(0,80,1), precision_all)
plt.show()
plt.plot(range(0,80,1), F_all)
plt.show()



thres = 0.8
data = pd.read_csv("donut_detection_"+str(thres * 100)+"result2.csv")
values = data['value']
labels = data['label']
score = data['score']
score_save =[]
score_save = score.copy()
score[score<22] = 0
score[score>=22] =1
# for i in range(0,len(values)-1440,1440):
#     plt.plot(values[i:i+1440])
#     plt.plot(labels[i:i+1440])
#     plt.plot(score[i:i+1440]+0.2)
#     plt.show()
recall_all =[]
precision_all = []
F_all =[]


writer = csv.writer(open('A_donut_detection_' + str(thres * 100) + '.csv', 'w+', newline=''))

for i in range(0,80,1):
    score = []
    score = score_save.copy()
    score[score < i] = 0
    score[score >= i] = 1
    recall = get_recall(labels, score, acceptance_delay=15)
    precision, _, _ = get_precision(labels, score, acceptance_delay=15)
    auc = get_AUC(labels, score, acceptance_delay=15)
    recall_all.append(recall)
    precision_all.append(precision)
    F_score = (2*precision*recall)/(precision+recall)
    F_all.append(F_score)
    writer.writerow([i,recall,precision,F_score,auc])

print(recall_all)
print(precision_all)
print(F_all)
plt.plot(range(0,80,1), recall_all)
plt.show()
plt.plot(range(0,80,1), precision_all)
plt.show()
plt.plot(range(0,80,1), F_all)
plt.show()
