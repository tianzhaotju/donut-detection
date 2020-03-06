import numpy as np
from donut import complete_timestamp, standardize_kpi, Donut, DonutTrainer, DonutPredictor
import pandas as pd
import tensorflow as tf
import keras as K
#Keras==2.0.1
from tfsnippet.modules import Sequential
from utilis1 import  get_recall, get_precision
import time
from tensorflow.contrib import  rnn
from utilis1 import getNext_batch_single, gen_train_data_normal
from tfsnippet.utils import (VarScopeObject,
                             reopen_variable_scope,
                             get_default_session_or_error)
from tfsnippet.utils import get_variables_as_dict, VariableSaver
from matplotlib import pyplot as plt
import csv

save_dir = "./save_dir"
# Read the raw data.
raw_data = pd.read_csv("tagdata.csv")
def get_data():
    values = []
    labels = []
    timestamp = []
    for i in range(len(raw_data['value1'])):
        values.append(raw_data['value1'][i])
        labels.append(raw_data['label1'][i])
        timestamp.append(i)
    values,labels,timestamp = np.array(values),np.array(labels),np.array(timestamp)

    # Complete the timestamp, and obtain the missing point indicators.
    timestamp, missing, (values, labels) = \
        complete_timestamp(timestamp, (values, labels))


    # Split the training and testing data.
    test_portion = 0.2
    test_n = int(len(values) * test_portion)
    train_values, test_values = values[:-test_n], values[-test_n:]
    train_labels, test_labels = labels[:-test_n], labels[-test_n:]
    train_missing, test_missing = missing[:-test_n], missing[-test_n:]

    # Standardize the training and testing data.
    train_values, mean, std = standardize_kpi(
        train_values, excludes=np.logical_or(train_labels, train_missing))
    test_values, _, _ = standardize_kpi(test_values, mean=mean, std=std)
    return train_values,train_labels,train_missing,mean,std,test_values,test_labels,test_missing

train_values,train_labels,train_missing,mean,std,test_values,test_labels,test_missing = get_data()


# We build the entire model within the scope of `model_vs`,
# it should hold exactly all the variables of `model`, including
# the variables created by Keras layers.
with tf.variable_scope('model') as model_vs:
    model = Donut(
        h_for_p_x=Sequential([
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        h_for_q_z=Sequential([
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
            K.layers.Dense(100, kernel_regularizer=K.regularizers.l2(0.001),
                           activation=tf.nn.relu),
        ]),
        x_dims=120,
        z_dims=5,
    )


trainer = DonutTrainer(model=model, model_vs=model_vs, max_epoch=300)
predictor = DonutPredictor(model)

with tf.Session().as_default():
    trainer.fit(train_values, train_labels, train_missing, mean, std)

    print('Testing size:', np.shape(test_values))
    start_time = time.time()
    test_score = predictor.get_score(test_values, test_missing)
    end_time = time.time()
    print('time comsuming', end_time - start_time)


    writer = csv.writer(open('donut_result.csv', 'w', newline=''))
    print(len(test_labels), len(test_score), len(test_values))
    for i in range(len(test_score)):
        writer.writerow([test_values[i + 119], test_labels[i + 119], test_score[i]])
    # test_score_save = []
    # test_score[test_score >= -1.7 * 20] = 0
    # test_score[test_score < -1.7 * 20] = 1
    # for i in range(12):
    #     plt.plot(test_values[i * 1440 + 120:(i + 1) * 1440 + 120])
    #     plt.plot(test_score[i * 1440:(i + 1) * 1440])
    #     plt.plot(test_labels[i * 1440 + 120:(i + 1) * 1440 + 120])
        # plt.savefig(str(i) + ".png")
        # plt.show()
    # accuracy = get_accuracy(test_labels[120:],test_score,10,-50)
    # print("Accuracy is :",accuracy)

    # recall = get_recall(test_labels[120:], test_score, acceptance_delay=15)
    # precisioin, _, _ = get_precision(test_labels[120:], test_score, acceptance_delay=15)
    # print(np.shape(recall))
    # print(np.shape(precisioin))
    # F = (2 * recall * precisioin) / (recall + precisioin)
    # print(recall)
    # print(precisioin)
    # print(F)