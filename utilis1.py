import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler
__all__ = ['loaddata']




def plot_results(data,labels,test_labels,win,strike,title = 'results'):
    len_max = np.min(len(data),len(labels),len(test_labels))
    for i in range(0, len_max-win, strike):
        plt.title(title)
        plt.plot(data[i:i+win],labels='data')
        plt.plot(labels[i:i + win], labels='data')
        plt.plot(test_labels[i:i + win], labels='data')
        plt.show()

def gen_seq(filename,condition_len,predict_len,strike):
    a = pd.read_csv(filename)
    all_value = a['value'].values
    all_label = a['label'].values
    winlen = condition_len+predict_len
    rows = len(range(0, len(all_value)-winlen+1,strike))
    array1 = np.zeros((rows, condition_len))
    array2 = np.zeros((rows, predict_len))

    j = 0
    for i in range(0, len(all_value)-winlen, strike):
        tmp_label = all_label[i:i+winlen]
        array1[j] = all_value[i:i+condition_len]
        array2[j] = all_value[i+condition_len:i+winlen]
        j += 1
    return array1, array2

def gen_seq_normal(filename,condition_len,predict_len,strike):
    a = pd.read_csv(filename)
    all_value = a['value'].values
    all_label = a['label'].values
    winlen = condition_len+predict_len
    rows = len(range(0, len(all_value)-winlen+1,strike))
    array1 = np.zeros((rows, condition_len))
    array2 = np.zeros((rows, predict_len))

    j = 0
    for i in range(0, len(all_value)-winlen, strike):
        tmp_label = all_label[i:i+winlen]
        if(tmp_label.sum() > 0):
            continue
        array1[j] = all_value[i:i+condition_len]
        array2[j] = all_value[i+condition_len:i+winlen]
        j += 1
    return array1, array2

def gen_train_data_normal(filename,winlen,strike):
    a = pd.read_csv(filename)
    all_value = a['value'].values
    all_label = a['label'].values

    # all_value = all_value.reshape((1,-1))
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # all_value = scaler.fit_transform(all_value)
    # all_value = all_value[0]
    s = 0
    end = 72
    thres = 0.4
    testn = int(s * 1440 + (end - s) * 1440 * thres)
    array = all_value[testn:end * 1440]
    label = all_label[testn:end * 1440]

    all_value = all_value[s*1440:testn]
    all_label = all_label[s*1440:testn]

    array = all_value[0:winlen]
    array = np.reshape(array, [-1, winlen])
    for i in range(0,len(all_value)-winlen-strike, strike):
        randomseed = np.random.randint(strike)
        temlabs = all_label[i+randomseed:i+winlen+randomseed]
        temvalue = all_value[i+randomseed:i+winlen+randomseed]
        if temlabs.sum() > 0:
            continue
        array = np.concatenate([array, np.reshape(temvalue,[-1,winlen])],0)
    return array

def gen_train_data_normal_2(filename,winlen,strike):
    a = pd.read_csv(filename)
    all_value = a['value'].values
    all_label = a['label'].values

    # all_value = all_value.reshape((1,-1))
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # all_value = scaler.fit_transform(all_value)
    # all_value = all_value[0]
    s = 0
    end = 72
    thres = 0.6
    testn = int(s * 1440 + (end - s) * 1440 * thres)
    array = all_value[testn:end * 1440]
    label = all_label[testn:end * 1440]

    all_value = all_value[s*1440:testn]
    all_label = all_label[s*1440:testn]

    array = all_value[0:winlen]
    array = np.reshape(array, [-1, winlen])
    for i in range(0,len(all_value)-winlen-strike, strike):
        randomseed = np.random.randint(strike)
        temlabs = all_label[i+randomseed:i+winlen+randomseed]
        temvalue = all_value[i+randomseed:i+winlen+randomseed]
        if temlabs.sum() > 0:
            continue
        array = np.concatenate([array, np.reshape(temvalue,[-1,winlen])],0)
    return array

def gen_train_data_normal_3(filename,winlen,strike):
    a = pd.read_csv(filename)
    all_value = a['value'].values
    all_label = a['label'].values

    # all_value = all_value.reshape((1,-1))
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # all_value = scaler.fit_transform(all_value)
    # all_value = all_value[0]
    s = 0
    end = 72
    thres = 0.8
    testn = int(s * 1440 + (end - s) * 1440 * thres)
    array = all_value[testn:end * 1440]
    label = all_label[testn:end * 1440]

    all_value = all_value[s*1440:testn]
    all_label = all_label[s*1440:testn]

    array = all_value[0:winlen]
    array = np.reshape(array, [-1, winlen])
    for i in range(0,len(all_value)-winlen-strike, strike):
        randomseed = np.random.randint(strike)
        temlabs = all_label[i+randomseed:i+winlen+randomseed]
        temvalue = all_value[i+randomseed:i+winlen+randomseed]
        if temlabs.sum() > 0:
            continue
        array = np.concatenate([array, np.reshape(temvalue,[-1,winlen])],0)
    return array
def gen_train_data(filename,winlen,strike):
    a = pd.read_csv(filename)
    all_value = a['value'].values
    all_label = a['label'].values

    all_value = all_value[0:1440 * 60]
    all_label = all_label[0:1440 * 60]

    rows = len(range(0, len(all_value)-winlen+1,strike))
    columns = winlen
    array = np.zeros((rows, columns))
    label = np.zeros((rows, columns))
    j = 0
    for i in range(0,len(all_value)-winlen+1,strike):
        array[j] = all_value[i:i+winlen]
        label[j] = all_label[i:i+winlen]
        j += 1
    return array, label


def gen_labeled_data(filename,winlen,strike):
    a = pd.read_csv(filename)
    all_value = a['value'].values
    all_label = a['label'].values

    all_value = all_value[0:1440 * 72]
    all_label = all_label[0:1440 * 72]

    rows = len(range(0, len(all_value)-winlen+1,strike))
    columns = winlen
    array = np.zeros((rows, columns))
    label = np.zeros((rows, 2))
    valid = np.array([0,1])
    flase = np.array([1,0])
    j = 0
    for i in range(0,len(all_value)-winlen+1,strike):
        array[j] = all_value[i:i+winlen]

        if np.sum(all_label[i:i+winlen]) > 0:
            label[j] = flase
        else:
            label[j] = valid
        j += 1
    return array, label

def gen_labeled_data1(filename,winlen,strike):
    a = pd.read_csv(filename)
    all_value = a['value'].values
    all_label = a['label'].values
    all_value[63*1440+960:63*1440+1000]=all_value[63*1440+960:63*1440+1000]*0.3



    s = 0
    end = 72
    thres = 0.4
    testn = int (s*1440+(end-s)*1440*thres)
    all_value = all_value[testn:end*1440]
    all_label = all_label[testn:end*1440]

    rows = len(range(0, len(all_value)-winlen+1,strike))
    columns = winlen
    array = np.zeros((rows, columns))
    label = np.zeros((rows, 2))
    valid = np.array([0,1])
    flase = np.array([1,0])
    j = 0
    for i in range(0,len(all_value)-winlen+1,strike):
        array[j] = all_value[i:i+winlen]

        if np.sum(all_label[i:i+winlen])>0:
            label[j] = flase
        else:
            label[j] = valid
        j += 1
    return array, label

def gen_labeled_data1_2(filename,winlen,strike):
    a = pd.read_csv(filename)
    all_value = a['value'].values
    all_label = a['label'].values
    all_value[63*1440+960:63*1440+1000]=all_value[63*1440+960:63*1440+1000]*0.3



    s = 0
    end = 72
    thres = 0.6
    testn = int (s*1440+(end-s)*1440*thres)
    all_value = all_value[testn:end*1440]
    all_label = all_label[testn:end*1440]

    rows = len(range(0, len(all_value)-winlen+1,strike))
    columns = winlen
    array = np.zeros((rows, columns))
    label = np.zeros((rows, 2))
    valid = np.array([0,1])
    flase = np.array([1,0])
    j = 0
    for i in range(0,len(all_value)-winlen+1,strike):
        array[j] = all_value[i:i+winlen]

        if np.sum(all_label[i:i+winlen])>0:
            label[j] = flase
        else:
            label[j] = valid
        j += 1
    return array, label

def gen_labeled_data1_3(filename,winlen,strike):
    a = pd.read_csv(filename)
    all_value = a['value'].values
    all_label = a['label'].values
    all_value[63*1440+960:63*1440+1000]=all_value[63*1440+960:63*1440+1000]*0.3



    s = 0
    end = 72
    thres = 0.8
    testn = int (s*1440+(end-s)*1440*thres)
    all_value = all_value[testn:end*1440]
    all_label = all_label[testn:end*1440]

    rows = len(range(0, len(all_value)-winlen+1,strike))
    columns = winlen
    array = np.zeros((rows, columns))
    label = np.zeros((rows, 2))
    valid = np.array([0,1])
    flase = np.array([1,0])
    j = 0
    for i in range(0,len(all_value)-winlen+1,strike):
        array[j] = all_value[i:i+winlen]

        if np.sum(all_label[i:i+winlen])>0:
            label[j] = flase
        else:
            label[j] = valid
        j += 1
    return array, label


def get_recall(true_label, test_label, acceptance_delay):
    tatol_number = np.sum(true_label)
    new_detected_number = 0
    detection_delay = 0
    detected_number = 0
    calcul_number = 0
    continous_Flag = False
    detected = False
    for i in range(len(true_label)):
        if true_label[i]==1:
            calcul_number = calcul_number+1
            if np.sum(test_label[i-1:i + acceptance_delay])>0:
                detected = True
                for j in range(acceptance_delay):
                    if test_label[i+j]>0:
                        new_detected_number = new_detected_number+1
                        detection_delay = detection_delay +j
                        break
            continous_Flag = True
        else:
            if continous_Flag & detected:
                    detected_number = detected_number +calcul_number
            calcul_number = 0
            continous_Flag = False
            detected = False
    print(detected_number)
    return  detection_delay/(new_detected_number +0.0001) ,detected_number*1.0/tatol_number


def get_precision(true_label, test_label, acceptance_delay):
    #tatol_number = np.sum(true_label)
    detected_number1 = 0
    calcul_number = 0
    continous_Flag = False
    detected = False
    for i in range(len(true_label)):
        if true_label[i]==1:
            calcul_number = calcul_number+1
            continous_Flag = True
            if np.sum(test_label[i:i + acceptance_delay])>0:
                detected = True
        else:
            if continous_Flag & detected:
                    detected_number1 = detected_number1 +calcul_number
            calcul_number = 0
            continous_Flag = False
            detected = False



    false_alter = 0
    detected_record = np.zeros(len(true_label))
    tatol_number = np.sum(test_label)
    detected_number = 0
    calcul_number = 0
    continous_Flag = False
    detected = False
    for i in range(len(test_label)):
        if test_label[i]==1:
            calcul_number = calcul_number+1
            continous_Flag = True
            if np.sum(true_label[i-acceptance_delay:i])>0:
                detected = True
                detected_record[i] = 1
        else:
            if continous_Flag & detected:
                    detected_number = detected_number +calcul_number
            if continous_Flag &~detected:
                  false_alter = false_alter+calcul_number
            calcul_number = 0
            continous_Flag = False
            detected = False
    return (detected_number1+detected_number*1.0)/(tatol_number+detected_number1), detected_record,false_alter*0.1/len(true_label)



def get_AUC(true_label, test_label, acceptance_delay):
    Assume_label = np.zeros(len(true_label))
    test_label = test_label[0:len(true_label)]
    for i in range(len(true_label)):
        if true_label[i]==1:
            for j in range(0,acceptance_delay):
                if test_label[i+j]==1:
                    if i+j <len(true_label):
                        Assume_label[i+j] = 1
    a,b,c = roc_curve(Assume_label,test_label,pos_label=1)
    c1 = auc(a,b)
    return c1


def gen_labeled_data11(filename,winlen,strike):
    a = pd.read_csv(filename)
    all_value = a['value'].values
    all_label = a['label'].values


    all_value = all_value[1440 * 60:1440 * 72]
    all_label = all_label[1440 * 60:1440 * 72]

    rows = len(range(0, len(all_value)-winlen+1,strike))
    columns = winlen
    array = np.zeros((rows, columns))
    label = np.zeros((rows, 2))
    valid = np.array([0,1])
    flase = np.array([1,0])
    j = 0
    for i in range(0,len(all_value)-winlen+1,strike):
        array[j] = all_value[i:i+winlen]

        if np.sum(all_label[i:i+winlen])>0:
            label[j] = flase
        else:
            label[j] = valid
        j += 1
    return array, label


def gen_labeled_data22(filename,winlen,strike):
    a = pd.read_csv(filename)
    all_value = a['value'].values
    all_label = a['label'].values


    all_value = all_value[1440 * 70:1440 * 72]
    all_label = all_label[1440 * 70:1440 * 72]

    rows = len(range(0, len(all_value)-winlen+1,strike))
    columns = winlen
    array = np.zeros((rows, columns))
    label = np.zeros((rows, 2))
    valid = np.array([0,1])
    flase = np.array([1,0])
    j = 0
    for i in range(0,len(all_value)-winlen+1,strike):
        array[j] = all_value[i:i+winlen]

        if np.sum(all_label[i:i+winlen])>0:
            label[j] = flase
        else:
            label[j] = valid
        j += 1
    return array, label

def gen_labeled_data33(filename,winlen,strike):
    a = pd.read_csv(filename)
    all_value = a['value'].values
    all_label = a['label'].values
    all_value[63*1440+960:63*1440+1000] = all_value[63*1440+960:63*1440+1000]*0.3


    all_value = all_value[1440 * 63:1440 * 72]
    all_label = all_label[1440 * 63:1440 * 72]

    rows = len(range(0, len(all_value)-winlen+1,strike))
    columns = winlen
    array = np.zeros((rows, columns))
    label = np.zeros((rows, 2))
    valid = np.array([0,1])
    flase = np.array([1,0])
    j = 0
    for i in range(0,len(all_value)-winlen+1,strike):
        array[j] = all_value[i:i+winlen]

        if np.sum(all_label[i:i+winlen])>0:
            label[j] = flase
        else:
            label[j] = valid
        j += 1
    return array, label


def gen_labeled_data2(filename):
    a = pd.read_csv(filename)
    all_value = a['value'].values
    all_label = a['label'].values
    all_value[63 * 1440 + 960:63 * 1440 + 1000] = all_value[63 * 1440 + 960:63 * 1440 + 1000] * 0.3
    s = 0
    end = 72
    thres = 0.4
    testn = int(s*1440+(end-s)*1440*thres)
    array = all_value[testn:end*1440]
    label = all_label[testn:end*1440]


    return array, label


def gen_labeled_data2_2(filename):
    a = pd.read_csv(filename)
    all_value = a['value'].values
    all_label = a['label'].values
    all_value[63 * 1440 + 960:63 * 1440 + 1000] = all_value[63 * 1440 + 960:63 * 1440 + 1000] * 0.3
    s = 0
    end = 72
    thres = 0.6
    testn = int(s*1440+(end-s)*1440*thres)
    array = all_value[testn:end*1440]
    label = all_label[testn:end*1440]


    return array, label

def gen_labeled_data2_3(filename):
    a = pd.read_csv(filename)
    all_value = a['value'].values
    all_label = a['label'].values
    all_value[63 * 1440 + 960:63 * 1440 + 1000] = all_value[63 * 1440 + 960:63 * 1440 + 1000] * 0.3
    s = 0
    end = 72
    thres = 0.8
    testn = int(s*1440+(end-s)*1440*thres)
    array = all_value[testn:end*1440]
    label = all_label[testn:end*1440]


    return array, label
def loaddata(fileName):
    trainingdata=[]
    with open(fileName) as txtData:
        lines=txtData.readlines()
        for line in lines:
            linedata=line.strip().split(',')
            trainingdata.append(linedata)

    character = []
    for i in range(len(trainingdata)):
        character.append([float(tk) for tk in trainingdata[i][:]])
    return np.array(character)


def standardize_kpi(values, mean=None, std=None, excludes=None):
    """
    Standardize a
    Args:
        values (np.ndarray): 1-D `float32` array, the KPI observations.
        mean (float): If not :obj:`None`, will use this `mean` to standardize
            `values`. If :obj:`None`, `mean` will be computed from `values`.
            Note `mean` and `std` must be both :obj:`None` or not :obj:`None`.
            (default :obj:`None`)
        std (float): If not :obj:`None`, will use this `std` to standardize
            `values`. If :obj:`None`, `std` will be computed from `values`.
            Note `mean` and `std` must be both :obj:`None` or not :obj:`None`.
            (default :obj:`None`)
        excludes (np.ndarray): Optional, 1-D `int32` or `bool` array, the
            indicators of whether each point should be excluded for computing
            `mean` and `std`. Ignored if `mean` and `std` are not :obj:`None`.
            (default :obj:`None`)

    Returns:
        np.ndarray: The standardized `values`.
        float: The computed `mean` or the given `mean`.
        float: The computed `std` or the given `std`.
    """
    values = np.asarray(values, dtype=np.float32)
    if (mean is None) != (std is None):
        raise ValueError('`mean` and `std` must be both None or not None')
    if excludes is not None:
        excludes = np.asarray(excludes, dtype=np.bool)
        if excludes.shape != values.shape:
            raise ValueError('The shape of `excludes` does not agree with '
                             'the shape of `values` ({} vs {})'.
                             format(excludes.shape, values.shape))

    if mean is None:
        if excludes is not None:
            val = values[np.logical_not(excludes)]
        else:
            val = values
        mean = val.min()
        std = val.std()

    return ((values - mean) / std).astype(np.float32), mean, std

def getNext_batch(data,labels, iter_num=0, batch_size=64):

    ro_num = len(data) / batch_size - 1
    if iter_num % ro_num == 0:

        length = len(data)
        perm = np.arange(length)
        np.random.shuffle(perm)
        data = np.array(data)
        data = data[perm]
        labels = np.array(labels)
        labels = labels[perm]

    return data[int((iter_num % ro_num) * batch_size): int((iter_num% ro_num + 1) * batch_size)] \
, labels[int((iter_num % ro_num) * batch_size): int((iter_num%ro_num + 1) * batch_size)]

def getNext_batch_single(data, iter_num=0, batch_size=64):

    ro_num = len(data) / batch_size - 1
    if iter_num % ro_num == 0:
        length = len(data)
        perm = np.arange(length)
        np.random.shuffle(perm)
        data = np.array(data)
        data = data[perm]

    return data[int((iter_num % ro_num) * batch_size): int((iter_num% ro_num + 1) * batch_size)]

def get_recall1(true_label, test_label):
    mixlabel = true_label+test_label
    target_num = np.sum(mixlabel==2)
    all_mun = np.sum(true_label == 1)
    return 1.0*target_num/(all_mun+1e-09)


def get_acc(true_label, test_label):
    mixlabel = true_label + test_label
    target_num = np.sum(mixlabel == 2)
    all_mun = np.sum(test_label == 1)
    return 1.0 * target_num / (all_mun+1e-09)

def get_error(true_label, test_label,batchsize):
    a = true_label + test_label
    flase_num = np.sum(a == 1)
    flase_rate = 1.0 * flase_num / batchsize
    return flase_rate