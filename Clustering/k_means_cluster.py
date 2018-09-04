from sklearn.cluster import KMeans
from utils import *
import math
import numpy as np
from scipy.stats import mode
# filename = 'dataset/ng20.tfidf.mat'  # class 20 ;0.4618320610687023
# filename = 'dataset/reuters.tfidf.mat'  #class 22;result：0.8500517063081696
filename = 'dataset/tmc.tfidf.mat'  # class 22;  0.4908519153802173

data = Load_Dataset(filename)

K = 22


def calc_acc(data, pred, K):
    label = data.gnd_test.argmax(1)
    cat_pred = pred
    real_pred = np.zeros_like(cat_pred)
    for cat in range(K):
        idx = cat_pred == cat
        lab = label[idx]
        if len(lab) == 0:
            continue
        real_pred[cat_pred == cat] = mode(lab).mode[0]
    return np.mean(real_pred == label)


def train(data, K, show_label=False):
    k_means = KMeans(n_clusters=K, init='k-means++', max_iter=500, n_init=1,
                     verbose=False)
    k_means.fit(data.train)
    result = list(k_means.predict(data.test))
    pred = k_means.predict(data.test)
    acc = calc_acc(data, pred, K)
    print('accuracy: ', acc)


print(filename)

acc = train(data, K)
