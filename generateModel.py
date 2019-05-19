import pickle, os, sys

import numpy as np

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from src.application import Application

models = ['multi_attention', 'bi_lstm', 'ap_bi_lstm', 'ap_bi_gru', 'bi_gru', 'cnn', 'ap_cnn']
for model in models:
    predicts = []
    with open("./data/Linux_25_" + model + ".csv", 'rb') as file:
        for line in file:
            predicts.append(float(line.decode().split("\t")[0]))
    predicts = np.asarray(predicts).reshape([len(predicts), 1])
    with open(Application.directory['model'] + model + Application.model['predict'], 'wb') as f:
        pickle.dump(predicts, f)
