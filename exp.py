# coding: utf-8
import pandas as pd
import numpy as np
import sys

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adagrad 

import utils
from grid import RoadGrid

def encode_rncci(rncid, ci):
    return rncid*100+ci

def get_train():
    # Load Training Data
    tr_dnames = ['forward0', 'forward1', 'forward2', 'forward3']
    tr_data = utils.get_4g_data(tr_dnames)
    tr_label = tr_data[['Longitude', 'Latitude']].values 

    # Load Testing Data
    te_dnames = ['backward2']
    te_data = utils.get_4g_data(te_dnames)
    te_label = te_data[['Longitude', 'Latitude']].values

    # Load Engineering Parameter
    eng_para = utils.get_4g_engpara()

    print '训练集大小: %d' % len(tr_data)
    print '测试集大小: %d' % len(te_data)

    # Grid Data
    tr_time, tr_feature, tr_label_ = utils.make_rf_dataset(tr_data, eng_para)
    te_time, te_feature, te_label_ = utils.make_rf_dataset(te_data, eng_para)
    rg = RoadGrid(tr_label_.values, 10)
    tr_label_ = np.array(rg.transform(tr_label_.values, False))
    print '格子数量: %d' % rg.n_grid

    # Get all RNC_CI
    rncci_set = set()
    for i in xrange(1,7):
        rncci_set |= set(encode_rncci(tr_feature['RNCID_%d'%i].values.astype(int), tr_feature['CellID_%d'%i].values.astype(int)))
    rncci_set.remove(encode_rncci(-999,-999))
    rncci_dict = dict(zip(list(rncci_set), range(len(rncci_set))))

    # Grid Statistics 
    dense_feature = [1] * len(tr_feature)
    sparse_feature = [1] * len(tr_feature)
    for i in xrange(rg.n_grid):
        sub_feature = tr_feature[tr_label_ == i]
        bs_count = np.zeros(len(rncci_set))
        bs_rscp = np.zeros(len(rncci_set))
        bs_ecno = np.zeros(len(rncci_set))
        for idx, row in sub_feature.iterrows():
            sp_rscp = np.zeros(len(rncci_set))
            sp_ecno = np.zeros(len(rncci_set))
            for j in xrange(1, 7):
                rncid = int(row['RNCID_%d'%j])
                ci = int(row['CellID_%d'%j])
                rscp = float(row['RSCP_%d'%j])
                ecno = float(row['EcNo_%d'%j])
                if rncid != -999 and ci != -999 and rscp != -999 and ecno != -999 and rscp != 0 and ecno != 0:
                    dict_idx = rncci_dict[encode_rncci(rncid, ci)]
                    bs_count[dict_idx] += 1
                    bs_rscp[dict_idx] += (rscp+140) # add constant scalar to be a positive number
                    bs_ecno[dict_idx] += (ecno+30)
                    sp_rscp[dict_idx] = (rscp+140)
                    sp_ecno[dict_idx] = (ecno+30)
            sparse_feature[idx-1] = list(np.hstack((sp_rscp, sp_ecno)))
        radio_map = np.hstack((bs_rscp / bs_count, bs_ecno / bs_count))
        nan_mask = np.array([np.isnan(x) for x in radio_map])
        nan_idx = np.where(nan_mask == True)[0]
        radio_map[nan_idx] = 0
        for idx, row in sub_feature.iterrows():
            dense_feature[idx-1] = list(radio_map)

    dense_feature = np.asarray(dense_feature)
    sparse_feature = np.asarray(sparse_feature)
    return sparse_feature, dense_feature

def build_network(n_dim, n_hidden):
    model = Sequential()
    model.add(Dense(output_dim=n_hidden, input_dim=n_dim))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=n_dim))

    return model

def main(n_hidden = 500, nb_epoch = 10, batch_size = 128):
    tr_feature, tr_label = get_train()

    network = build_network(tr_feature.shape[1], n_hidden)
    network.compile(loss='mean_squared_error', optimizer=Adagrad())
    network.summary()
    network.fit(tr_feature, tr_label, nb_epoch=nb_epoch, batch_size=batch_size, validation_split=0.1, verbose=1)

    network.to_json()
    
if __name__ == "__main__":
    params = {
        'n_hidden': 500,
        'nb_epoch': 100,
        'batch_size': 128
    }
    main(**params)
