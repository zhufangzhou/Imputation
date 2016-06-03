# encoding: utf-8
import disney
import pandas as pd
import sys
import numpy as np
from CellSense import CellSense, FingerPrint
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import NMF
from sklearn.externals import joblib
#from keras.optimizers import Adagrad
from mlp import mlp
from grid import RoadGrid, AreaGrid
from filters import Filter
import cPickle as pickle

tr_data = disney.get_2g_data(['forward1', 'forwardbackward2', 'bu_1', 'bu_2'])
te_data = disney.get_2g_data(['backward1'])

eng_para = disney.get_2g_engpara()

tr_time, tr_feature, tr_label = disney.make_rf_dataset(tr_data, eng_para)
te_time, te_feature, te_label = disney.make_rf_dataset(te_data, eng_para)

print tr_feature.shape
print te_feature.shape

# boundary of training area
padding=0.0015
x1 = min(tr_label.x)-padding*3
y1 = min(tr_label.y)-padding*2
x2 = max(tr_label.x)+padding*3
y2 = max(tr_label.y)+padding*2

ag = AreaGrid(2, x1, y1, x2, y2) 

tr_gid = np.array(map(lambda x: ag.gid(x), tr_feature[[u'经度', u'纬度']].values))
te_gid = np.array(map(lambda x: ag.gid(x), te_feature[[u'经度', u'纬度']].values))

print list(te_gid)
sys.exit(0)

error = []
for gid in range(1, len(ag.grids)+1):
    print 'gid --- %d' % gid

    tr_mask = (tr_gid == gid)
    tr_time_ = tr_time[tr_mask]
    tr_feature_ = tr_feature[tr_mask]
    tr_label_ = tr_label[tr_mask]
    print sum(tr_mask.astype(int))

    te_mask = (te_gid == gid)
    te_time_ = te_time[te_mask]
    te_feature_ = te_feature[te_mask]
    te_label_ = te_label[te_mask]
    print sum(te_mask.astype(int))
    continue

    rg = RoadGrid(tr_label_.values, 30)
    tr_label_ = rg.transform(tr_label_.values, False)
    #print rg.n_grid

    print 'train...'
    est = RandomForestClassifier(
        n_jobs=-1,
        n_estimators = 100,
        max_features='sqrt',
        bootstrap=True,
        criterion='gini'
    ).fit(tr_feature_.values, tr_label_)


    print 'predict...'
    te_pred = est.predict_proba(te_feature_.values)
    fl = Filter()    
    top_n = 1
    te_pred_ = te_pred.copy()

    pred_idx = np.argsort(-te_pred, axis=1)
    for i, idx in enumerate(pred_idx[:, top_n:]):
        te_pred_[i, idx] = 0.

    z = np.sum(te_pred_, axis=1)
    z.shape = len(z),1
    te_pred_ = te_pred_ / z

    te_pred_ = te_pred_.dot(rg.grid_center)
    
    te_pred_ = fl.mean_filter(te_pred_, 11, 10)
    error += [disney.distance(pt1, pt2) for pt1, pt2 in zip(te_pred_, te_label_.values)]
sys.exit(0)    
error = np.array(error)
good_count = np.sum((np.array(error) < 100).astype(int))

disney.report(error, 'Random Forest Multi-Classification\t%d\t%d\t%.2f%%' % (len(error), good_count, 100.0*good_count/len(error)))

