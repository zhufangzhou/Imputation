# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import math
import sys
from sklearn.ensemble import RandomForestRegressor
rc = 6378137
rj = 6356725

def rad(d):
    return d * math.pi / 180.0

def distance(true_pt, pred_pt):
    lat1 = float(true_pt[1])
    lng1 = float(true_pt[0])
    lat2 = float(pred_pt[1])
    lng2 = float(pred_pt[0])
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a/2),2) +
    math.cos(radLat1)*math.cos(radLat2)*math.pow(math.sin(b/2),2)))
    s = s * 6378.137
    s = round(s * 10000) / 10
    return s

lu_dpath = '../位置精度算法测试数据/位置精度算法测试-24G路测/'
bu_dpath = '../位置精度算法测试数据/位置精度算法测试-24G步测/'
data_4g = {
    'backward1': lu_dpath+'路测数据导出4G反向-1.csv',
    'forward1': lu_dpath+'路测数据导出4G正向合并.xlsx',
    'backward2': lu_dpath+'路测数据导出4G反向-2.xlsx',
    'forward2': lu_dpath+'路测数据导出4G正向-2.xlsx',
    'backward3': lu_dpath+'路测数据导出4G反向-3.csv',
    'forward3': lu_dpath+'路测数据导出4G正向-3.csv',
    'bu_1': bu_dpath+'步测数据导出4G-1.xlsx',
    'bu_2': bu_dpath+'步测数据导出4G-2.csv',
}
col_name = [
'Longitude',
'Latitude',
'ECI(eNodeBID/CellID)',
'PCC Serving Cell EARFCN',
'PCC Serving Cell PCI',
'PCC Serving Cell RSRP(dBm)',
'PCC Serving Cell RSRQ(dB)',
'PCC Serving Cell RSSI(dBm)',
'Listed Cell EARFCN',
'Listed Cell PCI',
'Listed Cell RSRP(dBm)',
'Listed Cell RSRQ(dB)',
'Listed Cell RSSI(dBm)',
'Detected Cell EARFCN',
'Detected Cell PCI',
'Detected Cell RSRP(dBm)',
'Detected Cell RSRQ(dB)',
'Detected Cell RSSI(dBm)',
]
fold_col_name = col_name[6:]

eng_para = pd.read_csv('../工参/4g_engpara.csv')
eng_para_lnglat = eng_para[['CGI','经度','纬度']]
eng_para_lnglat = eng_para_lnglat[eng_para_lnglat['CGI'].notnull()]
# 把CGI字段拆成两列，比如把460-00-107797-3拆成107797和3两列
def split_cgi(cgi):
    cgi = cgi.split('-')
    enodebid = int(cgi[2])
    ci = int(cgi[3])
    return [enodebid, ci]

# 加上两列enodebid, ci用来匹配数据
engpara_enodebid_ci = np.asarray(map(split_cgi, eng_para_lnglat['CGI'].values))
eng_para_lnglat['enodebid'] = engpara_enodebid_ci[:, 0]
eng_para_lnglat['ci'] = engpara_enodebid_ci[:, 1]

def get_4g_data(dname):
    #dname = 'backward1'
    data = pd.read_csv('%s' % (data_4g[dname])) if data_4g[dname].find('csv') >= 0 else pd.read_excel('%s' % (data_4g[dname]))
    data = data[data['Longitude'].notnull()]
    data['ECI(eNodeBID/CellID)'] = data['ECI(eNodeBID/CellID)'].fillna('-1/-1')
    def split_eci(eci):
        eci = eci.split('/')
        enodebid = int(eci[0])
        ci = int(eci[1])
        return [enodebid, ci]

    data_enodebid_ci = np.asarray(map(split_eci, data['ECI(eNodeBID/CellID)'].values))
    data['enodebid'] = data_enodebid_ci[:, 0]
    data['ci'] = data_enodebid_ci[:, 1]
    return data[col_name + ['Date & Time', 'enodebid', 'ci']]

def make_dataset(fnames, max_neigh):
    default_value = -999.
    data = pd.DataFrame()
    for dname in fnames:
        print dname
        data_ = get_4g_data(dname)
        data = pd.concat([data, data_])
    data = pd.merge(left=data,right=eng_para_lnglat,left_on=['enodebid','ci'],right_on=['enodebid','ci'])

    feature = data[['PCC Serving Cell EARFCN','PCC Serving Cell PCI','PCC Serving Cell RSRP(dBm)','经度','纬度', 'enodebid', 'ci']]
    label = data[['Longitude', 'Latitude']]

    for col in fold_col_name:
        ss = [[] for _ in range(max_neigh)]
        data_ = data[col].values
        for x in data_:
            if type(x) == float or type(x) == np.float64:
                if not math.isnan(x):
                    ss[0].append(float(x))
                else:
                    ss[0].append(default_value)
                for i in range(1, max_neigh):
                    ss[i].append(default_value)
            else:
                x_ = str(x).strip().split(';')[0:max_neigh]
                for idx, v in enumerate(x_):
                    if v != '':
                        ss[idx].append(float(v))
                    else:
                        ss[idx].append(default_value)
                for i in range(idx+1, max_neigh):
                    ss[i].append(default_value)
        for i in range(max_neigh):
            feature['%s_%d' % (col, i+1)] = ss[i]
    return data['Date & Time'], feature, label
