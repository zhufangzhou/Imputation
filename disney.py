# coding: utf-8
import numpy as np
import math as Math
import pandas as pd
from scipy.sparse import csc_matrix
from CellSense import FingerPrint

rc = 6378137
rj = 6356725

def rad(d):
    return d * Math.pi / 180.0

def distance(true_pt, pred_pt):
    lat1 = float(true_pt[1])
    lng1 = float(true_pt[0])
    lat2 = float(pred_pt[1])
    lng2 = float(pred_pt[0])
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * Math.asin(Math.sqrt(Math.pow(Math.sin(a/2),2) +
    Math.cos(radLat1)*Math.cos(radLat2)*Math.pow(Math.sin(b/2),2)))
    s = s * 6378.137
    s = round(s * 10000) / 10
    return s

def gaussian_distribution(x, mean, variance):
    """ Gaussian Distibution
        Return gaussian distribution(`mean`, `variance`) probability given `x`
    """
    return np.exp(-(x-mean)**2 / (variance**2)) / (np.sqrt(2*np.pi)*variance)
def to_string(data):
    return map(lambda x: str(x), data)

def zip_name(name, data):
    return zip([name]*len(data), data)

def preprocess_(dataset):
    n, m = dataset.shape
    dataset = dataset.astype(np.float32)

    # feature standardization
    for i in range(m):
        mean = np.mean(dataset[:, i])
        std = np.std(dataset[:, i])
        dataset[:, i] = (dataset[:, i] - mean) / std if std else 0

    return dataset

def preprocess(tr_feature, te_feature=None):
    # feature size of train data and test data must be equal

    if te_feature is not None:
        assert(tr_feature.shape[1] == te_feature.shape[1])
        tr_size = tr_feature.shape[0]
        # te_size = te_feature.shape[0]

        # combine train data and test data
        dataset = np.vstack((tr_feature, te_feature))

        # preprocessing data
        dataset = preprocess_(dataset)

        # split dataset to train data and test data
        tr_feature = dataset[0:tr_size, :]
        te_feature = dataset[tr_size:, :]
        tr_feature = tr_feature.astype(np.float32)
        te_feature = te_feature.astype(np.float32)
        return tr_feature, te_feature
    else:
        # preprocessing data
        tr_feature = preprocess_(tr_feature)
        tr_feature = tr_feature.astype(np.float32)
        return tr_feature

def make_vocab(tr_ids, te_ids=None):
    if te_ids is not None:
        # check whether number of id features are same between train and test
        assert(len(tr_ids) == len(te_ids))
        dataset = tr_ids + te_ids
    else:
        dataset = tr_ids

    # Stack dataset itself
    dataset = reduce(lambda x,y: x+y, dataset)

    unique_ids = set(dataset)

    return len(set(dataset)), dict(zip(unique_ids, range(len(unique_ids))))

# ------------------------------------------------------

lu_dpath_4g = '../位置精度算法测试数据/LTE Mr/'
bu_dpath_4g = '../位置精度算法测试数据/LTE Mr/'
data_4g = {
    'backward1': lu_dpath_4g+'_pci_mr_路测数据导出4G反向-1_903373250.csv',
    'forward1': lu_dpath_4g+'_pci_mr_路测数据导出4G正向合并_903383343.csv',
    'backward2': lu_dpath_4g+'_pci_mr_路测数据导出4G反向-2_903368843.csv',
    'forward2': lu_dpath_4g+'_pci_mr_路测数据导出4G正向-2_903360656.csv',
    'backward3': lu_dpath_4g+'_pci_mr_路测数据导出4G反向-3_903358531.csv',
    'forward3': lu_dpath_4g+'_pci_mr_路测数据导出4G正向-3_903371421.csv',
    'forwardbackward4': lu_dpath_4g+'_pci_mr_路测数据导出4G正反向-4_995679437.csv',
    'forwardbackward5': lu_dpath_4g+'_pci_mr_路测数据导出4G正反向-5_996915109.csv',
    'forward0': lu_dpath_4g+'_pci_mr_路测数据导出4G正向-1_904161156.csv',
    'bu_1': bu_dpath_4g+'_pci_mr_步测数据导出4G-1_903368046.csv',
    'bu_2': bu_dpath_4g+'_pci_mr_步测数据导出4G-2_900298203.csv',
    'bu_3' : bu_dpath_4g+'_pci_mr_步测数据导出4G-3_903182171.csv'
}
dpath_4g_test = '../位置精度算法测试数据/Test/'
data_4g_test = [dpath_4g_test+'TXTmrinfo_610397875_0.csv', dpath_4g_test+'TXTmrinfo_610460765_0.csv']
lu_dpath = '../位置精度算法测试数据/位置精度算法测试-24G路测/'
bu_dpath = '../位置精度算法测试数据/位置精度算法测试-24G步测/'
data_2g = {
    'backward1': lu_dpath+'路测数据导出2G反向-1.csv',
    'forward1': lu_dpath+'路测数据导出2G正向-1.csv',
    'forwardbackward2': lu_dpath+'路测数据导出2G正反向-2.xlsx',
    'bu_1': bu_dpath+'步测数据导出2G-1.csv',
    'bu_2': bu_dpath+'步测数据导出2G-2.xlsx'
}
not_null_col_2g = ['All-LAC', 'All-Cell Id', 'All-Longitude', 'All-Latitude']
feature_dict_2g = {
    'Time': 'time',
    'All-LAC': 'lac',
    'All-Cell Id': 'ci',
    'All-BSIC (Num)': 'bsic',
    'All-ARFCN BCCH': 'bcch',
    'All-RxLev Sub (dBm)': 'rxlev',
    'All-Longitude': 'x',
    'All-Latitude': 'y',
    'All-Neighbor LAC (Sorted)[1]': 'lac1',
    'All-Neighbor LAC (Sorted)[2]': 'lac2',
    'All-Neighbor LAC (Sorted)[3]': 'lac3',
    'All-Neighbor LAC (Sorted)[4]': 'lac4',
    'All-Neighbor LAC (Sorted)[5]': 'lac5',
    'All-Neighbor LAC (Sorted)[6]': 'lac6',
    'All-Neighbor Cell Id (Sorted)[1]': 'ci1',
    'All-Neighbor Cell Id (Sorted)[2]': 'ci2',
    'All-Neighbor Cell Id (Sorted)[3]': 'ci3',
    'All-Neighbor Cell Id (Sorted)[4]': 'ci4',
    'All-Neighbor Cell Id (Sorted)[5]': 'ci5',
    'All-Neighbor Cell Id (Sorted)[6]': 'ci6',
    'All-Neighbor BSIC (Num) (Sorted)[1]': 'bsic1',
    'All-Neighbor BSIC (Num) (Sorted)[2]': 'bsic2',
    'All-Neighbor BSIC (Num) (Sorted)[3]': 'bsic3',
    'All-Neighbor BSIC (Num) (Sorted)[4]': 'bsic4',
    'All-Neighbor BSIC (Num) (Sorted)[5]': 'bsic5',
    'All-Neighbor BSIC (Num) (Sorted)[6]': 'bsic6',
    'All-Neighbor ARFCN (Sorted)[1]': 'bcch1',
    'All-Neighbor ARFCN (Sorted)[2]': 'bcch2',
    'All-Neighbor ARFCN (Sorted)[3]': 'bcch3',
    'All-Neighbor ARFCN (Sorted)[4]': 'bcch4',
    'All-Neighbor ARFCN (Sorted)[5]': 'bcch5',
    'All-Neighbor ARFCN (Sorted)[6]': 'bcch6',
    'All-Neighbor RxLev (dBm) (Sorted)[1]': 'rxlev1',
    'All-Neighbor RxLev (dBm) (Sorted)[2]': 'rxlev2',
    'All-Neighbor RxLev (dBm) (Sorted)[3]': 'rxlev3',
    'All-Neighbor RxLev (dBm) (Sorted)[4]': 'rxlev4',
    'All-Neighbor RxLev (dBm) (Sorted)[5]': 'rxlev5',
    'All-Neighbor RxLev (dBm) (Sorted)[6]': 'rxlev6',
}
col_name_2g = ['lac', 'ci'] + ['%s%d'%(col,i) for i in range(1,7) for col in ['lac','ci','bsic','bcch','rxlev']]
dpath_2g_test = '../位置精度算法测试数据/Test/2G mr原始文档/'
data_2g_test = [dpath_2g_test+'2G mr107_0513.csv', dpath_2g_test+'2G mr107_0516.csv', dpath_2g_test+'2G mr221_0513.csv', dpath_2g_test+'2G mr221_0516.csv']


def get_4g_engpara():
    eng_para = pd.read_csv('../工参/4G工参20160505.CSV', encoding='gbk')
    eng_para_lnglat = eng_para[['CGI',u'经度',u'纬度']]
    eng_para_lnglat = eng_para_lnglat[eng_para_lnglat['CGI'].notnull()]
    # 把CGI字段拆成两列，比如把460-00-107797-3拆成107797和3两列
    def split_cgi(cgi):
        cgi = cgi.split('-')
        enodebid = int(cgi[2])
        ci = int(cgi[3])
        return [enodebid, ci]

    # 加上两列enodebid, ci用来匹配数据
    engpara_enodebid_ci = np.asarray(map(split_cgi, eng_para_lnglat['CGI'].values))
    eng_para_lnglat['LAC'] = engpara_enodebid_ci[:, 0]
    eng_para_lnglat['CI'] = engpara_enodebid_ci[:, 1]
    eng_para_lnglat = eng_para_lnglat.drop(['CGI'], axis=1)
    eng_para_lnglat = eng_para_lnglat[eng_para_lnglat[u'经度'].notnull() & eng_para_lnglat['LAC'].notnull()]
    eng_para_lnglat = eng_para_lnglat.drop_duplicates()
    return eng_para_lnglat

def get_2g_engpara():
    eng_para = pd.read_csv('../工参/2G工参20160505.CSV', encoding='gbk')
    eng_para = eng_para[['LAC', 'CI', u'经度', u'纬度']]
    eng_para = eng_para[eng_para.LAC.notnull() & eng_para[u'经度'].notnull()]
    eng_para = eng_para.drop_duplicates()
    return eng_para

def get_4g_data(dnames):
    data = pd.DataFrame()
    for dname in dnames:
        data = pd.concat([data, pd.read_csv(data_4g[dname], sep='\t')])
    return data

def get_2g_data(dnames):
    data = pd.DataFrame()
    for dname in dnames:
        data = pd.concat([data, get_2g_data_(dname)])
    return data

def get_2g_test(idx):
    data = pd.read_csv(data_2g_test[idx], encoding='gbk')
    data = data.rename(columns={u'时间点': 'time', 'longitude': 'x', 'latitude': 'y', 'rxlev_dl_sub': 'rxlev'})
    data.ci = data.ci - (data.lac.values << 16)
    return data

def get_2g_data_(dname):
    #dname = 'backward1'
    data = pd.read_csv('%s' % (data_2g[dname])) if data_2g[dname].find('csv') >= 0 else pd.read_excel('%s' % (data_2g[dname]))
    for col in not_null_col_2g:
        data = data[data[col].notnull()]
    data = data.drop_duplicates()
    data = data.rename(columns=feature_dict_2g)[feature_dict_2g.values()]
    data = data.fillna(-999)
    data.lac = data.lac.astype(int)
    data.ci = data.ci.astype(int)
    for i in range(1, 7):
        data['lac%d'%i] = data['lac%d'%i].astype(int)
        data['ci%d'%i] = data['ci%d'%i].astype(int)
    return data

def report(error, info=''):
    mean = np.mean(error)
    median = np.median(error)
    error = sorted(error)
    p67 = error[int(len(error)*0.67)]
    p80 = error[int(len(error)*0.8)]
    p90 = error[int(len(error)*0.9)]
    if info != '':
        info += '\t'
    print '%s%.2f\t%.2f\t%.2f\t%.2f\t%.2f' % (info, mean,median,p67,p80,p90)

def make_rf_dataset(data, eng_para, is_test=True):
    # default_value = -999.
    # data = pd.merge(left=data,right=eng_para,left_on=['RNCID_1','CellID_1'],right_on=['LAC','CI'],how='left')
    data = data.merge(eng_para, left_on=['lac', 'ci'], right_on=['LAC','CI'], how='left')
    for i in range(1, 7):
        data = data.merge(eng_para, left_on=['lac%d' % i, 'ci%d' % i], right_on=['LAC','CI'], how='left', suffixes=('', '%d' % i))
        # data['RSSI_%d'%i] = data['RSCP_%d'%i]-data['EcNo_%d'%i]
        # data = data.drop(['LAC', 'CI'], axis=1)
    data = data.fillna(-999.)

    feature = data[col_name_2g+[u'经度',u'纬度',u'经度6',u'纬度6',u'经度2',u'纬度2',u'经度3',u'纬度3',u'经度4',u'纬度4',u'经度5',u'纬度5']]

    label = data[['x', 'y']]

    if not is_test:
        bts = feature[[u'经度', u'纬度']]
        keep_list = []
        for i, (pt1, pt2) in enumerate(zip(bts.values, label.values)):
            if distance(pt1, pt2) < 400:
                keep_list.append(i)
        feature = feature.iloc[keep_list, :]
        label = label.iloc[keep_list, :]
        # print len(keep_list)

    return data['time'], feature, label

def compute_bs_distance(pt, pt1, nei_pts):
    dist = []
    for nei_pt in nei_pts:
        if nei_pt[0] != -999. or nei_pt[1] != -999.:
            dist.append(distance(pt1, nei_pt))
    # return distance(pt, pt1)
    if len(dist) > 0:
        return np.mean(dist)
    else:
        return distance(pt, pt1)


def bs_distance(data, label):
    return [compute_bs_distance(pt, pt1, [pt2, pt3, pt4, pt5, pt6]) for pt, pt1, pt2, pt3, pt4, pt5, pt6 in \
            zip(label, data[[u'经度', u'纬度']].values, data[[u'经度2', u'纬度2']].values, data[[u'经度3', u'纬度3']].values, \
                data[[u'经度4', u'纬度4']].values, data[[u'经度5', u'纬度5']].values, data[[u'经度6', u'纬度6']].values)]

def cellsense_transform(df):
    data = []
    for idx, row in df.iterrows():
        id_list = []
        st_list = []
        for i in range(1, 7):
            lac = row['RNCID_%d'%(i)]
            ci = row['CellID_%d'%(i)]
            ecno = row['EcNo_%d'%(i)]
            rscp = row['RSCP_%d'%(i)]
            try:
                if int(lac) == -999 or int(ci) == -999 or int(ecno) == -999 or int(rscp) == -999:
                    continue
                id_list.append('%d:%d'%(int(lac),int(ci)))
                st_list.append(rscp-ecno)
            except Exception, _data:
                print _data, lac, ci, idx
                exit()
        data.append(FingerPrint(id_list, st_list, (row['Longitude'], row['Latitude'])))
    return data

def mlp_feature(data, eng_para, is_train, col_name):
    data = data[data.Longitude.notnull() & data.Latitude.notnull()]
    label = data[['Longitude', 'Latitude']]
    data = data[col_name]
    for i in range(1, 7):
        data = data.merge(eng_para, left_on=['RNCID_%d' % i, 'CellID_%d' % i], right_on=['LAC','CI'], how='left', suffixes=('', '_%d' % i))
        data = data.drop(['LAC', 'CI'], axis=1)
            # data = data.drop(['enodebid_%d' % i, 'ci_%d' % i], axis=1)
    data = data.fillna(-999)

    lacci_vals = []
    for nei_id in range(1, 7):
        lacci_vals.append(map(lambda x: '%.0f,%.0f' % (x[0], x[1]),
                    data[['RNCID_%d'%nei_id, 'CellID_%d'%nei_id]].values))
        data = data.drop(['RNCID_%d'%nei_id, 'CellID_%d'%nei_id], axis=1)

    return data.values, label.values, [lacci_vals]

def get_sparse_matrix(feature, rncci_dict):
    n_row = len(feature)
    n_col = len(rncci_dict)
    data = []
    row = []
    col = []
    for i in xrange(1, 7):
        rncci = map(lambda x: int(x[0])*100+int(x[1]), zip(feature['RNCID_%d'%i].values, feature['CellID_%d'%i].values))
        rscp = feature['RSCP_%d'%i].values + 141
        ecno = feature['EcNo_%d'%i].values + 31
        for j, (rncci_, rscp_, ecno_) in enumerate(zip(rncci, rscp, ecno)):
            if rncci_dict.has_key(rncci_) and rscp_ >= 0 and ecno_ >= 0:
                # BS
                data.append(1)
                row.append(j)
                col.append(rncci_dict[rncci_])
                # RSCP
                data.append(rscp_)
                row.append(j)
                col.append(len(rncci_dict)+rncci_dict[rncci_])
                # EcNo
                data.append(ecno_)
                row.append(j)
                col.append(2*len(rncci_dict)+rncci_dict[rncci_])

    return csc_matrix((data, (row, col)), shape=(n_row, 3*n_col))

def get_all_rncci(fnames):
    data = get_4g_data(fnames)
    rnccis = set()
    for i in xrange(1, 7):
        s = filter(lambda x: x>0, map(lambda x: int(x[0])*100+int(x[1]), zip(data['RNCID_%d'%i].values, data['CellID_%d'%i].values)))
        rnccis |= set(s)
    return dict(zip(rnccis, range(len(rnccis)))), list(rnccis)
rncci_dict, rncci_list = get_all_rncci(['forward0', 'forward1', 'forward2', 'forward3', 'bu_1', 'bu_2' ,'bu_3', 'backward1', 'backward2', 'backward3'])
