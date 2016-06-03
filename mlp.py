# coding: utf-8
import cPickle as pickle

import numpy as np
from grid import RoadGrid
from keras.models import Graph
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.embeddings import Embedding

from utils import preprocess, make_vocab, distance, to_string

def gen_report(true_pts, pred_pts, pickle_name, params):
    tot_error = []
    for true_pt, pred_pt in zip(true_pts, pred_pts):
        tot_error.append(distance(pred_pt, true_pt))
    f_report = open('report.txt', 'a')
    report_content = to_string(params)
#    f_report.write(pickle_name + '\n')
#    f_report.write('Total Test size\t%d\n' % len(tot_error))
    report_content.append(str(len(tot_error)))
    tot_error = sorted(tot_error)
    report_content += to_string([np.max(tot_error), \
                       np.min(tot_error), \
                       np.mean(tot_error), \
                       np.median(tot_error), \
                       tot_error[int(len(tot_error)*0.67)], \
                       tot_error[int(len(tot_error)*0.8)], \
                       tot_error[int(len(tot_error)*0.9)] \
                    ])
    f_report.write(','.join(report_content) + '\n')
    f_report.flush()
#    f_report.write('Total Max error\t%f\n' % np.max(tot_error))
#    f_report.write('Total Min error\t%f\n' % np.min(tot_error))
#    f_report.write('Total Mean error\t%f\n' % np.mean(tot_error))
#    f_report.write('Total Median error\t%f\n' % np.median(tot_error))
#    f_report.write('Total 67%% error\t%f\n' % tot_error[int(len(tot_error) * 0.67)])
#    f_report.write('Total 80%% error\t%f\n' % tot_error[int(len(tot_error) * 0.8)])
#    f_report.write('Total 90%% error\t%f\n\n' % tot_error[int(len(tot_error) * 0.9)])

def build_mlp(n_con, n_dis, dis_dims, vocab_sizes, n_grid, hidden_size=800):
    emb_size=10
    assert(n_dis == len(dis_dims) == len(vocab_sizes))
    # Define a graph
    network = Graph()

    # Input Layer
    input_layers = []
    network.add_input(name='con_input', input_shape=(n_con,))
    input_layers.append('con_input')
    ## embedding inputs
    for i in range(n_dis):
        network.add_input(name='emb_input%d' % i, input_shape=(dis_dims[i],), dtype=int)
        network.add_node(Embedding(input_dim=vocab_sizes[i], output_dim=emb_size, input_length=dis_dims[i]), name='emb%d' % i, input='emb_input%d' % i)
        network.add_node(Flatten(), name='fla_emb%d' % i, input='emb%d' % i)
        input_layers.append('fla_emb%d' % i)

    # Hidden Layer
    network.add_node(layer=Dense(hidden_size), name='hidden1', inputs=input_layers, merge_mode='concat')
    network.add_node(Activation('tanh'), name='hidden1_act', input='hidden1')

    # Ouput Layer
    network.add_node(Dense(n_grid), name='hidden2', input='hidden1_act')
    network.add_node(Activation('softmax'), name='hidden2_act', input='hidden2')
    network.add_output(name='output', input='hidden2_act')

    return network

def mlp(tr_data, te_data, eng_para, col_name, grid_size, \
        optimizer, batch_size, hidden_size, mlp_feature, \
        nb_epoch, prediction, model_name, is_train):
    # Load the dataset
    print 'Loading dataset ...'
    tr_feature, tr_label, tr_ids = mlp_feature(tr_data, eng_para, True, col_name)
    te_feature, te_label, te_ids = mlp_feature(te_data, eng_para, True, col_name)
    rg = RoadGrid(np.vstack((tr_label, te_label)), grid_size)
    tr_label = rg.transform(tr_label)
    # te_label = rg.transform(te_label)

    ## !!! maybe here need to ensure train data are the same shape as test data
    train_size, n_con = tr_feature.shape
    test_size, n_con = te_feature.shape
    n_dis = len(tr_ids)

    # Create neural network model
    print 'Preprocessing data ...'
    # Standardize continous input
    # tr_feature, te_feature = preprocess(tr_feature, te_feature)
    tr_feature, te_feature = preprocess(tr_feature, te_feature)
    # te_feature = preprocess(te_feature)
    tr_input = {'con_input' : tr_feature, 'output' : tr_label}
    te_input = {'con_input' : te_feature}
    # Prepare embedding input
    dis_dims, vocab_sizes = [], []
    for ii, tr_ids_, te_ids_ in zip(range(n_dis), tr_ids, te_ids): # make sure tr_ids contain several different discrete features
        vocab_size, vocab_dict = make_vocab(tr_ids_, te_ids_)
        tr_id_idx_, te_id_idx_ = [], []
        dis_dim = len(tr_ids_)
        for i in range(dis_dim):
            tr_id_idx_ += map(lambda x: vocab_dict[x], tr_ids_[i])
            te_id_idx_ += map(lambda x: vocab_dict[x], te_ids_[i])
        tr_ids = np.array(tr_id_idx_, dtype=np.int32).reshape(dis_dim, train_size).transpose()
        te_ids = np.array(te_id_idx_, dtype=np.int32).reshape(dis_dim, test_size).transpose()

        ## Add discrete feature to dict
        tr_input['emb_input%d' % ii] = tr_ids
        te_input['emb_input%d' % ii] = te_ids

        dis_dims.append(dis_dim)
        vocab_sizes.append(vocab_size)

    print 'Building model and compiling functions ...'
    # Define network structure
    grid_info = rg.grid_center
    network = build_mlp(n_con, n_dis, dis_dims, vocab_sizes, len(grid_info), hidden_size)

#network.compile(loss={'output': 'categorical_crossentropy'}, optimizer=SGD(lr=1e-2, momentum=0.9, nesterov=True))
    network.compile(loss={'output': 'categorical_crossentropy'}, optimizer=optimizer)

    # Build network
    # pickle_name = 'MLP-softmax-0.4.pickle'
    pickle_name = model_name

    if is_train:
        history = network.fit(tr_input, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
        # Dump Network
        with open('model/'+pickle_name, 'wb') as f:
           pickle.dump(network, f, -1)
    else:
        # Load Network
        f = open('model/'+pickle_name)
        network = pickle.load(f)

    # Make prediction
    ## 1. weighted
    if prediction == 'weighted':
        te_pred = np.asarray(network.predict(te_input)['output'])
        te_pred = te_pred.dot(grid_info)
    # Generate report
    # gen_report(te_label, te_pred, pickle_name, [type(optimizer), batch_size, hidden_size, 'Weighted'])
    elif prediction == 'argmax':
    ## 2. argmax
        te_pred = np.asarray(network.predict(te_input)['output'])
        te_pred = np.argmax(te_pred, axis=1)
        te_pred = [grid_info[idx] for idx in te_pred]
    # Generate report
    # gen_report(te_label, te_pred, pickle_name, [type(optimizer), batch_size, hidden_size, 'Argmax'])
    else:
        te_pred = None
    return te_pred

    # f_out = open('pred.csv', 'w')
    # for pred_pt, true_pt in zip(te_pred, te_label):
        # f_out.write('%f,%f,%f,%f\n' % (pred_pt[0], pred_pt[1], true_pt[0], true_pt[1]))

