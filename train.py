#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:06:41 2018

@author: maxiao
"""

import argparse, msgpack, logging, os, pdb
from utils import DictClass
from mxnet.contrib import text
from model import TextCNN, SequentialTextCNN, ParaTextCNN, Source2TokenAttention
from layers import SelfAttentionNet

import numpy as np
from sklearn import metrics

import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd, gluon, init, autograd
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def arg_parse():
    parser = argparse.ArgumentParser(
            description = 'Convlutional nerual net for sentence classification.'
    )

    parser.add_argument('--data_file', default = '20180805/data.msgpack')
    parser.add_argument('--meta_file', default = '20180805/meta.msgpack')
    parser.add_argument('--log_file', default = 'log/training_test.log')
    parser.add_argument('--best_in_valid_model_name', default = 'best_in_valid.param')
    parser.add_argument('--resume', action = 'store_true')
    parser.add_argument('--saved_model_name', default = None)
    # training
    parser.add_argument('--max_epoch', type = int, default = 10)
    parser.add_argument('--log_per_batch', type = int, default = 5)
    parser.add_argument('--batch_size', type = int, default = 50)
    parser.add_argument('--learning_rate', type = float, default = 5e-4)
    parser.add_argument('--no_update_embedding', action='store_true')

    # model
    parser.add_argument('--dropout_rate', type = float, default = 0.5)
    parser.add_argument('--kernel_size', nargs = '+', type = int, default = [3])
    parser.add_argument('--title_len', type = int, default = 20)
    parser.add_argument('--num_neighbor', type = int, default = 10)
    parser.add_argument('--feature_map', type = int, default = 64)
    parser.add_argument('--hidden_dim', type = int, default = 128)
    parser.add_argument('--activation', default = 'relu')
    parser.add_argument('--num_classes', type = int, default = 2)


    args = parser.parse_args()
    return DictClass(vars(args))

def get_log(args):
    logging.basicConfig(level=logging.INFO, format = '%(asctime)s %(message)s')
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt = '%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)
    return log

def load_data(args):
    with open('{:s}'.format(args.meta_file), 'rb') as f:
        meta = msgpack.load(f, encoding = 'utf-8', use_list = True)
    args['embedding_dim'] = meta['embedding_dim']
    args['vocab_size'] = meta['vocab_size']
    args['embedding_file'] = meta['embedding_file']
    with open('{:s}'.format(args.data_file), 'rb') as f:
        data = msgpack.load(f, encoding = 'utf-8', use_list = True)

    ret = []
    for target in ['train', 'valid', 'test']:

        ret.append([sent[:] for sent in nd.array(data[target]['data'])])
        ret.append(nd.array(data[target]['label']))
        #for content in ['data', 'label']:
        #    ret.append(nd.array(data[target][content]))
    ret.append(nd.array(meta['embedding']))
    return ret

def test(net, test_data, is_training = False):
    ctx = mx.cpu()
    total_right = 0
    total_loss = 0.
    total_size = 0.

    all_label = []
    all_pred_score = []

    for data, label in test_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        if is_training:
            if (np.random.rand() < 0.9): continue
        out = net(data)
        loss = softmax_cross_entropy(out, label)
        total_loss += loss.sum().asscalar()
        predictions = np.argmax(out.asnumpy(), axis=1)
        right = np.sum(predictions==label.asnumpy())
        total_right += right
        total_size += data.shape[0]
        all_pred_score.extend(nd.softmax(out).asnumpy()[:, 1].tolist())
        all_label.extend(label.asnumpy().tolist())
    auc = metrics.roc_auc_score((np.array(all_label) == 1).astype(int), all_pred_score)
    mean_loss = float(total_loss) / total_size
    acc = float(total_right) / total_size
    return auc, mean_loss, acc


def train(net, train_data, valid_data, args, log):
    ctx = mx.cpu()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': args['learning_rate']})
    best_score = 0.
    for i in range(args.max_epoch):
        train_loss = 0.
        batch_id = 0
        for data, label in train_data:
            batch_id += 1
            # forward + backward
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
                loss.backward()
            # update parameters
            trainer.step(args['batch_size'])
            train_loss += loss.sum().asscalar()
            if batch_id % args['log_per_batch'] == 0:
                log.info('%d samples processed, batch loss: %s'
                         % (batch_id * args['batch_size'], loss.sum().asscalar()))
        log.info ('epoch %s done, epoch mean loss: %s' % (i, train_loss / len(train_data._dataset)))
        auc, mean_loss, acc = test(net, train_data, is_training = True)
        log.info ("train auc = {:.5f}, mean_loss = {:.5f}, acc = {:.5f}".format(auc, mean_loss, acc))
        auc, mean_loss, acc = test(net, valid_data)
        log.info ("validation auc = {:.5f}, mean_loss = {:.5f}, acc = {:.5f}".format(auc, mean_loss, acc))
        if auc > best_score:
            net.save_parameters(args.best_in_valid_model_name)
            best_score = auc

def run(args, log):
    train_x, train_y, valid_x, valid_y, test_x, test_y, embedding = load_data(args)
    log.info(vars(args))

    #embedding = text.embedding.CustomEmbedding(args.embedding_file, elem_delim = ' ')
    #embedding.update_token_vectors('<unk>', nd.uniform(low = -0.05, high = 0.05, shape = 300))

    net = Source2TokenAttention(args)
    if args.resume and os.path.exists(args.saved_model_name):
        net.load_parameters(args.saved_model_name, ctx = mx.cpu())
        log.info("model {:s} loaded".format(args.saved_model_name))
        log.info(net)
    else:
        net.hybridize()
        net.collect_params().initialize(init.Xavier(), ctx = mx.cpu())
        log.info(net)
        if isinstance(net, nn.Sequential):
            net[0].weight.set_data(embedding)
            if args.no_update_embedding:
                net[0].weight.grad_req = 'null'
        else:
            net.embedding.weight.set_data(embedding)
            if True or args.no_update_embedding:
                net.embedding.weight.grad_req = 'null'

    dataset_train = gluon.data.ArrayDataset(train_x, train_y)
    dataset_valid = gluon.data.ArrayDataset(valid_x, valid_y)
    dataset_test = gluon.data.ArrayDataset(test_x, test_y)

    train_data = gluon.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, last_batch='rollover')
    valid_data = gluon.data.DataLoader(dataset_valid, batch_size=args.batch_size, last_batch = 'rollover')
    test_data = gluon.data.DataLoader(dataset_test, batch_size=args.batch_size, last_batch = 'rollover')

    train(net, train_data, valid_data, args, log)
    log.info('training completed.')
    auc, mean_loss, acc = test(net, test_data)
    log.info ("test auc = {:.5f}, mean_loss = {:.5f}, acc = {:.5f}".format(auc, mean_loss, acc))

    best_net = Source2TokenAttention(args)
    best_net.load_parameters(args.best_in_valid_model_name, ctx = mx.cpu())
    auc, mean_loss, acc = test(best_net, test_data)
    log.info ("best net auc = {:.5f}, mean_loss = {:.5f}, acc = {:.5f}".format(auc, mean_loss, acc))


def main():
    args = arg_parse()
    log = get_log(args)
    run(args, log)


if __name__ == '__main__':
    main()







