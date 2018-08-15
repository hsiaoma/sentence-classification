#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 15:38:50 2018

@author: maxiao
"""
import ujson, sys, os, logging, pdb
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)-15s %(message)s')
import numpy as np
from sklearn import metrics

import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd, gluon, init, autograd
from mxnet.contrib import text


class Convpool(nn.Block):
    def __init__(self, channels, kernel_size, **kwargs):
        super(Convpool, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv1D(channels, kernel_size, strides = 1)
            self.pooling = nn.GlobalMaxPool1D()
    def forward(self, x):
        x = self.conv(x)
        x = nd.relu(x)
        return self.pooling(x).flatten()

class TextCNN(nn.Block):
    def __init__(self, config, **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        with self.name_scope():
            self.embedding = nn.Embedding(input_dim = config['vocab_size'],
                                          output_dim = config['embedding_dim']
                                          )
            self.conv1 = Convpool(config['feature_map'],
                                  config['kernel_size'][0])
            #self.conv2 = Convpool(config['num_feature_map'],
            #                      config['kernel_size'][1])
            #self.conv3 = Convpool(config['num_feature_map'],
            #                      config['kernel_size'][2])
            self.dropout = nn.Dropout(config['dropout_rate'])
            self.fc = nn.Dense(2)
    def forward(self, x):
        x = self.embedding(x).transpose((0, 2, 1))
        #o1, o2, o3 = self.conv1(x), self.conv2(x), self.conv3(x)
        #outputs = self.fc(self.dropout(nd.concat(o1, o2, o3)))

        o = self.conv1(x)
        outputs = self.fc(self.dropout(o))

        return outputs

def SequentialTextCNN(config):

    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Embedding(input_dim = config['vocab_size'],
                             output_dim = config['embedding_dim']))
        net.add(nn.Lambda(lambda x: x.transpose((0, 2, 1))))
        net.add(nn.Conv1D(channels = config['feature_map'],
                          kernel_size = config['kernel_size'][0],
                          strides = 1))
        net.add(nn.BatchNorm(axis=1))
        net.add(nn.Activation('relu'))
        net.add(nn.GlobalMaxPool1D())
        net.add(nn.Dropout(rate = config['dropout_rate']))
        net.add(nn.Dense(units = 2))
    return net


class ParaConvpool(nn.Block):
    def __init__(self, config, kernel_idx = 0, **kwargs):
        super(ParaConvpool, self).__init__(**kwargs)
        with self.name_scope():
            self.conv = nn.Conv2D(channels = config['feature_map'] * config['num_neighbor'],
                                  kernel_size = (config['kernel_size'][kernel_idx], config['embedding_dim']),
                                  strides = (1, config['embedding_dim']),
                                  groups = config['num_neighbor'])
            self.pooling = nn.GlobalMaxPool1D()
            self.norm = nn.BatchNorm(axis = 1)
    def forward(self, x):
        x = self.conv(x)
        x = nd.squeeze(x, axis = -1)
        x = self.norm(x)
        x = nd.relu(x)
        x = self.pooling(x).flatten()
        return x

class Sumpool(nn.Block):
    def __init__(self, config, **kwargs):
        super(Sumpool, self).__init__(**kwargs)
        self.config = config
        with self.name_scope():
            self.conv = nn.Conv1D(channels = config['num_neighbor'] * int(config['feature_map'] / 2),
                                   kernel_size = config['feature_map'] * len(self.config['kernel_size']),
                                   strides = config['feature_map'],
                                   groups = config['num_neighbor'],
                                   activation = 'relu')
    def forward(self, x):
        x = x.reshape((-1,
                       self.config['num_neighbor'],
                       self.config['feature_map'] * len(self.config['kernel_size'])))
        x = self.conv(x)
        return x.squeeze(axis = -1)

class ParaTextCNN(nn.Block):
    def __init__(self, config, **kwargs):
        super(ParaTextCNN, self).__init__(**kwargs)
        self.config = config
        with self.name_scope():
            self.embedding = nn.Embedding(input_dim = config['vocab_size'],
                                          output_dim = config['embedding_dim'])
            self.conv1 = ParaConvpool(config, kernel_idx = 0)
            self.conv2 = ParaConvpool(config, kernel_idx = 1)
            self.conv3 = ParaConvpool(config, kernel_idx = 2)
            self.sumpool = Sumpool(config)
            self.output = nn.Dense(units = config['num_classes'])
            self.dropout = nn.Dropout(config['dropout_rate'])

            self.conv2 = ParaConvpool(config, kernel_idx = 1)
            self.conv3 = ParaConvpool(config, kernel_idx = 2)


    def forward(self, x):
        sent = self.embedding(x)
        x = sent.reshape((-1,
                          self.config['num_neighbor'],
                          self.config['title_len'],
                          self.config['embedding_dim'])) # bs, ch, tl, ed
        #x = self.convpool(x)

        o1 = self.conv1(x)
        o2 = self.conv2(x)
        o3 = self.conv3(x)
        x = self.dropout(nd.concat(o1, o2, o3))

        x = self.sumpool(x)
        x = self.dropout(x)

        return self.output(x)

class Source2TokenAttention(nn.Block):
    def __init__(self, config):
        super(Source2TokenAttention, self).__init__()
        with self.name_scope():
            self.embedding = nn.Embedding(input_dim = config.vocab_size,
                                          output_dim = config.embedding_dim)
            self.map = nn.Dense(in_units = config.embedding_dim,
                                units = config.hidden_dim,
                                activation = config.activation,
                                flatten = False)
            self.linear1 = nn.Dense(in_units = config.hidden_dim,
                                    units = config.hidden_dim,
                                    activation = config.activation,
                                    flatten = False)
            self.linear2 = nn.Dense(in_units = config.hidden_dim,
                                    units = config.hidden_dim,
                                    activation = None,
                                    flatten = False)
            self.output = nn.Dense(in_units = config.hidden_dim,
                               units = config.num_classes,
                               activation = None,
                               )
    def forward(self, x, x_mask = None):
        x = self.embedding(x)
        x = self.map(x)
        map1 = self.linear1(x)
        map2 = self.linear2(map1)
        #map2 = exp_mask_for_tensor(map2, x_mask)
        soft = nd.softmax(map2, axis = 1)
        out = (soft * x).sum(1)
        return self.output(out)


if __name__ == '__main__':
    from train import arg_parse
    config = arg_parse()
    config['embedding_dim'] = 300
    config['vocab_size'] = 10
    net = ParaTextCNN(config)

    data = nd.array([[np.random.randint(10) for _ in range(200)] for _ in range(1000)])
    label = nd.array([np.random.randint(2) for _ in range(1000)])
    dataset_train = gluon.data.ArrayDataset(data, label)
    train_data = gluon.data.DataLoader(dataset_train, batch_size=50, shuffle=True, last_batch='rollover')
    embedding = text.embedding.CustomEmbedding('embedding_files/dummy.embedding', elem_delim = ' ')

    net.collect_params().initialize(init.Xavier(), ctx = mx.cpu())
    net.embedding.weight.set_data(embedding.idx_to_vec)
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    for data, label in train_data:
        with autograd.record():
            out = net(data)
            loss = softmax_cross_entropy(out, label)
            loss.backward()
        print(loss.sum().asscalar())

