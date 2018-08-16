#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 19:19:46 2018

@author: maxiao
"""
import pdb
import numpy as np
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd, gluon, init, autograd
from mxnet.contrib import text
VERY_BIG_NUMBER = 1e20
VERY_SMALL_NUMBER = 1e-20
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

def exp_mask_for_tensor(val, val_mask):
    val_mask = val_mask.expand_dims(-1).astype('float32')
    return val + (1.0 - val_mask) * VERY_NEGATIVE_NUMBER

def mask_for_tensor(val, val_mask):
    val_mask = nd.expand_dims(val_mask, -1)
    return val * val_mask.astype('float32')


class MultiDimensionalAttention(nn.Block):
    def __init__(self, config, axis = 1):
        self.axis = axis
        super(MultiDimensionalAttention, self).__init__()
        with self.name_scope():
            self.linear1 = nn.Dense(in_units = config.hidden_dim,
                                    units = config.hidden_dim,
                                    activation = config.activation,
                                    flatten = False)
            self.linear2 = nn.Dense(in_units = config.hidden_dim,
                                    units = config.hidden_dim,
                                    activation = None,
                                    flatten = False)

    def forward(self, x, _mask = None):
        map1 = self.linear1(x)
        map2 = self.linear2(map1)
        #map2 = exp_mask_for_tensor(map2, x_mask)
        soft = nd.softmax(map2, axis = self.axis)
        out = (soft * x).sum(axis = self.axis)
        return out


def scaled_tanh(x, scale = 5.0):
    return scale * nd.tanh(1.0 / scale * x)

def get_direct_mask(n, t, direction = 'forward'):
    x, y = np.meshgrid(range(t), range(t))
    if direction == 'forward':
        mask = np.greater(x, y)
    else:
        mask = np.greater(y, x)
    out = np.tile(np.expand_dims(mask, axis = 0), (n, 1, 1))
    return nd.array(out.astype(np.float32))


class DiSelfAttention(nn.Block):
    def __init__(self, config, direction):
        super(DiSelfAttention, self).__init__()
        self.config = config
        self.direction = direction
        #self.f_bias = nd.zeros([config.hidden_dim])
        #self.o_bias = nd.zeros([config.hidden_dim])
        with self.name_scope():
            self.linear1 = nn.Dense(in_units = config.embedding_dim,
                                    units = config.hidden_dim,
                                    activation = config.activation,
                                    flatten = False)
            self.linear2 = nn.Dense(in_units = config.hidden_dim,
                                    units = config.hidden_dim,
                                    use_bias = False,
                                    flatten = False,
                                    activation = None)
            self.linear3 = nn.Dense(in_units = config.hidden_dim,
                                    units = config.hidden_dim,
                                    use_bias = False,
                                    flatten = False,
                                    activation = None)
            self.linear4 = nn.Dense(in_units = config.hidden_dim,
                                    units = config.hidden_dim,
                                    use_bias = False,
                                    flatten = False,
                                    activation = None)
            self.linear5 = nn.Dense(in_units = config.hidden_dim,
                                    units = config.hidden_dim,
                                    use_bias = False,
                                    flatten = False,
                                    activation = None)
            self.dropout = nn.Dropout(rate = config.dropout_rate)
            self.f_bias = nd.zeros([config.hidden_dim])
            self.f_bias.attach_grad()
            self.o_bias = nd.zeros([config.hidden_dim])
            self.o_bias.attach_grad()
    def forward(self, x, x_mask = None):
        N, T, D = tuple(x.shape) # bs, sl, vec
        bs, sl, vec = tuple(x.shape)
        direct_mask = get_direct_mask(bs, sl, self.direction)
        #x_mask_tile = x_mask.expand_dims(1)
        #mask = np.logical_and(direct_mask, x_mask_tile).astype(float)
        mask = direct_mask.astype('float32')
        x_map = self.linear1(x) # bs, sl, vec
        #x_map_tile = x_map.expand_dims(1) #
        x_map_tile = nd.tile(x_map.expand_dims(1), (1, sl, 1, 1)) # bs, sl, sl, vec
        x_map_drop = self.dropout(x_map)

        dependent = self.linear2(x_map_drop)
        dependent_etd = dependent.expand_dims(1)
        head = self.linear3(x_map_drop)
        head_etd = head.expand_dims(2)
        loggits = scaled_tanh(dependent_etd + head_etd + self.f_bias, 5.0)

        loggits_masked = exp_mask_for_tensor(loggits, mask)
        attn_score = nd.softmax(loggits_masked, 2)
        attn_score = mask_for_tensor(attn_score, mask)

        attn_result = (attn_score * x_map_tile).nansum(2)
        fusion_gate = nd.sigmoid(self.linear4(x_map) + self.linear5(attn_result) + self.o_bias)
        output = fusion_gate * x_map + (1 - fusion_gate) * attn_result
        return output

class SelfAttentionNet(nn.Block):
    def __init__(self, config):
        super(SelfAttentionNet, self).__init__()
        with self.name_scope():
            self.embedding = nn.Embedding(input_dim = config.vocab_size,
                                          output_dim = config.embedding_dim)
            self.di_fw = DiSelfAttention(config, direction = 'forward')
            self.di_bw = DiSelfAttention(config, direction = 'backward')
            self.multi_self_attention = MultiDimensionalAttention(config)
            self.output = nn.Dense(in_units = 2 * config.hidden_dim,
                                   units = config.num_classes,
                                   activation = None,
                                   )
    def forward(self, x, x_mask = None):
        x = self.embedding(x)
        fw_attn = self.di_fw(x, x_mask)
        bw_attn = self.di_bw(x, x_mask)
        attn = nd.concat(fw_attn, bw_attn, dim=2)
        out = self.multi_self_attention(attn, x_mask)
        out = self.output(out)
        #out = nd.softmax(out, axis = -1)
        return out






if __name__ == '__main__':
    from utils import DictClass
    config = {'hidden_dim':3,
              'activation':'relu',
              'dropout_rate':0.5,
              'vocab_size': 10,
              'embedding_dim': 300,
              'num_classes': 2,
              'feature_map':64,
              'kernel_size':[5],
              }
    config = DictClass(config)
    net = SelfAttentionNet(config)
    from model import TextCNN
    net = TextCNN(config)
    # 4 words, embedding_dim = 3
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
