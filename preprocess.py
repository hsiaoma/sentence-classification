#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:06:41 2018

@author: maxiao
"""

import argparse, ujson, msgpack, logging, multiprocessing, pdb
import numpy as np
from mxnet import nd
from mxnet.contrib import text
from utils import DictClass
from text_util import TextUtil

def arg_parse():
    parser = argparse.ArgumentParser(
            description = 'Preprocessing text files.'
    )

    parser.add_argument('--data_folder', default = '20180805/', help = 'directory of data folder')
    parser.add_argument('--data_out_file', default = 'data.msgpack')
    parser.add_argument('--meta_out_file', default = 'meta.msgpack')
    parser.add_argument('--is_test', type = int, default = 0)
    parser.add_argument('--log_file', default = 'log/pre_process.log')
    # mode
    parser.add_argument('--mode', default = 'concat_and_pad')

    # specification
    parser.add_argument('--sent_len', type = int, default = 20)
    parser.add_argument('--list_len', type = int, default = 10)
    parser.add_argument('--embedding_file', type = str, default = 'embedding_file/sgns.sogounews.bigram-char')
    parser.add_argument('--words_file', type = str, default = 'embedding_files/words.txt')
    parser.add_argument('--num_workers', type = int, default = 4)

    args = parser.parse_args()
    return DictClass(vars(args))

def parse_one_line(line):
    tmps = line.rstrip('\n').split('\t')
    timestamp = int(tmps[1])
    label = int(tmps[2])
    features = ujson.loads(tmps[3]).get("sim_titles", [])[:200]
    sent = [tup[1] for tup in features]
    if sum(sent) < 1e-3: return ()
    return (timestamp, label, sent)


def load_encoded_csv(path, num_workers):
    #data, label, timestamp = [], [], []
    lines = []
    with open(path, 'r') as f:
        for line in f:
            lines.append(line)
            #if len(lines) == 1000: break
        #lines = [line for line in f]
    pool = multiprocessing.Pool(processes = num_workers)
    data = pool.map(parse_one_line, lines)
    pool.close()
    pool.join()
    data = [i for i in data if len(i) > 0]
    data = sorted(data, key = lambda x : x[0])
    return data



def load_raw_text(path, is_test = False):
    data, label = [], []
    with open(path, 'r') as train_f:
        for line in train_f:
            tmps = line.rstrip('\n').split('\t')
            data.append(ujson.loads(tmps[1]))
            label.append(int(tmps[0]))
            if is_test and len(data) == 100:
                break
    return data, label

def onehot_enc(raw_data, embedding, sent_len, list_len, mode, is_test = False):
    ret = []
    if mode == 'concat_and_pat':
        for sent_list in raw_data:
            line = []
            for sent in sent_list[:list_len]:
                d = embedding.to_indices(sent)
                line.extend(d)
            if len(line) < sent_len:
                line.extend([0 for _ in range(sent_len - len(line))])
            else:
                line = line[ : sent_len]
            ret.append(line)
    elif mode == 'pad_and_concat':
        for sent_list in raw_data:
            line = []
            for sent in sent_list[:list_len]:
                d = embedding.to_indices(sent)
                if len(d) < sent_len:
                    d.extend([0 for _ in range(sent_len - len(d))])
                else:
                    d = d[:sent_len]
                line.extend(d)
            if len(line) < sent_len * list_len:
                line.extend([0 for _ in range(sent_len * list_len - len(line))])
            ret.append(line)
    return ret

def main():
    args = arg_parse()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    log = logging.getLogger(__name__)
    log.info(vars(args))
    #embedding = text.embedding.CustomEmbedding(args.embedding_file, elem_delim = ' ')
    #embedding.update_token_vectors('<unk>', nd.uniform(low = -0.05, high = 0.05, shape = 300))

    embedding = TextUtil(args.words_file, args.embedding_file)
    log.info('embedding file loaded, # of words = {:d}, dimension = {:d}'.format(len(embedding), embedding.vec_len))

#    data = {}
#    for target in ['train', 'valid', 'test']:
#        raw_data, label = load_raw_text(args.data_folder + '{:s}.txt'.format(target), is_test = args.is_test == 1)
#        enc = onehot_enc(raw_data, embedding, args.sent_len, args.list_len, args.mode)
#        data[target] = {}
#        data[target]['data'] = enc
#        data[target]['label'] = label
#        log.info('# of {:s} examples = {:d}'.format(target, len(enc)))
    data = {}
    data_tuple = load_encoded_csv('{:s}/raw_data.csv'.format(args.data_folder), args.num_workers)
    targets = ['train', 'valid', 'test']
    interval = [(0, 0.8), (0.8, 0.9), (0.9, 1)]
    for i in range(len(targets)):
        target = targets[i]
        start = int(interval[i][0] * len(data_tuple))
        end = int(interval[i][1] * len(data_tuple))
        data[target] = {}
        data[target]['label'] = [tup[1] for tup in data_tuple[start : end]]
        data[target]['data'] = [tup[2] for tup in data_tuple[start : end]]
        log.info('# of {:s} examples = {:d}'.format(target, len(data[target]['label'])))


    with open('{:s}/{:s}'.format(args.data_folder, args.data_out_file), 'wb') as f:
        msgpack.dump(data, f)

    meta = {
        'embedding_file': args.embedding_file,
        'embedding_dim': embedding.vec_len,
        'vocab_size': len(embedding),
        'embedding': embedding.idx_to_vec,
    }
    with open('{:s}/{:s}'.format(args.data_folder, args.meta_out_file), 'wb') as f:
        msgpack.dump(meta, f)

if __name__ == '__main__':
    main()
#    data, label = load_raw_csv("20180805/raw_data.csv")
#    with open("20180805/test_out.csv", 'w') as f:
#        for i in range(len(data)):
#            d, l = data[i], label[i]
#            line = str(l) + '\t' + ujson.dumps(d) + '\n'
#            f.write(line)
