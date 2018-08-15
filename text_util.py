#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import os, sys, pdb

class TextUtil(object):
    def __init__(self, token_file_path, embedding_file_path, vec_len = 300):
        self.vec_len = vec_len
        self.token_to_idx = {}
        self.token_to_idx["__PAD__"] = 0
        self.token_to_idx["__EOF__"] = 1
        self.token_to_idx["__OOV__"] = 2
        i = 3
        with open(token_file_path, 'r') as fp:
            for line in fp:
                self.token_to_idx[line.strip().decode('utf-8')] = i
                i += 1
        self.idx_to_vec = [None] * len(self.token_to_idx)
        with open(embedding_file_path, 'r') as fp:
            for line in fp:
                tmps = line.rstrip('\n').split(' ', 1)
                w = tmps[0].decode('utf-8')
                emb = [float(i) for i in tmps[1].rstrip().split(" ")]
                if w not in self.token_to_idx or len(emb) != self.vec_len:
                    continue
                self.idx_to_vec[self.token_to_idx[w]] = emb
        np.random.seed(42)
        for i in range(1, 3):
            self.idx_to_vec[i] = [np.random.uniform(-0.05,0.05) for _ in range(self.vec_len)]
        self.idx_to_vec[0] = [0] * 300
    def __len__(self):
        return len(self.token_to_idx)
    def get_idx(self, token):
        return self.token_to_idx.get(token, self.token_to_idx["__OOV__"])

if __name__ == "__main__":
    import time
    t0 = time.time()
    text = TextUtil("embedding_files/words.txt", "embedding_files/sgns.sogounews.bigram-char")
    print("time elpased = %f" %(time.time() - t0))
    pdb.set_trace()

