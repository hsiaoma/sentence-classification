#!/bin/bash

python train.py \
    --data_file='20180805/data.msgpack' \
    --meta_file='20180805/meta.msgpack' \
    --log_file='log/paratext_multi.log' \
    --best_in_valid_model_name='paratext_multi.param' \
    --saved_model_name='paratext_multi.param' \
    --max_epoch=10 \
    --batch_size=64 \
    --log_per_batch=50 \
    --learning_rate=1e-3 \
    --dropout_rate=0.5 \
    --kernel_size 3 4 5 \
    --feature_map=64 \
    --hidden_dim=128 \
    --activation='relu' \
    --num_classes=2 \
    --no_update_embedding \
