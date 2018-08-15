#!/bin/bash

python train.py \
    --data_file='20180805/data.msgpack' \
    --meta_file='20180805/meta.msgpack' \
    --log_file='log/s2t_double.log' \
    --best_in_valid_model_name='s2t_double.param' \
    --saved_model_name='' \
    --max_epoch=10 \
    --batch_size=64 \
    --log_per_batch=50 \
    --learning_rate=1e-3 \
    --dropout_rate=0.2 \
    --kernel_size 3 4 5 \
    --feature_map=64 \
    --hidden_dim=128 \
    --activation='relu' \
    --num_classes=2 \
    --no_update_embedding \
