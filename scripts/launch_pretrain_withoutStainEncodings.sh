#!/bin/bash

### Global and Local loss pretraining without stain encodings ###
python pretrain.py \
    --data_root_dir ../data/pretrain/ACROBAT/feats_h5 \
    --results_dir results_brca \
    --cohort brca \
    --dataset ACROBAT \
    --csv_fpath ../dataset_csv/ACROBAT/ACROBAT.csv \
    --wsi_encoder abmil \
    --n_heads 4 \
    --patch_embedding_dim 512 \
    --wsi_encoder_hidden_dim 512 \
    --global_loss "info-nce" \
    --local_loss "got" \
    --local_loss_weight 1.0 \
    --temperature 0.001 \
    --lr 0.0001 \
    --max_epochs 120 \
    --batch_size 65 \
    --num_gpus 3 \
    --opt adamW \
    --num_workers 20 \
    --n_subsamples 2048 \
    --activation softmax \
    --warmup_epochs 5 \
    --warmup \
    --symmetric_cl \
    --precision bfloat16
