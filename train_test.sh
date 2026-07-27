#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 8 ]; then
    echo "Usage: bash train_test.sh METHOD DATA_DIR SAVE_PATH DATASET LABEL_NUM BATCH_SIZE GPU SAM_CHECKPOINT"
    exit 1
fi

METHOD=$1
DATA_DIR=$2
SAVE_PATH=$3
DATASET=$4
LABEL_NUM=$5
BATCH_SIZE=$6
GPU=$7
SAM_CHECKPOINT=$8

# Stage 1: supervised pre-training on the labeled subset.
python train.py --stage pretrain --method "$METHOD" \
                --data_dir "$DATA_DIR" --dataset "$DATASET" \
                --labeled_num "$LABEL_NUM" --batch_size "$BATCH_SIZE" \
                --base_lr 0.01 --gpu "$GPU" --save_path "$SAVE_PATH"

CHECKPOINT_VNET="${SAVE_PATH}/${DATASET}_lab-${LABEL_NUM}/${METHOD}/pretrain/best_model.pth"
LABEL_BS=$((BATCH_SIZE / 2))
if [ "$LABEL_BS" -lt 1 ]; then
    LABEL_BS=1
fi

# Stage 2: semi-supervised UP-SAM training.
python train.py --stage semi_train --method "$METHOD" \
                --data_dir "$DATA_DIR" --dataset "$DATASET" \
                --labeled_num "$LABEL_NUM" --batch_size "$BATCH_SIZE" \
                --labeled_bs "$LABEL_BS" --base_lr 0.00001 \
                --checkpoint_path_vnet_amc "$CHECKPOINT_VNET" \
                --checkpoint_path_sam "$SAM_CHECKPOINT" \
                --gpu "$GPU" --save_path "$SAVE_PATH"

# Inference with the best domain-specific model.
MODEL_PATH="${SAVE_PATH}/${DATASET}_lab-${LABEL_NUM}/${METHOD}/semi_train/best_model.pth"
python test.py --data_dir "$DATA_DIR" --gpu "$GPU" --model_path "$MODEL_PATH"
