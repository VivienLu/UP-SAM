METHOD=$1
DATA_DIR=$2
SAVE_PATH=$3
DATASET=$4
LABEL_NUM=$5
BATCH_SIZE=$6
GPU=$7

python train.py  --stage 'pretrain' --method $METHOD \
                 --data_dir $DATA_DIR --dataset $DATASET \
                 --labeled_num $LABEL_NUM --batch_size $BATCH_SIZE \
                 --gpu $GPU --save_path $SAVE_PATH 

CHECKPOINT_VNET='${SAVE_PATH}/${DATASET}_lab-${LABEL_NUM}/${METHOD}/pretrain/best_model.pth'
LABEL_BS=1
python train.py  --stage 'semi-supervised' --method $METHOD \
                 --data_dir $DATA_DIR --dataset $DATASET \
                 --labeled_num $LABEL_NUM --batch_size $BATCH_SIZE \
                 --gpu $GPU --save_path $SAVE_PATH \
                 --labeled_bs $LABEL_BS --checkpoint_path_vnet_amc $CHECKPOINT_VNET


MODEL_PATH='${SAVE_PATH}/${DATASET}_lab-${LABEL_NUM}/${METHOD}/semi-supervised/best_model.pth'
LIST_PATH='${DATA_DIR}/lists/${DATASET}/test.list'
python test.py --data_dir $DATA_DIR --gpu $GPU --model_path $MODEL_PATH --list_dir $LIST_PATH