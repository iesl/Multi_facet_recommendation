#!/bin/bash
echo "code is running"

PY_PATH="~/anaconda3/bin/python"

##INPUT
INPUT_FOLDER="./data/processed/gorc_fix_uncased_min_5"
WORD_EMB_FILE="./resources/cbow_ACM_dim200_gorc_fix_uncased_min_5.txt"

##OUTPUT
OUTPUT_FOLDER="./models/gorc_fix"

##OPTIONS
N_EMBS=1
MODEL_MAGNITUDE=N
ENCODER=TRANS
#ENCODER=GRU

if [ $MODEL_MAGNITUDE == "Y" ]; then
    LOSS_ARG="--target_norm False --loss_type sim --target_l2 1e-6 --de_coeff_model TRANS_two_heads"
else
    LOSS_ARG="--target_norm True --loss_type dist --de_coeff_model TRANS_old"
fi

eval $PY_PATH src/main.py --batch_size 50 --save $OUTPUT_FOLDER --data $INPUT_FOLDER --source_emb_file $WORD_EMB_FILE --de_model TRANS --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --nlayers 2 --dropout 0.1 --dropout_prob_trans 0.1 --dropouti 0.1 --dropoute 0 --dropoutp 0.1 --n_basis $N_EMBS --seed 11 --epochs 100 --tensor_folder tensors_cold_0 --neg_sample_w 1 --coeff_opt max --tag_w 1 --user_w 5 --auto_w 1 --rand_neg_method shuffle --lr 0.0002 --target_emsize 100 --inv_freq_w True --auto_avg True --de_output_layer single_dynamic $LOSS_ARG

