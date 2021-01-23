#!/bin/bash

##OPTIONS
MIN_FREQ=5
MIN_FREQ_TARGET=2
MIN_FREQ_TAG=5
LOWERCASE=True
LOWERCASE_TARGET=FALSE
MAX_SENT_LEN="512"

PY_PATH="~/anaconda3/bin/python"

##INPUT
INPUT_NAME=gorc
INPUT_FILE_ALL="./data/raw/$INPUT_NAME/all_paper_data"
CBOW_IN="/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/resources/word2vec_pretraining/dim200/embeddings-basic-cbow.txt"


##OUTPUT
DATA_NAME="${INPUT_NAME}_fix_uncased_min_${MIN_FREQ}"
OUTPUT_DIR="./data/processed/$DATA_NAME/"
OUTPUT_DIR_FEATURE="./data/processed/$DATA_NAME/feature/"
OUTPUT_DIR_TYPE="./data/processed/$DATA_NAME/type/"
OUTPUT_DIR_USER="./data/processed/$DATA_NAME/user/"
OUTPUT_DIR_TAG="./data/processed/${DATA_NAME}/tag/"
TENSOR_FOLDER="tensors_cold"
CBOW_OUT="./resources/cbow_ACM_dim200_${DATA_NAME}.txt"


##Path for intermediate files
INPUT_FILE="./data/raw/$INPUT_NAME/meta"
INPUT_FILE_USER="./data/raw/$INPUT_NAME/user"
INPUT_FILE_TAG="./data/raw/$INPUT_NAME/tags"

echo "convert words to indices"

mkdir -p $OUTPUT_DIR_TYPE
eval $PY_PATH src/preprocessing/Amazon/amazon_split_files.py -i $INPUT_FILE_ALL -f $INPUT_FILE -y $OUTPUT_DIR_TYPE/corpus_index -u $INPUT_FILE_USER -t $INPUT_FILE_TAG
eval $PY_PATH src/preprocessing/map_tokens_to_indices.py --data $INPUT_FILE --save $OUTPUT_DIR_FEATURE --min_freq $MIN_FREQ --lowercase $LOWERCASE --eos True
eval $PY_PATH src/preprocessing/map_tokens_to_indices.py --data $INPUT_FILE_USER --save $OUTPUT_DIR_USER --min_freq $MIN_FREQ_TARGET --lowercase $LOWERCASE_TARGET --ignore_unk True
eval $PY_PATH src/preprocessing/map_tokens_to_indices.py --data $INPUT_FILE_TAG --save $OUTPUT_DIR_TAG --min_freq $MIN_FREQ_TAG --lowercase $LOWERCASE_TARGET --ignore_unk True

echo "filter word embedding"
eval $PY_PATH src/preprocessing/filter_emb.py -f $OUTPUT_DIR_FEATURE/dictionary_index -e $CBOW_IN -o $CBOW_OUT

echo "convert indices to tensor"
eval $PY_PATH src/preprocessing/map_indices_to_tensors.py --data_feature $OUTPUT_DIR_FEATURE --data_type $OUTPUT_DIR_TYPE --data_user $OUTPUT_DIR_USER --data_tag $OUTPUT_DIR_TAG --save $OUTPUT_DIR/$TENSOR_FOLDER --max_sent_len $MAX_SENT_LEN --cv_fold_num 10 --only_first_fold True

