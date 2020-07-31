#!/bin/bash
#MIN_FREQ=$1
MIN_FREQ=5
MIN_FREQ_TARGET=2
MIN_FREQ_TAG=5
LOWERCASE=True
#LOWERCASE=False
LOWERCASE_TARGET=FALSE
INPUT_NAME=gorc
#echo $MIN_FREQ
INPUT_FILE_ALL="./data/raw/$INPUT_NAME/all_paper_data"
INPUT_FILE="./data/raw/$INPUT_NAME/meta"
#INPUT_FILE_TYPE="./data/raw/amazon/$INPUT_NAME/type"
INPUT_FILE_USER="./data/raw/$INPUT_NAME/user"
INPUT_FILE_TAG="./data/raw/$INPUT_NAME/tags"
#INPUT_FILE_ALL="./data/raw/$INPUT_NAME/all_paper_org_data"
#INPUT_FILE="./data/raw/$INPUT_NAME/meta_org"
#INPUT_FILE_USER="./data/raw/$INPUT_NAME/user_org"
#INPUT_FILE_TAG="./data/raw/$INPUT_NAME/tags_org"
#INPUT_FILE="./data/raw/wiki2016.txt"
#DATA_NAME="wiki2016_min$MIN_FREQ"
#DATA_NAME="${INPUT_NAME}_cased_min_50"
#DATA_NAME="${INPUT_NAME}_org_uncased_min_${MIN_FREQ}"
DATA_NAME="${INPUT_NAME}_fix_uncased_min_${MIN_FREQ}"
OUTPUT_DIR="./data/processed/$DATA_NAME/"
OUTPUT_DIR_FEATURE="./data/processed/$DATA_NAME/feature/"
OUTPUT_DIR_TYPE="./data/processed/$DATA_NAME/type/"
OUTPUT_DIR_USER="./data/processed/$DATA_NAME/user/"
OUTPUT_DIR_TAG="./data/processed/${DATA_NAME}/tag/"

#GLOVE_IN="/iesl/data/word_embedding/glove.840B.300d.txt"
#GLOVE_OUT="./resources/glove.840B.300d_filtered_${DATA_NAME}.txt"
CBOW_IN="/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/resources/word2vec_pretraining/dim200/embeddings-basic-cbow.txt"
CBOW_OUT="./resources/cbow_ACM_dim200_${DATA_NAME}.txt"

TENSOR_FOLDER="tensors_cold"
MAX_SENT_LEN="512"
#MAX_SENT_LEN="128"

echo "convert words to indices"

mkdir -p $OUTPUT_DIR_TYPE
~/anaconda3/bin/python src/preprocessing/Amazon/amazon_split_files.py -i $INPUT_FILE_ALL -f $INPUT_FILE -y $OUTPUT_DIR_TYPE/corpus_index -u $INPUT_FILE_USER -t $INPUT_FILE_TAG
~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --data $INPUT_FILE --save $OUTPUT_DIR_FEATURE --min_freq $MIN_FREQ --lowercase $LOWERCASE --eos True
~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --data $INPUT_FILE_USER --save $OUTPUT_DIR_USER --min_freq $MIN_FREQ_TARGET --lowercase $LOWERCASE_TARGET --ignore_unk True
~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --data $INPUT_FILE_TAG --save $OUTPUT_DIR_TAG --min_freq $MIN_FREQ_TAG --lowercase $LOWERCASE_TARGET --ignore_unk True

echo "filter word embedding"
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR/dictionary_index -e $GLOVE_IN -o $GLOVE_OUT
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR_FEATURE/dictionary_index -e $GLOVE_IN -o $GLOVE_OUT
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR_FEATURE/dictionary_index -e $WORD2VEC_IN -o $WORD2VEC_OUT
~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR_FEATURE/dictionary_index -e $CBOW_IN -o $CBOW_OUT

echo "convert indices to tensor"
~/anaconda3/bin/python src/preprocessing/map_indices_to_tensors.py --data_feature $OUTPUT_DIR_FEATURE --data_type $OUTPUT_DIR_TYPE --data_user $OUTPUT_DIR_USER --data_tag $OUTPUT_DIR_TAG --save $OUTPUT_DIR/$TENSOR_FOLDER --max_sent_len $MAX_SENT_LEN --cv_fold_num 10 --only_first_fold True

