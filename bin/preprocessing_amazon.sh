#!/bin/bash
#MIN_FREQ=$1
MIN_FREQ=5
#MIN_FREQ=50
MIN_FREQ_TARGET=0
#LOWERCASE=True
LOWERCASE=False
LOWERCASE_TARGET=FALSE
#INPUT_NAME=home
#INPUT_NAME=electronics
#INPUT_NAME=sport
INPUT_NAME=tool
#INPUT_NAME=cloth
#INPUT_NAME=phone
#INPUT_NAME=toy
#INPUT_NAME=movie
#INPUT_NAME=citeulike-t
#echo $MIN_FREQ

#DOWNLOAD_NAME="Cell_Phones_and_Accessories"
#DOWNLOAD_NAME="Electronics"
#DOWNLOAD_NAME="Home_and_Kitchen"
#DOWNLOAD_NAME="Sports_and_Outdoors"
DOWNLOAD_NAME="Tools_and_Home_Improvement"
REVIEW_ONLY="data/raw/amazon/${INPUT_NAME}/${DOWNLOAD_NAME}.csv"
PRODUCT_INFO="data/raw/amazon/${INPUT_NAME}/meta_${DOWNLOAD_NAME}.json.gz"
ASIN_USER_FILE="./data/raw/amazon/$INPUT_NAME/asin_to_user.tsv"
INPUT_FILE_ALL="./data/raw/amazon/$INPUT_NAME/all_${INPUT_NAME}_data"
INPUT_FILE="./data/raw/amazon/$INPUT_NAME/meta"
#INPUT_FILE_TYPE="./data/raw/amazon/$INPUT_NAME/type"
INPUT_FILE_USER="./data/raw/amazon/$INPUT_NAME/user"
INPUT_FILE_TAG="./data/raw/amazon/$INPUT_NAME/tags"
#INPUT_FILE="./data/raw/wiki2016.txt"
#DATA_NAME="wiki2016_min$MIN_FREQ"
#DATA_NAME="${INPUT_NAME}_cased_min_50"
DATA_NAME="${INPUT_NAME}_cased_min_${MIN_FREQ}"
OUTPUT_DIR="./data/processed/amazon/$DATA_NAME/"
OUTPUT_DIR_FEATURE="./data/processed/amazon/$DATA_NAME/feature/"
OUTPUT_DIR_TYPE="./data/processed/amazon/$DATA_NAME/type/"
OUTPUT_DIR_USER="./data/processed/amazon/$DATA_NAME/user/"
OUTPUT_DIR_TAG="./data/processed/amazon/$DATA_NAME/tag/"

GLOVE_IN="/iesl/data/word_embedding/glove.840B.300d.txt"
GLOVE_OUT="./resources/glove.840B.300d_filtered_${DATA_NAME}.txt"
#WORD2VEC_IN="/iesl/data/word_embedding/GoogleNews-vectors-negative300.txt"
#WORD2VEC_OUT="./resources/Google-vec-neg300_filtered_${DATA_NAME}.txt"
#WORD2VEC_IN="/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/resources/word2vec_sg_wiki2016_lower_min100_w5_s5.txt"
#WORD2VEC_OUT="./resources/word2vec_sg_wiki2016_min100_w5_s5_${DATA_NAME}.txt"
#CBOW_IN="/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/resources/word2vec_pretraining/dim200/embeddings-basic-cbow.txt"
#CBOW_OUT="./resources/cbow_ACM_dim200_${DATA_NAME}.txt"

TENSOR_FOLDER="tensors_cold"
#MAX_SENT_LEN="512"
MAX_SENT_LEN="128"

echo "convert words to indices"
~/anaconda3/bin/python src/preprocessing/Amazon/amazon_stats.py -i $REVIEW_ONLY -o $ASIN_USER_FILE
~/anaconda3/bin/python src/preprocessing/Amazon/amazon_feature_preparation.py -a $ASIN_USER_FILE -m $PRODUCT_INFO -o $INPUT_FILE_ALL

#cut -d$'\t' -f 1 $INPUT_FILE_ALL > $INPUT_FILE
#cut -d$'\t' -f 3 $INPUT_FILE_ALL > $INPUT_FILE_USER
#cut -d$'\t' -f 4 $INPUT_FILE_ALL > $INPUT_FILE_TAG
mkdir -p $OUTPUT_DIR_TYPE
~/anaconda3/bin/python src/preprocessing/Amazon/amazon_split_files.py -i $INPUT_FILE_ALL -f $INPUT_FILE -y $OUTPUT_DIR_TYPE/corpus_index -u $INPUT_FILE_USER -t $INPUT_FILE_TAG
~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --data $INPUT_FILE --save $OUTPUT_DIR_FEATURE --min_freq $MIN_FREQ --lowercase $LOWERCASE --eos True
~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --data $INPUT_FILE_USER --save $OUTPUT_DIR_USER --min_freq $MIN_FREQ_TARGET --lowercase $LOWERCASE_TARGET
~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --data $INPUT_FILE_TAG --save $OUTPUT_DIR_TAG --min_freq $MIN_FREQ_TARGET --lowercase $LOWERCASE_TARGET

echo "filter word embedding"
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR/dictionary_index -e $GLOVE_IN -o $GLOVE_OUT
~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR_FEATURE/dictionary_index -e $GLOVE_IN -o $GLOVE_OUT
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR_FEATURE/dictionary_index -e $WORD2VEC_IN -o $WORD2VEC_OUT
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR_FEATURE/dictionary_index -e $CBOW_IN -o $CBOW_OUT

echo "convert indices to tensor"
~/anaconda3/bin/python src/preprocessing/map_indices_to_tensors.py --data_feature $OUTPUT_DIR_FEATURE --data_type $OUTPUT_DIR_TYPE --data_user $OUTPUT_DIR_USER --data_tag $OUTPUT_DIR_TAG --save $OUTPUT_DIR/$TENSOR_FOLDER --max_sent_len $MAX_SENT_LEN --cv_fold_num 10 --only_first_fold True

