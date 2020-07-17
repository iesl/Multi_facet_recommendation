#!/bin/bash
#MIN_FREQ=$1
MIN_FREQ_TARGET=0
MIN_FREQ_TAG=0
LOWERCASE=True
#LOWERCASE=False
LOWERCASE_TARGET=FALSE
#INPUT_NAME=NeurIPS2019_bid_score
#INPUT_NAME=ICLR2018_bid_score
#INPUT_NAME=ICLR2019_bid_score
#INPUT_NAME=ICLR2020_bid_score
#INPUT_NAME=ICLR2020
#INPUT_NAME=ICLR2020_bid_low
#INPUT_NAME=ICLR2020_bid_high
#INPUT_NAME=UAI2019
INPUT_NAME=UAI2019_bid_score
#INPUT_NAME=UAI2019_bid_low
#INPUT_NAME=UAI2019_bid_high
#echo $MIN_FREQ
#INPUT_FILE_ALL="./data/raw/$INPUT_NAME/all_paper_data"
INPUT_FILE_TRAIN_ALL="./data/raw/openreview/$INPUT_NAME/all_reviewer_paper_data"
#INPUT_FILE_TEST_ALL="./data/raw/openreview/$INPUT_NAME/all_submission_paper_data"
INPUT_FILE_TEST_ALL="./data/raw/openreview/$INPUT_NAME/all_submission_bid_data"
#INPUT_FEATURE_VOCAB_FILE="./data/processed/gorc_uncased_min_5/feature/dictionary_index"
INPUT_FEATURE_VOCAB_FILE="./data/processed/gorc_org_uncased_min_5/feature/dictionary_index"
INPUT_FILE="./data/raw/openreview/$INPUT_NAME/meta"
INPUT_FILE_USER="./data/raw/openreview/$INPUT_NAME/user"
INPUT_FILE_TAG="./data/raw/openreview/$INPUT_NAME/tags"
#INPUT_FILE="./data/raw/wiki2016.txt"
#DATA_NAME="wiki2016_min$MIN_FREQ"
#DATA_NAME="${INPUT_NAME}_cased_min_50"
#DATA_NAME="${INPUT_NAME}_gorc_uncased"
DATA_NAME="${INPUT_NAME}_gorc_org_uncased"
OUTPUT_DIR="./data/processed/$DATA_NAME/"
OUTPUT_DIR_FEATURE="./data/processed/$DATA_NAME/feature/"
OUTPUT_DIR_TYPE="./data/processed/$DATA_NAME/type/"
OUTPUT_DIR_USER="./data/processed/$DATA_NAME/user/"
OUTPUT_DIR_BID_SCORE="./data/processed/$DATA_NAME/bid_score/"
OUTPUT_DIR_TAG="./data/processed/${DATA_NAME}/tag/"

#GLOVE_IN="/iesl/data/word_embedding/glove.840B.300d.txt"
#GLOVE_OUT="./resources/glove.840B.300d_filtered_${DATA_NAME}.txt"
#CBOW_IN="/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/resources/word2vec_pretraining/dim200/embeddings-basic-cbow.txt"
#CBOW_OUT="./resources/cbow_ACM_dim200_${DATA_NAME}.txt"

TENSOR_FOLDER="tensors_cold"
MAX_SENT_LEN="512"
#MAX_SENT_LEN="128"

echo "convert words to indices"

mkdir -p $OUTPUT_DIR_TYPE
mkdir -p $OUTPUT_DIR_FEATURE
mkdir -p $OUTPUT_DIR_BID_SCORE
~/anaconda3/bin/python src/preprocessing/Amazon/amazon_split_files.py -i $INPUT_FILE_TRAIN_ALL -f ${INPUT_FILE}_train -y $OUTPUT_DIR_TYPE/corpus_index -u ${INPUT_FILE_USER}_train -t ${INPUT_FILE_TAG}_train -p ${OUTPUT_DIR}/paper_id_train
~/anaconda3/bin/python src/preprocessing/Amazon/amazon_split_files.py -i $INPUT_FILE_TEST_ALL -f ${INPUT_FILE}_test -y $OUTPUT_DIR_TYPE/corpus_index_test -u ${INPUT_FILE_USER}_test -t ${INPUT_FILE_TAG}_test -p ${OUTPUT_DIR}/paper_id_test -b ${OUTPUT_DIR_BID_SCORE}/corpus_index_test

cp $INPUT_FEATURE_VOCAB_FILE $OUTPUT_DIR_FEATURE/dictionary_index
~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --input_vocab $INPUT_FEATURE_VOCAB_FILE --update_dict False --data ${INPUT_FILE}_train --save ${OUTPUT_DIR_FEATURE}/corpus_index --lowercase $LOWERCASE --eos True
~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --input_vocab $INPUT_FEATURE_VOCAB_FILE --update_dict False --data ${INPUT_FILE}_test --save ${OUTPUT_DIR_FEATURE}/corpus_index_test --lowercase $LOWERCASE --eos True

~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --data ${INPUT_FILE_USER}_train --save $OUTPUT_DIR_USER --min_freq $MIN_FREQ_TARGET --lowercase $LOWERCASE_TARGET --ignore_unk True
~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --input_vocab ${OUTPUT_DIR_USER}/dictionary_index --update_dict False --data ${INPUT_FILE_USER}_test --save ${OUTPUT_DIR_USER}/corpus_index_test --lowercase $LOWERCASE_TARGET --ignore_unk True

~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --data ${INPUT_FILE_TAG}_train --save $OUTPUT_DIR_TAG --min_freq $MIN_FREQ_TAG --lowercase $LOWERCASE_TARGET --ignore_unk True
~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --input_vocab ${OUTPUT_DIR_TAG}/dictionary_index --update_dict True --data ${INPUT_FILE_TAG}_test --save $OUTPUT_DIR_TAG --output_file_name corpus_index_test --min_freq $MIN_FREQ_TAG --lowercase $LOWERCASE_TARGET --ignore_unk True

echo "filter word embedding"
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR/dictionary_index -e $GLOVE_IN -o $GLOVE_OUT
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR_FEATURE/dictionary_index -e $GLOVE_IN -o $GLOVE_OUT
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR_FEATURE/dictionary_index -e $WORD2VEC_IN -o $WORD2VEC_OUT
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR_FEATURE/dictionary_index -e $CBOW_IN -o $CBOW_OUT

echo "convert indices to tensor"
~/anaconda3/bin/python src/preprocessing/map_indices_to_tensors.py --input_file_name corpus_index --data_feature $OUTPUT_DIR_FEATURE --data_type $OUTPUT_DIR_TYPE --data_user $OUTPUT_DIR_USER --data_tag $OUTPUT_DIR_TAG --save $OUTPUT_DIR/$TENSOR_FOLDER/train.pt --max_sent_len $MAX_SENT_LEN --cv_fold_num 0
~/anaconda3/bin/python src/preprocessing/map_indices_to_tensors.py --input_file_name corpus_index_test --data_feature $OUTPUT_DIR_FEATURE --data_type $OUTPUT_DIR_TYPE --data_user $OUTPUT_DIR_USER --data_tag $OUTPUT_DIR_TAG --save $OUTPUT_DIR/$TENSOR_FOLDER/val.pt --max_sent_len $MAX_SENT_LEN --cv_fold_num 0 --data_bid_score $OUTPUT_DIR_BID_SCORE
cp $OUTPUT_DIR/$TENSOR_FOLDER/val.pt $OUTPUT_DIR/$TENSOR_FOLDER/test.pt

