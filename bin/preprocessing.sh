#!/bin/bash
#MIN_FREQ=$1
MIN_FREQ=5
MIN_FREQ_TARGET=0
LOWERCASE=True
LOWERCASE_TARGET=FALSE
#LOWERCASE=False
#INPUT_NAME=citeulike-a
INPUT_NAME=citeulike-t
echo $MIN_FREQ
INPUT_FILE="./data/raw/$INPUT_NAME/paper_text"
INPUT_FILE_USER="./data/raw/$INPUT_NAME/user"
INPUT_FILE_TAG="./data/raw/$INPUT_NAME/tags"
#INPUT_FILE="./data/raw/wiki2016.txt"
#DATA_NAME="wiki2016_min$MIN_FREQ"
DATA_NAME="${INPUT_NAME}_lower"
OUTPUT_DIR="./data/processed/$DATA_NAME/"
OUTPUT_DIR_FEATURE="./data/processed/$DATA_NAME/feature/"
OUTPUT_DIR_USER="./data/processed/$DATA_NAME/user/"
OUTPUT_DIR_TAG="./data/processed/$DATA_NAME/tag/"

#GLOVE_IN="/iesl/data/word_embedding/glove.840B.300d.txt"
#GLOVE_OUT="./resources/glove.840B.300d_filtered_${DATA_NAME}.txt"
#WORD2VEC_IN="/iesl/data/word_embedding/GoogleNews-vectors-negative300.txt"
#WORD2VEC_OUT="./resources/Google-vec-neg300_filtered_${DATA_NAME}.txt"
#WORD2VEC_IN="/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/resources/word2vec_sg_wiki2016_lower_min100_w5_s5.txt"
#WORD2VEC_OUT="./resources/word2vec_sg_wiki2016_min100_w5_s5_${DATA_NAME}.txt"
CBOW_IN="/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/resources/word2vec_pretraining/dim200/embeddings-basic-cbow.txt"
CBOW_OUT="./resources/cbow_ACM_dim200_${DATA_NAME}.txt"

TENSOR_FOLDER="tensors_cold"
MAX_SENT_LEN="512"


echo "convert words to indices"
#~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --data $INPUT_FILE --save $OUTPUT_DIR_FEATURE --min_freq $MIN_FREQ --lowercase $LOWERCASE --eos True
~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --data $INPUT_FILE_USER --save $OUTPUT_DIR_USER --min_freq $MIN_FREQ_TARGET --lowercase $LOWERCASE_TARGET
~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --data $INPUT_FILE_TAG --save $OUTPUT_DIR_TAG --min_freq $MIN_FREQ_TARGET --lowercase $LOWERCASE_TARGET

echo "filter word embedding"
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR/dictionary_index -e $GLOVE_IN -o $GLOVE_OUT
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR_FEATURE/dictionary_index -e $WORD2VEC_IN -o $WORD2VEC_OUT
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR_FEATURE/dictionary_index -e $CBOW_IN -o $CBOW_OUT

echo "convert indices to tensor"
~/anaconda3/bin/python src/preprocessing/map_indices_to_tensors.py --data_feature $OUTPUT_DIR_FEATURE --data_user $OUTPUT_DIR_USER --data_tag $OUTPUT_DIR_TAG --save $OUTPUT_DIR/$TENSOR_FOLDER --max_sent_len $MAX_SENT_LEN

