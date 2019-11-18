#!/bin/bash
#MIN_FREQ=$1
MIN_FREQ=100
#LOWERCASE=True
LOWERCASE=False
echo $MIN_FREQ
INPUT_FILE="./data/raw/wiki2016_bert_tok.txt"
#DATA_NAME="wiki2016_min$MIN_FREQ"
DATA_NAME="wiki2016_bert_min$MIN_FREQ"
OUTPUT_DIR="./data/processed/$DATA_NAME/"

#GLOVE_IN="/iesl/data/word_embedding/glove.840B.300d.txt"
#GLOVE_OUT="./resources/glove.840B.300d_filtered_${DATA_NAME}.txt"
#WORD2VEC_IN="/iesl/data/word_embedding/GoogleNews-vectors-negative300.txt"
#WORD2VEC_OUT="./resources/Google-vec-neg300_filtered_${DATA_NAME}.txt"

#TENSOR_FOLDER="tensors"
#MAX_SENT_LEN="50"
#MULTI_SENT="False"


echo "convert words to indices"
~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --data $INPUT_FILE --save $OUTPUT_DIR --min_freq $MIN_FREQ --lowercase $LOWERCASE

#echo "filter word embedding"
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR/dictionary_index -e $GLOVE_IN -o $GLOVE_OUT
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR/dictionary_index -e $WORD2VEC_IN -o $WORD2VEC_OUT

#echo "convert indices to tensor"
#~/anaconda3/bin/python src/preprocessing/map_indices_to_tensors.py --data $OUTPUT_DIR --save $OUTPUT_DIR/$TENSOR_FOLDER/ --max_sent_len $MAX_SENT_LEN --multi_sent $MULTI_SENT

#TENSOR_FOLDER="tensors_multi150"
#MAX_SENT_LEN="150"
#MULTI_SENT="True"
#~/anaconda3/bin/python src/preprocessing/map_indices_to_tensors.py --data $OUTPUT_DIR --save $OUTPUT_DIR/$TENSOR_FOLDER/ --max_sent_len $MAX_SENT_LEN --multi_sent $MULTI_SENT
