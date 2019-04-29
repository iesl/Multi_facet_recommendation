#!/bin/bash
#MIN_FREQ=$1
MIN_FREQ=100
LOWERCASE=True
echo $MIN_FREQ
INPUT_FILE="./data/raw/wiki2016_nchunk.txt"
#DATA_NAME="wiki2016_min$MIN_FREQ"
DATA_NAME="wiki2016_nchunk_lower_min$MIN_FREQ"
OUTPUT_DIR="./data/processed/$DATA_NAME/"

#GLOVE_IN="/iesl/data/word_embedding/glove.42B.300d.txt"
GLOVE_IN="./resources/glove.42B.300d.txt"
GLOVE_OUT="./resources/glove.42B.300d_filtered_${DATA_NAME}.txt"
#WORD2VEC_IN="/iesl/data/word_embedding/GoogleNews-vectors-negative300.txt"
#WORD2VEC_OUT="./resources/Google-vec-neg300_filtered_${DATA_NAME}.txt"

TENSOR_FOLDER="tensors"
MAX_SENT_LEN="6"


echo "convert words to indices"
#~/anaconda3/bin/python src/preprocessing/map_tokens_to_indices.py --data $INPUT_FILE --save $OUTPUT_DIR --min_freq $MIN_FREQ --lowercase $LOWERCASE --min_sent_length 0

#src/preprocessing/analyze_freq_threshold.py

#subsampling the dataset?

#DATASET="./dataset_testing/SemEval2013/FCT_format/en.trainSet"
#~/anaconda3/bin/python src/testing/sim_word/OOV_percentage_checking.py -d $OUTPUT_DIR/dictionary_index -t $DATASET
#DATASET="./dataset_testing/SemEval2013/FCT_format/en.testSet"
#~/anaconda3/bin/python src/testing/sim_word/OOV_percentage_checking.py -d $OUTPUT_DIR/dictionary_index -t $DATASET
#DATASET="./dataset_testing/Turney2012/FCT_format/jair.data"
#~/anaconda3/bin/python src/testing/sim_word/OOV_percentage_checking.py -d $OUTPUT_DIR/dictionary_index -t $DATASET



echo "check overlapping between testing files and dictionary file"

echo "filter word embedding"
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR/dictionary_index -e $GLOVE_IN -o $GLOVE_OUT
#~/anaconda3/bin/python src/preprocessing/filter_emb.py -f $OUTPUT_DIR/dictionary_index -e $WORD2VEC_IN -o $WORD2VEC_OUT

echo "convert indices to tensor"
WIN_SIZE=5
~/anaconda3/bin/python src/preprocessing/map_word_indices_to_tensors_memory_saving.py --data $OUTPUT_DIR --save $OUTPUT_DIR/$TENSOR_FOLDER/ --max_sent_len $MAX_SENT_LEN --window_size $WIN_SIZE

