#!/bin/bash
set -e
echo "code is running"

PY_PATH="~/anaconda3/bin/python"

##INPUT
DATASET="ICLR2020"
TEXT_DATA_DIR="data/raw/openreview/${DATASET}"
REVIEWER_DIR="${TEXT_DATA_DIR}/source_data/archives"
SUBMISSION_DIR="${TEXT_DATA_DIR}/source_data/submissions"
MODEL_PATH="./models/gorc_fix-20200722-104422"
INPUT_FEATURE_VOCAB_FILE="./data/processed/gorc_fix_uncased_min_5/feature/dictionary_index"
SPECTER_FOLDER="/iesl/canvas/hschang/recommendation/specter"
CUDA_DEVICE_IDX="0" #Required by SPECTER

#OLD_CONFERENCE="Y"
OLD_CONFERENCE="N"
if [ $OLD_CONFERENCE == "Y" ]; then
    ASSIGNMENT_FILE="${TEXT_DATA_DIR}/source_data/assignments/assignments.json"
    BID_FILE="${TEXT_DATA_DIR}/source_data/bids/bids.json"
    TENSOR_FOLDER="tensors_cold"
    TEXT_DATA_BID_DIR="${TEXT_DATA_DIR}_bid_score"
    SUBMISSION_BID_FILE="${TEXT_DATA_BID_DIR}/all_submission_bid_data"
    PROCESSED_BID_DATA_DIR="./data/processed/${DATASET}_bid_score_gorc_fix_uncased"
else
    ASSIGNMENT_FILE=""
fi

EXPERTISE_FILE=""
#EXPERTISE_FILE="${TEXT_DATA_DIR}/source_data/profiles_expertise/profiles_expertise.json"

##OUTPUT
OUTPUT_CSV="gen_log/${DATASET}_specter_ours_sim.csv"

##Path for intermediate files
REVIWER_FILE="${TEXT_DATA_DIR}/all_reviewer_paper_data"
SUBMISSION_FILE="${TEXT_DATA_DIR}/all_submission_paper_data"
PROCESSED_DATA_DIR="./data/processed/${DATASET}_gorc_fix_uncased"
OUR_SIM="./gen_log/${DATASET}_trans_n1.np"

SAMPLE_ID_TRAIN="`pwd`/$PROCESSED_DATA_DIR/paper_id_train"
SAMPLE_ID_TEST="`pwd`/$PROCESSED_DATA_DIR/paper_id_test"
SPECTER_TRAIN_FILE="`pwd`/$TEXT_DATA_DIR/paper_data_spector_train.json"
SPECTER_TEST_FILE="`pwd`/$TEXT_DATA_DIR/paper_data_spector_test.json"
SPECTER_TRAIN_EMB_RAW="`pwd`/gen_log/${DATASET}_emb_spector_raw_train_duplicate.jsonl"
SPECTER_TEST_EMB_RAW="`pwd`/gen_log/${DATASET}_emb_spector_raw.jsonl"
SEPCTER_CSV="gen_log/${DATASET}_spector_sim.csv"
SEPCTER_SIM="./gen_log/${DATASET}_spector_dist.np"

#Preprocessing

mkdir -p gen_log

echo "running preprocessing"
eval $PY_PATH src/preprocessing/gorc/prepare_data_for_reviewer_emb.py -i $REVIEWER_DIR -e $EXPERTISE_FILE -o $REVIWER_FILE
eval $PY_PATH src/preprocessing/gorc/prepare_data_for_assignment_testing.py -i $SUBMISSION_DIR -e $EXPERTISE_FILE -a $ASSIGNMENT_FILE -o $SUBMISSION_FILE

./bin/preprocessing_openreview.sh $TEXT_DATA_DIR $REVIWER_FILE $SUBMISSION_FILE $PROCESSED_DATA_DIR $INPUT_FEATURE_VOCAB_FILE $PY_PATH


#Compute SPECTER similarity
echo "running spector"
eval $PY_PATH src/preprocessing/gorc/convert_paper_train_spector.py -i $REVIEWER_DIR -o $SPECTER_TRAIN_FILE
eval $PY_PATH src/preprocessing/gorc/convert_paper_train_spector.py -i $SUBMISSION_DIR -o $SPECTER_TEST_FILE
cd $SPECTER_FOLDER #If do not do this, specter will give you the error saying it cannot find data/scibert_scivocab_uncased/scibert.tar.gz, but I cannot find where to change that path
eval $PY_PATH $SPECTER_FOLDER/scripts/embed_abs_path.py --py_path $PY_PATH --specter_folder $SPECTER_FOLDER --ids $SAMPLE_ID_TRAIN --metadata $SPECTER_TRAIN_FILE --model $SPECTER_FOLDER/model.tar.gz --output-file $SPECTER_TRAIN_EMB_RAW --vocab-dir $SPECTER_FOLDER/data/vocab/ --batch-size 16 --cuda-device $CUDA_DEVICE_IDX
eval $PY_PATH $SPECTER_FOLDER/scripts/embed_abs_path.py --py_path $PY_PATH --specter_folder $SPECTER_FOLDER --ids $SAMPLE_ID_TEST --metadata $SPECTER_TEST_FILE --model $SPECTER_FOLDER/model.tar.gz --output-file $SPECTER_TEST_EMB_RAW --vocab-dir $SPECTER_FOLDER/data/vocab/ --batch-size 16 --cuda-device $CUDA_DEVICE_IDX
cd -
DIST_OPT="max"
eval $PY_PATH src/testing/avg_baseline/doc_sim_to_csv.py -s $SPECTER_TEST_EMB_RAW -r $SPECTER_TRAIN_EMB_RAW -d "sim_${DIST_OPT}" -p $REVIWER_FILE -o $SEPCTER_CSV
eval $PY_PATH src/testing/avg_baseline/paper_dist_from_csv.py -i $SEPCTER_CSV -d $PROCESSED_DATA_DIR -o $SEPCTER_SIM

#Compute our similarity
CBOW_FILE="/dev/null"
echo "running our methods"
eval $PY_PATH src/main.py --batch_size 50 --save $MODEL_PATH --de_output_layer single_dynamic --continue_train --data $PROCESSED_DATA_DIR --source_emb_file $CBOW_FILE --de_model TRANS --en_model TRANS --encode_trans_layers 3 --trans_layers 3 --n_basis 1 --seed 111 --epochs 100 --tensor_folder tensors_cold --coeff_opt max --tag_w 1 --user_w 5 --rand_neg_method shuffle --target_norm True --loss_type dist --inv_freq_w True --de_coeff_model TRANS_old --lr 0.0002 --target_emsize 100 --freeze_encoder_decoder True --loading_target_embedding False --always_save_model True --target_embedding_suffix _${DATASET}_test --log_file_name log_${DATASET}_test.txt --neg_sample_w 0

eval $PY_PATH src/recommend_test.py --checkpoint $MODEL_PATH --outf $OUR_SIM --tag_emb_file tag_emb_${DATASET}_test_always.pt --user_emb_file user_emb_${DATASET}_test_always.pt --batch_size 50 --n_basis 1 --data $PROCESSED_DATA_DIR --tensor_folder tensors_cold --coeff_opt max --loss_type dist --test_tag False --source_emsize 200 --en_model TRANS --encode_trans_layers 3 --trans_layers 3 --de_coeff_model TRANS_old --de_output_layer single_dynamic --store_dist user


#Combine both
echo "Merging two methods and dump the results"
MERGE_ALPHA=0.8
eval $PY_PATH src/testing/avg_baseline/dump_paper_dist_to_csv.py -d $PROCESSED_DATA_DIR -f $SEPCTER_SIM -s $OUR_SIM -a $MERGE_ALPHA -o $OUTPUT_CSV

if [ $OLD_CONFERENCE == "Y" ]; then
    eval $PY_PATH src/preprocessing/gorc/prepare_data_for_assignment_testing.py -i $SUBMISSION_DIR -e $EXPERTISE_FILE -s "bid" -b $BID_FILE -o $SUBMISSION_BID_FILE
    ./bin/preprocessing_openreview_bid_score.sh ${TEXT_DATA_BID_DIR} $REVIWER_FILE $SUBMISSION_BID_FILE $PROCESSED_BID_DATA_DIR $INPUT_FEATURE_VOCAB_FILE $PY_PATH
    eval $PY_PATH src/testing/avg_baseline/merge_dist.py -i $SEPCTER_SIM -j $OUR_SIM -d $PROCESSED_DATA_DIR -t $TENSOR_FOLDER -a $MERGE_ALPHA
    eval $PY_PATH src/testing/avg_baseline/merge_dist.py -i $SEPCTER_SIM -j $OUR_SIM -d $PROCESSED_BID_DATA_DIR -t $TENSOR_FOLDER -a $MERGE_ALPHA
fi
