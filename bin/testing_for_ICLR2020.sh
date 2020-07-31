#!/bin/bash
module load python3/current
echo "code is running"

DATASET="ICLER2020"

DATA_FOLDER="./data/processed/${DATASET}_gorc_fix_uncased"
TENSOR_FOLDER="tensors_cold"
SUBMISSION_DATA="$DATA_FOLDER/$TENSOR_FOLDER/test.pt"
REVIEW_DATA="$DATA_FOLDER/$TENSOR_FOLDER/train.pt"
WORD_DICT="$DATA_FOLDER/feature/dictionary_index"
PAPER_DICT="$DATA_FOLDER/user/dictionary_index"


#Compute SPECTER similarity
SPECTER_TEST_EMB_RAW="gen_log/${DATASET}_emb_spector_raw.jsonl"
SPECTER_TRAIN_EMB_RAW="gen_log/${DATASET}_emb_spector_raw_train.jsonl"
SPECTER_TEST_EMB="gen_log/${DATASET}_emb_spector_norm.txt"
SPECTER_TRAIN_EMB="gen_log/${DATASET}_emb_spector_train_norm.txt"
SEPCTER_SIM="./gen_log/${DATASET}_spector_max_dist.np"

~/anaconda3/bin/python src/testing/avg_baseline/compute_avg_emb_spector.py -i $SPECTER_TEST_EMB_RAW -o $SPECTER_TEST_EMB
~/anaconda3/bin/python src/testing/avg_baseline/compute_avg_emb_spector.py -i $SPECTER_TRAIN_EMB_RAW -o $SPECTER_TRAIN_EMB
~/anaconda3/bin/python src/testing/avg_baseline/eval_avg_emb.py -i $WORD_DICT -u $PAPER_DICT -v $SUBMISSION_DATA -s $SPECTER_TEST_EMB -t $REVIEW_DATA -r $SPECTER_TRAIN_EMB -o $SEPCTER_SIM -m 'save_dist' -d 'sim_max'


#Compute our similarity
MODEL_PATH="./models/gorc_fix-20200722-104422"
OUR_SIM="./gen_log/${DATASET}_trans_n1.np"
CBOW_FILE="/dev/null"
~/anaconda3/bin/python src/main.py --batch_size 50 --save $MODEL_PATH --de_output_layer single_dynamic --continue_train --data $DATA_FOLDER --source_emb_file $CBOW_FILE --de_model TRANS --en_model TRANS --encode_trans_layers 3 --trans_layers 3 --n_basis 1 --seed 111 --epochs 100 --tensor_folder tensors_cold --coeff_opt max --tag_w 1 --user_w 5 --rand_neg_method shuffle --target_norm True --loss_type dist --inv_freq_w True --de_coeff_model TRANS_old --lr 0.0002 --target_emsize 100 --freeze_encoder_decoder True --loading_target_embedding False --always_save_model True --target_embedding_suffix _${DATASET}_test --log_file_name log_${DATASET}_test.txt

~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf $OUR_SIM --tag_emb_file tag_emb_${DATASET}_test_always.pt --user_emb_file user_emb_${DATASET}_test_always.pt --batch_size 50 --n_basis 1 --data $DATA_FOLDER --tensor_folder tensors_cold --coeff_opt max --loss_type dist --test_tag False --source_emsize 200 --en_model TRANS --encode_trans_layers 3 --trans_layers 3 --de_coeff_model TRANS_old --de_output_layer single_dynamic --store_dist user


#Combine both
MERGE_ALPHA=0.8
OUTPUT_CSV="gen_log/${DATASET}_specter_ours_sim.csv"
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i $SEPCTER_SIM -j $OUR_SIM -d $DATA_FOLDER -t $TENSOR_FOLDER -a $MERGE_ALPHA
~/anaconda3/bin/python src/testing/avg_baseline/dump_paper_dist_to_csv.py -d $DATA_FOLDER -f $SEPCTER_SIM -s $OUR_SIM -a $MERGE_ALPHA -o $OUTPUT_CSV
