#!/bin/bash
module load python3/current
echo "code is running"

ENCODER="TRANS"

#MODEL_PATH="./models/gorc-20200413-205948" ; LOG_SUFFIX="no_lin_auto5_alldrop01" ; N_BASIS="1" ; DE_OUT="no" ; LOSS_TYPE="dist"
#MODEL_PATH="./models/gorc-20200419-025712" ; LOG_SUFFIX="_sim_no_lin_auto1_alldrop01_l2_switch" ; N_BASIS="1" ; DE_OUT="no" ; LOSS_TYPE="sim"
#MODEL_PATH="./models/gorc-20200419-010034" ; LOG_SUFFIX="_sim_no_lin_auto1_alldrop01_l2_switch" ; N_BASIS="5" ; DE_OUT="no" ; LOSS_TYPE="sim"
#MODEL_PATH="./models/gorc-20200419-031002" ; LOG_SUFFIX="_sim_no_lin_auto5_alldrop01" ; N_BASIS="1" ; DE_OUT="no" ; LOSS_TYPE="sim"
#MODEL_PATH="./models/gorc-20200419-031048" ; LOG_SUFFIX="_sim_no_lin_auto5_alldrop01" ; N_BASIS="5" ; DE_OUT="no" ; LOSS_TYPE="sim"
#MODEL_PATH="./models/gorc-20200419-170306" ; LOG_SUFFIX="_sim_w_freq_no_lin_auto5_alldrop01_l2" ; N_BASIS="1" ; DE_OUT="no" ; LOSS_TYPE="sim"
#MODEL_PATH="./models/gorc-20200419-170149" ; LOG_SUFFIX="_sim_w_freq_no_lin_auto5_alldrop01_l2" ; N_BASIS="5" ; DE_OUT="no" ; LOSS_TYPE="sim"

#MODEL_PATH="./models/gorc-20200420-053713" ; LOG_SUFFIX="_w_freq_no_lin_no_cite_auto1_alldrop01" ; N_BASIS="1" ; DE_OUT="no" ; LOSS_TYPE="dist"
#MODEL_PATH="./models/gorc-20200420-053802" ; LOG_SUFFIX="_w_freq_no_lin_no_cite_auto1_alldrop01" ; N_BASIS="5" ; DE_OUT="no" ; LOSS_TYPE="dist"
#MODEL_PATH="./models/gorc-20200420-095807" ; LOG_SUFFIX="_w_freq_single_auto_avg1_alldrop01" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="dist"
#MODEL_PATH="./models/gorc-20200420-095149" ; LOG_SUFFIX="_w_freq_no_lin_auto1_alldrop01" ; N_BASIS="3" ; DE_OUT="no" ; LOSS_TYPE="dist"
#MODEL_PATH="./models/gorc-20200521-151850" ; LOG_SUFFIX="_w_freq_single_fix_no_cite_auto_avg1_alldrop01" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="dist"

#MODEL_PATH="./models/gorc-20200420-152605" ; LOG_SUFFIX="_sim_single_auto_avg1_alldrop01_l2" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="sim"
#MODEL_PATH="./models/gorc-20200420-215007" ; LOG_SUFFIX="_sim_no_lin_auto_avg1_alldrop01_l2" ; N_BASIS="5" ; DE_OUT="no" ; LOSS_TYPE="sim"

#MODEL_PATH="./models/gorc-20200421-013748" ; LOG_SUFFIX="_w_freq_single_auto0_alldrop01_tfreq200" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="dist"
#MODEL_PATH="./models/gorc-20200421-013733" ; LOG_SUFFIX="_w_freq_single_auto0_alldrop01_tfreq200" ; N_BASIS="1" ; DE_OUT="single" ; LOSS_TYPE="dist"
#MODEL_PATH="./models/gorc-20200521-151537" ; LOG_SUFFIX="_w_freq_single_fix_auto0_alldrop01" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="dist"

#MODEL_PATH="./models/gorc-20200413-034855" ; LOG_SUFFIX="no_lin_w_freq_auto1_alldrop01" ; N_BASIS="1" ; DE_OUT="no" ; LOSS_TYPE="dist"
#MODEL_PATH="./models/gorc-20200413-034815" ; LOG_SUFFIX="no_lin_w_freq_auto1_alldrop01" ; N_BASIS="5" ; DE_OUT="no" ; LOSS_TYPE="dist"

#MODEL_PATH="./models/gorc-20200423-155956" ; LOG_SUFFIX="_w_freq_single_stable_auto_avg1_alldrop01_tfreq50" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="dist"
#MODEL_PATH="./models/gorc-20200517-102332" ; LOG_SUFFIX="_w_freq_single_stable_fix_auto_avg1_alldrop01" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="dist"
#MODEL_PATH="./models/gorc-20200430-010818" ; LOG_SUFFIX="_w_freq_single_stable_auto_avg1_alldrop01_tfreq" ; N_BASIS="3" ; DE_OUT="single" ; LOSS_TYPE="dist"
#MODEL_PATH="./models/gorc-20200423-155505" ; LOG_SUFFIX="_w_freq_single_stable_auto_avg1_alldrop01_tfreq50" ; N_BASIS="1" ; DE_OUT="single" ; LOSS_TYPE="dist"

#MODEL_PATH="./models/gorc-20200521-201555" ; LOG_SUFFIX="_GRU_w_freq_single_stable_fix_auto_avg1_alldrop01" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="dist" ; ENCODER="GRU"
#MODEL_PATH="./models/gorc-20200521-202530" ; LOG_SUFFIX="_GRU_w_freq_single_stable_fix_auto_avg1_alldrop01" ; N_BASIS="1" ; DE_OUT="single" ; LOSS_TYPE="dist" ; ENCODER="GRU"

#MODEL_PATH="./models/gorc-20200424-154712" ; LOG_SUFFIX="_sim_no_lin_auto_avg1_alldrop01" ; N_BASIS="5" ; DE_OUT="no" ; LOSS_TYPE="sim"
#MODEL_PATH="./models/gorc-20200424-154249" ; LOG_SUFFIX="_sim_no_lin_auto_avg1_alldrop01" ; N_BASIS="1" ; DE_OUT="no" ; LOSS_TYPE="sim"

#MODEL_PATH="./models/gorc-20200425-155800" ; LOG_SUFFIX="_single_stable_auto_avg1_alldrop01" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="dist"
#MODEL_PATH="./models/gorc-20200430-010628" ; LOG_SUFFIX="_single_stable_auto_avg1_alldrop01" ; N_BASIS="3" ; DE_OUT="single" ; LOSS_TYPE="dist"
#MODEL_PATH="./models/gorc-20200425-155846" ; LOG_SUFFIX="_single_stable_auto_avg1_alldrop01" ; N_BASIS="1" ; DE_OUT="single" ; LOSS_TYPE="dist"

#MODEL_PATH="./models/gorc-20200427-101205" ; LOG_SUFFIX="_sim_w_freq_no_lin_auto_avg1_alldrop01" ; N_BASIS="5" ; DE_OUT="no" ; LOSS_TYPE="sim"
MODEL_PATH="./models/gorc-20200430-012204" ; LOG_SUFFIX="_sim_w_freq_single_auto_avg1_alldrop01" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="sim"
#MODEL_PATH="./models/gorc-20200430-012204" ; LOG_SUFFIX="_sim_tnorm_w_freq_single_auto_avg1_alldrop01" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="sim_norm"
#MODEL_PATH="./models/gorc-20200427-101424" ; LOG_SUFFIX="_sim_w_freq_no_lin_auto_avg1_alldrop01" ; N_BASIS="1" ; DE_OUT="no" ; LOSS_TYPE="sim"
#MODEL_PATH="./models/gorc-20200427-101424" ; LOG_SUFFIX="_sim_tnorm_w_freq_no_lin_auto_avg1_alldrop01" ; N_BASIS="1" ; DE_OUT="no" ; LOSS_TYPE="sim_norm"
#MODEL_PATH="./models/gorc-20200427-164951" ; LOG_SUFFIX="_sim_w_freq_no_lin_auto_avg1_alldrop01" ; N_BASIS="3" ; DE_OUT="no" ; LOSS_TYPE="sim"

#MODEL_PATH="./models/gorc-20200521-202622" ; LOG_SUFFIX="_sim_GRU_w_freq_single_auto_avg1_alldrop01" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="sim" ; ENCODER="GRU"
#MODEL_PATH="./models/gorc-20200521-202744" ; LOG_SUFFIX="_sim_GRU_w_freq_single_auto_avg1_alldrop01" ; N_BASIS="1" ; DE_OUT="single" ; LOSS_TYPE="sim" ; ENCODER="GRU"

#MODEL_PATH="./models/gorc-20200427-102444" ; LOG_SUFFIX="_sim_w_freq_no_lin_alldrop01" ; N_BASIS="5" ; DE_OUT="no" ; LOSS_TYPE="sim"
#MODEL_PATH="./models/gorc-20200429-102522" ; LOG_SUFFIX="_sim_w_freq_single_alldrop01" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="sim"
#MODEL_PATH="./models/gorc-20200427-102459" ; LOG_SUFFIX="_sim_w_freq_no_lin_alldrop01" ; N_BASIS="1" ; DE_OUT="no" ; LOSS_TYPE="sim"

#MODEL_PATH="./models/gorc-20200427-103241" ; LOG_SUFFIX="_sim_w_freq_no_lin_no_user_alldrop01" ; N_BASIS="5" ; DE_OUT="no" ; LOSS_TYPE="sim"
#MODEL_PATH="./models/gorc-20200429-102525" ; LOG_SUFFIX="_sim_w_freq_single_no_user_alldrop01" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="sim"
#MODEL_PATH="./models/gorc-20200427-103213" ; LOG_SUFFIX="_sim_w_freq_no_lin_no_user_alldrop01" ; N_BASIS="1" ; DE_OUT="no" ; LOSS_TYPE="sim"

#MODEL_PATH="./models/gorc-20200517-103913" ; LOG_SUFFIX="_w_freq_single_max_norm_stable_auto_avg1_alldrop01_tfreq" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="dist"
#MODEL_PATH="./models/gorc-20200517-103853" ; LOG_SUFFIX="_w_freq_single_max_norm_stable_auto_avg1_alldrop01_tfreq" ; N_BASIS="1" ; DE_OUT="single" ; LOSS_TYPE="dist"

#MODEL_PATH="./models/gorc-20200517-015004" ; LOG_SUFFIX="_sim_allow_neg_w_freq_single_auto_avg1_alldrop01" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="sim"
#MODEL_PATH="./models/gorc-20200517-014947" ; LOG_SUFFIX="_sim_allow_neg_w_freq_single_auto_avg1_alldrop01" ; N_BASIS="1" ; DE_OUT="single" ; LOSS_TYPE="sim"

#MODEL_PATH="./models/gorc-20200517-015022" ; LOG_SUFFIX="_w_freq_single_max_1_stable_auto_avg1_alldrop01_tfreq" ; N_BASIS="5" ; DE_OUT="single" ; LOSS_TYPE="dist"
#MODEL_PATH="./models/gorc-20200517-015021" ; LOG_SUFFIX="_w_freq_single_max_1_stable_auto_avg1_alldrop01_tfreq" ; N_BASIS="1" ; DE_OUT="single" ; LOSS_TYPE="dist"

if [ $LOSS_TYPE == "dist" ]; then
    LOSS_ARG="--target_norm True --loss_type dist --inv_freq_w True"
    #LOSS_ARG="--target_norm True --loss_type dist"
    DE_COEFF="TRANS_old"
elif [ $LOSS_TYPE == "sim_norm" ]; then
    LOSS_ARG="--target_norm True --norm_basis_when_freezing True --loss_type sim --target_l2 1e-6 --inv_freq_w True"
    DE_COEFF="TRANS_two_heads"
else
    LOSS_ARG="--target_norm False --loss_type sim --target_l2 1e-6 --inv_freq_w True"
    #LOSS_ARG="--target_norm False --loss_type sim --target_l2 1e-6"
    DE_COEFF="TRANS_two_heads"
fi

#~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/gorc_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/gorc_uncased_min_5 --tensor_folder tensors_cold_0 --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic --store_dist user --cuda False
#~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/gorc_new_max_cbow_freq_4_dist.np -j ./gen_log/gorc_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np -d ./data/processed/gorc_uncased_min_5/ -t tensors_cold_0 -a 0.9 > ./eval_log/gorc_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_09_$LOG_SUFFIX
#~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/gorc_new_avg_cbow_freq_4_dist.np -j ./gen_log/gorc_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np -d ./data/processed/gorc_uncased_min_5/ -t tensors_cold_0 -a 0.8 > ./eval_log/gorc_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_avg_$LOG_SUFFIX

#~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/gorc_tag_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np --batch_size 10 --n_basis ${N_BASIS} --data ./data/processed/gorc_uncased_min_5 --tensor_folder tensors_cold_0 --coeff_opt max --loss_type $LOSS_TYPE --test_user False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic --store_dist tag --cuda False
#~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/gorc_tag_new_max_cbow_freq_4_dist.np -j ./gen_log/gorc_tag_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np -d ./data/processed/gorc_uncased_min_5/ -t tensors_cold_0 -e tag -a 0.9 > ./eval_log/gorc_tag_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_09_$LOG_SUFFIX
#~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/gorc_tag_new_avg_cbow_freq_4_dist.np -j ./gen_log/gorc_tag_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np -d ./data/processed/gorc_uncased_min_5/ -t tensors_cold_0 -e tag -a 0.8 > ./eval_log/gorc_tag_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_avg_$LOG_SUFFIX
#exit

#Link prediction
#~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/gorc_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX --de_output_layer ${DE_OUT}_dynamic --batch_size 10 --n_basis $N_BASIS --data ./data/processed/gorc_uncased_min_5 --tensor_folder tensors_cold_0 --coeff_opt max --loss_type $LOSS_TYPE --test_tag True --source_emsize 200 --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --en_model $ENCODER > ./eval_log/gorc_rec_test_trans_bsz200_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX
#exit

##~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/gorc_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX --de_output_layer ${DE_OUT}_dynamic --batch_size 10 --n_basis $N_BASIS --data ./data/processed/gorc_uncased_min_5 --tensor_folder tensors_cold_0 --coeff_opt max --loss_type $LOSS_TYPE --test_user False --test_tag True --source_emsize 200 --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF > ./eval_log/gorc_rec_test_trans_bsz200_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX
#exit

##~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/gorc_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX --de_output_layer ${DE_OUT}_dynamic --batch_size 10 --n_basis $N_BASIS --data ./data/processed/gorc_uncased_min_5 --tensor_folder tensors_cold_0 --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --tag_emb_file None --source_emsize 200 --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF > ./eval_log/gorc_rec_test_trans_bsz200_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX

~/anaconda3/bin/python src/main.py --batch_size 50 --save $MODEL_PATH --de_output_layer ${DE_OUT}_dynamic --continue_train --data ./data/processed/NeurIPS2020_final_gorc_uncased --source_emb_file ./resources/cbow_ACM_dim200_gorc_uncased_min_5.txt --de_model TRANS --en_model TRANS --encode_trans_layers 3 --trans_layers 3 --n_basis $N_BASIS --seed 111 --epochs 100 --tensor_folder tensors_cold --coeff_opt max --tag_w 1 --user_w 5 --rand_neg_method shuffle $LOSS_ARG --de_coeff_model $DE_COEFF --lr 0.0002 --target_emsize 100 --freeze_encoder_decoder True --loading_target_embedding False --always_save_model True --target_embedding_suffix _NeurIPS2020_final --log_file_name log_NeurIPS2020_final.txt
~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/NeurIPS2020_final_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np --tag_emb_file tag_emb_NeurIPS2020_final_always.pt --user_emb_file user_emb_NeurIPS2020_final_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/NeurIPS2020_final_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic --store_dist user --remove_testing_duplication False
#exit

#NeurIPS2019
#~/anaconda3/bin/python src/main.py --batch_size 50 --save $MODEL_PATH --de_output_layer ${DE_OUT}_dynamic --continue_train --data ./data/processed/NeurIPS2019_bid_score_gorc_uncased --source_emb_file ./resources/cbow_ACM_dim200_gorc_uncased_min_5.txt --de_model TRANS --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --n_basis $N_BASIS --seed 111 --epochs 100 --tensor_folder tensors_cold --coeff_opt max --tag_w 1 --user_w 5 --rand_neg_method shuffle $LOSS_ARG --de_coeff_model $DE_COEFF --lr 0.0002 --target_emsize 100 --freeze_encoder_decoder True --loading_target_embedding False --always_save_model True --target_embedding_suffix _NeurIPS2019 --log_file_name log_NeurIPS2019.txt

#~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/NeurIPS2019_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX --tag_emb_file tag_emb_NeurIPS2019_always.pt --user_emb_file user_emb_NeurIPS2019_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/NeurIPS2019_bid_score_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic > ./eval_log/NeurIPS2019_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX
#~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/NeurIPS2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}_bid_score.np --tag_emb_file tag_emb_NeurIPS2019_always.pt --user_emb_file user_emb_NeurIPS2019_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/NeurIPS2019_bid_score_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic --store_dist user
#~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/NeurIPS2019_max_cbow_freq_4_dist.np -j ./gen_log/NeurIPS2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}_bid_score.np -d ./data/processed/NeurIPS2019_bid_score_gorc_uncased/ > ./eval_log/NeurIPS2019_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_$LOG_SUFFIX
#~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/NeurIPS2019_avg_cbow_freq_4_dist.np -j ./gen_log/NeurIPS2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}_bid_score.np -d ./data/processed/NeurIPS2019_bid_score_gorc_uncased/ > ./eval_log/NeurIPS2019_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_avg_$LOG_SUFFIX
#exit

#ICLR2020
~/anaconda3/bin/python src/main.py --batch_size 50 --save $MODEL_PATH --de_output_layer ${DE_OUT}_dynamic --continue_train --data ./data/processed/ICLR2020_gorc_uncased --source_emb_file ./resources/cbow_ACM_dim200_gorc_uncased_min_5.txt --de_model TRANS --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --n_basis $N_BASIS --seed 111 --epochs 100 --tensor_folder tensors_cold --coeff_opt max --tag_w 1 --user_w 5 --rand_neg_method shuffle $LOSS_ARG --de_coeff_model $DE_COEFF --lr 0.0002 --target_emsize 100 --freeze_encoder_decoder True --loading_target_embedding False --always_save_model True --target_embedding_suffix _ICLR2020 --log_file_name log_ICLR2020.txt
~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/ICLR2020_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX --tag_emb_file tag_emb_ICLR2020_always.pt --user_emb_file user_emb_ICLR2020_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/ICLR2020_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic > ./eval_log/ICLR2020_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX
~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/ICLR2020_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np --tag_emb_file tag_emb_ICLR2020_always.pt --user_emb_file user_emb_ICLR2020_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/ICLR2020_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic --store_dist user
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/ICLR2020_new_max_cbow_freq_4_dist.np -j ./gen_log/ICLR2020_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np -d ./data/processed/ICLR2020_gorc_uncased/ > ./eval_log/ICLR2020_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_$LOG_SUFFIX
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/ICLR2020_new_avg_cbow_freq_4_dist.np -j ./gen_log/ICLR2020_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np -d ./data/processed/ICLR2020_gorc_uncased/ > ./eval_log/ICLR2020_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_avg_$LOG_SUFFIX

~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/ICLR2020_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX --tag_emb_file tag_emb_ICLR2020_always.pt --user_emb_file user_emb_ICLR2020_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/ICLR2020_bid_score_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic > ./eval_log/ICLR2020_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX
~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/ICLR2020_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}_bid_score.np --tag_emb_file tag_emb_ICLR2020_always.pt --user_emb_file user_emb_ICLR2020_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/ICLR2020_bid_score_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic --store_dist user
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/ICLR2020_new_max_cbow_freq_4_dist_bid_score.np -j ./gen_log/ICLR2020_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}_bid_score.np -d ./data/processed/ICLR2020_bid_score_gorc_uncased/ > ./eval_log/ICLR2020_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_$LOG_SUFFIX
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/ICLR2020_new_avg_cbow_freq_4_dist_bid_score.np -j ./gen_log/ICLR2020_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}_bid_score.np -d ./data/processed/ICLR2020_bid_score_gorc_uncased/ > ./eval_log/ICLR2020_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_avg_$LOG_SUFFIX


#UAI2019
~/anaconda3/bin/python src/main.py --batch_size 50 --save $MODEL_PATH --de_output_layer ${DE_OUT}_dynamic --continue_train --data ./data/processed/UAI2019_gorc_uncased --source_emb_file ./resources/cbow_ACM_dim200_gorc_uncased_min_5.txt --de_model TRANS --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --n_basis ${N_BASIS} --seed 111 --epochs 200 --tensor_folder tensors_cold --coeff_opt max --tag_w 1 --user_w 5 --rand_neg_method shuffle $LOSS_ARG --de_coeff_model $DE_COEFF --lr 0.0002 --target_emsize 100 --freeze_encoder_decoder True --loading_target_embedding False --always_save_model True --target_embedding_suffix _UAI2019 --log_file_name log_UAI2019.txt --log-interval 50 
~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/UAI2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX --tag_emb_file tag_emb_UAI2019_always.pt --user_emb_file user_emb_UAI2019_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/UAI2019_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic > ./eval_log/UAI2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX
~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/UAI2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np --tag_emb_file tag_emb_UAI2019_always.pt --user_emb_file user_emb_UAI2019_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/UAI2019_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic --store_dist user
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/UAI2019_new_max_cbow_freq_4_dist.np -j ./gen_log/UAI2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np -d ./data/processed/UAI2019_gorc_uncased/ > ./eval_log/UAI2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_$LOG_SUFFIX
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/UAI2019_new_avg_cbow_freq_4_dist.np -j ./gen_log/UAI2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np -d ./data/processed/UAI2019_gorc_uncased/ > ./eval_log/UAI2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_avg_$LOG_SUFFIX

~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/UAI2019_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX --tag_emb_file tag_emb_UAI2019_always.pt --user_emb_file user_emb_UAI2019_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/UAI2019_bid_score_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic > ./eval_log/UAI2019_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX
~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/UAI2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}_bid_score.np --tag_emb_file tag_emb_UAI2019_always.pt --user_emb_file user_emb_UAI2019_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/UAI2019_bid_score_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic --store_dist user
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/UAI2019_new_max_cbow_freq_4_dist_bid_score.np -j ./gen_log/UAI2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}_bid_score.np -d ./data/processed/UAI2019_bid_score_gorc_uncased/ > ./eval_log/UAI2019_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_$LOG_SUFFIX
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/UAI2019_new_avg_cbow_freq_4_dist_bid_score.np -j ./gen_log/UAI2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}_bid_score.np -d ./data/processed/UAI2019_bid_score_gorc_uncased/ > ./eval_log/UAI2019_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_avg_$LOG_SUFFIX

#ICLR2019
~/anaconda3/bin/python src/main.py --batch_size 50 --save $MODEL_PATH --de_output_layer ${DE_OUT}_dynamic --continue_train --data ./data/processed/ICLR2019_gorc_uncased --source_emb_file ./resources/cbow_ACM_dim200_gorc_uncased_min_5.txt --de_model TRANS --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --n_basis $N_BASIS --seed 111 --epochs 100 --tensor_folder tensors_cold --coeff_opt max --tag_w 1 --user_w 5 --rand_neg_method shuffle $LOSS_ARG --de_coeff_model $DE_COEFF --lr 0.0002 --target_emsize 100 --freeze_encoder_decoder True --loading_target_embedding False --always_save_model True --target_embedding_suffix _ICLR2019 --log_file_name log_ICLR2019.txt
~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/ICLR2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX --tag_emb_file tag_emb_ICLR2019_always.pt --user_emb_file user_emb_ICLR2019_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/ICLR2019_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic > ./eval_log/ICLR2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX
~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/ICLR2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np --tag_emb_file tag_emb_ICLR2019_always.pt --user_emb_file user_emb_ICLR2019_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/ICLR2019_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic --store_dist user
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/ICLR2019_max_cbow_freq_4_dist.np -j ./gen_log/ICLR2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np -d ./data/processed/ICLR2019_gorc_uncased/ > ./eval_log/ICLR2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_$LOG_SUFFIX
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/ICLR2019_avg_cbow_freq_4_dist.np -j ./gen_log/ICLR2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np -d ./data/processed/ICLR2019_gorc_uncased/ > ./eval_log/ICLR2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_avg_$LOG_SUFFIX

~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/ICLR2019_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX --tag_emb_file tag_emb_ICLR2019_always.pt --user_emb_file user_emb_ICLR2019_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/ICLR2019_bid_score_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic > ./eval_log/ICLR2019_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX
~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/ICLR2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}_bid_score.np --tag_emb_file tag_emb_ICLR2019_always.pt --user_emb_file user_emb_ICLR2019_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/ICLR2019_bid_score_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic --store_dist user
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/ICLR2019_max_cbow_freq_4_dist_bid_score.np -j ./gen_log/ICLR2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}_bid_score.np -d ./data/processed/ICLR2019_bid_score_gorc_uncased/ > ./eval_log/ICLR2019_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_$LOG_SUFFIX
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/ICLR2019_avg_cbow_freq_4_dist_bid_score.np -j ./gen_log/ICLR2019_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}_bid_score.np -d ./data/processed/ICLR2019_bid_score_gorc_uncased/ > ./eval_log/ICLR2019_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_avg_$LOG_SUFFIX

#ICLR2018
~/anaconda3/bin/python src/main.py --batch_size 50 --save $MODEL_PATH --de_output_layer ${DE_OUT}_dynamic --continue_train --data ./data/processed/ICLR2018_gorc_uncased --source_emb_file ./resources/cbow_ACM_dim200_gorc_uncased_min_5.txt --de_model TRANS --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --n_basis $N_BASIS --seed 111 --epochs 100 --tensor_folder tensors_cold --coeff_opt max --tag_w 1 --user_w 5 --rand_neg_method shuffle $LOSS_ARG --de_coeff_model $DE_COEFF --lr 0.0002 --target_emsize 100 --freeze_encoder_decoder True --loading_target_embedding False --always_save_model True --target_embedding_suffix _ICLR2018 --log_file_name log_ICLR2018.txt
~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/ICLR2018_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX --tag_emb_file tag_emb_ICLR2018_always.pt --user_emb_file user_emb_ICLR2018_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/ICLR2018_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic > ./eval_log/ICLR2018_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX
~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/ICLR2018_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np --tag_emb_file tag_emb_ICLR2018_always.pt --user_emb_file user_emb_ICLR2018_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/ICLR2018_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic --store_dist user
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/ICLR2018_max_cbow_freq_4_dist.np -j ./gen_log/ICLR2018_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np -d ./data/processed/ICLR2018_gorc_uncased/ > ./eval_log/ICLR2018_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_$LOG_SUFFIX
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/ICLR2018_avg_cbow_freq_4_dist.np -j ./gen_log/ICLR2018_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}.np -d ./data/processed/ICLR2018_gorc_uncased/ > ./eval_log/ICLR2018_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_avg_$LOG_SUFFIX

~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/ICLR2018_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX --tag_emb_file tag_emb_ICLR2018_always.pt --user_emb_file user_emb_ICLR2018_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/ICLR2018_bid_score_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic > ./eval_log/ICLR2018_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX
~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/ICLR2018_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}_bid_score.np --tag_emb_file tag_emb_ICLR2018_always.pt --user_emb_file user_emb_ICLR2018_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/ICLR2018_bid_score_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --en_model $ENCODER --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic --store_dist user
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/ICLR2018_max_cbow_freq_4_dist_bid_score.np -j ./gen_log/ICLR2018_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}_bid_score.np -d ./data/processed/ICLR2018_bid_score_gorc_uncased/ > ./eval_log/ICLR2018_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_$LOG_SUFFIX
~/anaconda3/bin/python src/testing/avg_baseline/merge_dist.py -i ./gen_log/ICLR2018_avg_cbow_freq_4_dist_bid_score.np -j ./gen_log/ICLR2018_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_${LOG_SUFFIX}_bid_score.np -d ./data/processed/ICLR2018_bid_score_gorc_uncased/ > ./eval_log/ICLR2018_bid_score_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_CBOW_avg_$LOG_SUFFIX


#~/anaconda3/bin/python src/main.py --batch_size 50 --save $MODEL_PATH --de_output_layer ${DE_OUT}_dynamic --continue_train --data ./data/processed/ICLR2020_gorc_uncased --source_emb_file ./resources/cbow_ACM_dim200_gorc_uncased_min_5.txt --de_model TRANS --en_model TRANS --encode_trans_layers 3 --trans_layers 3 --n_basis $N_BASIS --seed 111 --epochs 100 --tensor_folder tensors_cold --coeff_opt max --tag_w 1 --user_w 5 --rand_neg_method shuffle $LOSS_ARG --de_coeff_model $DE_COEFF --lr 0.0002 --target_emsize 100 --freeze_encoder_decoder True --loading_target_embedding False --always_save_model True --target_embedding_suffix _ICLR2020_switch --switch_user_tag_roles True --log_file_name log_ICLR2020_switch.txt
#~/anaconda3/bin/python src/recommend_test.py --checkpoint $MODEL_PATH --outf ./gen_log/ICLR2020_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX --tag_emb_file tag_emb_ICLR2020_switch_always.pt --user_emb_file user_emb_ICLR2020_switch_always.pt --batch_size 50 --n_basis ${N_BASIS} --data ./data/processed/ICLR2020_gorc_uncased --tensor_folder tensors_cold --coeff_opt max --loss_type $LOSS_TYPE --test_tag False --source_emsize 200 --encode_trans_layers 3 --trans_layers 3 --de_coeff_model $DE_COEFF --de_output_layer ${DE_OUT}_dynamic --switch_user_tag_roles True > ./eval_log/ICLR2020_rec_test_trans_bsz50_n${N_BASIS}_shuffle_uni_max_lr2e-4_$LOG_SUFFIX