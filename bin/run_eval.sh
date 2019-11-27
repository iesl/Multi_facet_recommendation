#!/bin/bash
module load python3/current
echo "code is running"

GLOVE_UPPER="./resources/glove.840B.300d_filtered_wiki2016.txt"
WORD2VEC_UPPER="./resources/word2vec_wiki2016_min100.txt"
DICT_UPPER="./data/processed/wiki2016_min100/dictionary_index"

STSB_TRAIN="./dataset_testing/STS/stsbenchmark/sts-train.csv"
STSB_DEV="./dataset_testing/STS/stsbenchmark/sts-dev.csv"
STSB_TEST="./dataset_testing/STS/stsbenchmark/sts-test.csv"
STS_2012_TRAIN="./dataset_testing/STS/sts_2012_train"
STS_2012_6_TEST="./dataset_testing/STS/sts_all_years_test"
STS_2012_TEST="./dataset_testing/STS/sts_test_year_2012"
STS_2013_TEST="./dataset_testing/STS/sts_test_year_2013"
STS_2014_TEST="./dataset_testing/STS/sts_test_year_2014"
STS_2015_TEST="./dataset_testing/STS/sts_test_year_2015"
STS_2016_TEST="./dataset_testing/STS/sts_test_year_2016"

#topic_file_name = "./gen_log/STS_dev_updated_glove_lc_elayer2_bsz200_ep6_linear_cosine.json"
#topic_file_name = "./gen_log/STS_dev_glove_lc_elayer2_bsz200_ep7_linear_cosine.json"
#topic_file_name = "./gen_log/STS_dev_glove_lc_elayer2_bsz200_ep7_posi_cosine.json"
#topic_file_name = "./gen_log/STS_dev_word2vec_lc_elayer1_bsz200_ep4.json"
#topic_file_name = "./gen_log/STS_dev_word2vec_lc_elayer1_bsz200_ep7_linear_cosine.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_word2vec_maxlc_bsz200_ep2_0.json"
#w_emb_file_name = "./resources/word2vec_wiki2016_min100.txt"
#topic_file_name = "./gen_log/STS_dev_wiki2016_glove_lc_bsz200_ep2_0.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_glove_trans_bsz200_ep1_1.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_glove_trans_bsz200_ep2_1.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_glove_trans_n20_bsz200_ep1_0.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_glove_trans_n20_bsz200_ep1_1_fix.json"
#topic_file_name = "./gen_log/STS_train_wiki2016_glove_trans_n10_bsz200_ep2_1.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_glove_trans_n3_bsz200_ep1_1.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_glove_trans_n1_bsz200_ep1_0.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_glove_trans_RMSProp_bsz200_ep1_0.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_glove_trans_no_connect_bsz200_ep2_0.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_word2vec_trans_n20_bsz200_ep1_0.json"
#w_emb_file_name = "./resources/word2vec_wiki2016_min100.txt"
#topic_file_name = "./gen_log/STS_dev_wiki2016_lex_crawl_trans_bsz200_ep2_1.json"
#w_emb_file_name = "./resources/lexvec_wiki2016_min100"

#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t <topic_file_name> -w <w_emb_file_name> -d <freq_file_name> -g <gt_file_name> -m <pc_mode> -p <path_to_pc>
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_train_wiki2016_word2vec_trans_n10_bsz200_ep2_0.json -w $WORD2VEC_UPPER -d $DICT_UPPER -g $STSB_TRAIN -m save -p ./gen_log/pc/pc_word2vec_sts_train_n10 > eval_log/STS_train_wiki2016_word2vec_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_word2vec_trans_n10_bsz200_ep2_0.json -w $WORD2VEC_UPPER -d $DICT_UPPER -g $STSB_DEV -m load -p ./gen_log/pc/pc_word2vec_sts_train_n10 > eval_log/STS_dev_wiki2016_word2vec_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_test_wiki2016_word2vec_trans_n10_bsz200_ep2_0.json -w $WORD2VEC_UPPER -d $DICT_UPPER -g $STSB_TEST -m load -p ./gen_log/pc/pc_word2vec_sts_train_n10 > eval_log/STS_test_wiki2016_word2vec_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012_train_wiki2016_word2vec_trans_n10_bsz200_ep2_0.json -w $WORD2VEC_UPPER -d $DICT_UPPER -g $STS_2012_TRAIN -m save -p ./gen_log/pc/pc_word2vec_sts_2012_train_n10 > eval_log/STS_2012_train_wiki2016_word2vec_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_word2vec_trans_n10_bsz200_ep2_0.json -w $WORD2VEC_UPPER -d $DICT_UPPER -g $STS_2012_TEST -m load -p ./gen_log/pc/pc_word2vec_sts_2012_train_n10 > eval_log/STS_2012_test_wiki2016_word2vec_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_word2vec_trans_n10_bsz200_ep2_0.json -w $WORD2VEC_UPPER -d $DICT_UPPER -g $STS_2013_TEST -m load -p ./gen_log/pc/pc_word2vec_sts_2012_train_n10 > eval_log/STS_2013_test_wiki2016_word2vec_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_word2vec_trans_n10_bsz200_ep2_0.json -w $WORD2VEC_UPPER -d $DICT_UPPER -g $STS_2014_TEST -m load -p ./gen_log/pc/pc_word2vec_sts_2012_train_n10 > eval_log/STS_2014_test_wiki2016_word2vec_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_word2vec_trans_n10_bsz200_ep2_0.json -w $WORD2VEC_UPPER -d $DICT_UPPER -g $STS_2015_TEST -m load -p ./gen_log/pc/pc_word2vec_sts_2012_train_n10 > eval_log/STS_2015_test_wiki2016_word2vec_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_word2vec_trans_n10_bsz200_ep2_0.json -w $WORD2VEC_UPPER -d $DICT_UPPER -g $STS_2016_TEST -m load -p ./gen_log/pc/pc_word2vec_sts_2012_train_n10 > eval_log/STS_2016_test_wiki2016_word2vec_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_word2vec_trans_n10_bsz200_ep2_0.json -w $WORD2VEC_UPPER -d $DICT_UPPER -g $STS_2012_6_TEST -m load -p ./gen_log/pc/pc_word2vec_sts_2012_train_n10 > eval_log/STS_2012-6_test_wiki2016_word2vec_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_train_wiki2016_glove_trans_n10_bsz200_ep2_1.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TRAIN -m save -p ./gen_log/pc/pc_sts_train_n10  > eval_log/STS_train_wiki2016_glove_trans_n10_bsz200_ep2_1
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_glove_trans_n10_bsz200_ep2_1.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_DEV -m load -p ./gen_log/pc/pc_sts_train_n10 > eval_log/STS_dev_wiki2016_glove_trans_n10_bsz200_ep2_1_pc_train
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_test_wiki2016_glove_trans_n10_bsz200_ep2_1.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TEST -m load -p ./gen_log/pc/pc_sts_train_n10 > eval_log/STS_test_wiki2016_glove_trans_n10_bsz200_ep2_1_pc_train
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_train_wiki2016_glove_trans_n10_bsz200_ep2_0_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TRAIN -m save -p ./gen_log/pc/pc_sts_train_n10_final  > eval_log/final_STS_train_wiki2016_glove_trans_n10_bsz200_ep2_0
~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012_train_wiki2016_glove_trans_n1_bsz200_ep2_2.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STS_2012_TRAIN -m save -p ./gen_log/pc/pc_sts_2012_train_n1  > eval_log/STS_2012_train_wiki2016_glove_trans_n1_bsz200_ep2_2
~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_glove_trans_n1_bsz200_ep2_2.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STS_2012_TEST -m load -p ./gen_log/pc/pc_sts_2012_train_n1  > eval_log/STS_2012_test_wiki2016_glove_trans_n1_bsz200_ep2_2
~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_glove_trans_n1_bsz200_ep2_2.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STS_2013_TEST -m load -p ./gen_log/pc/pc_sts_2012_train_n1  > eval_log/STS_2013_test_wiki2016_glove_trans_n1_bsz200_ep2_2
~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_glove_trans_n1_bsz200_ep2_2.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STS_2014_TEST -m load -p ./gen_log/pc/pc_sts_2012_train_n1  > eval_log/STS_2014_test_wiki2016_glove_trans_n1_bsz200_ep2_2
~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_glove_trans_n1_bsz200_ep2_2.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STS_2015_TEST -m load -p ./gen_log/pc/pc_sts_2012_train_n1  > eval_log/STS_2015_test_wiki2016_glove_trans_n1_bsz200_ep2_2
~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_glove_trans_n1_bsz200_ep2_2.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STS_2016_TEST -m load -p ./gen_log/pc/pc_sts_2012_train_n1  > eval_log/STS_2016_test_wiki2016_glove_trans_n1_bsz200_ep2_2
~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_glove_trans_n1_bsz200_ep2_2.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STS_2012_6_TEST -m load -p ./gen_log/pc/pc_sts_2012_train_n1  > eval_log/STS_2012-6_test_wiki2016_glove_trans_n1_bsz200_ep2_2
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012_train_wiki2016_glove_trans_n10_bsz200_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STS_2012_TRAIN -m save -p ./gen_log/pc/pc_sts_2012_train_n10  > eval_log/STS_2012_train_wiki2016_glove_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_glove_trans_n10_bsz200_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STS_2012_TEST -m load -p ./gen_log/pc/pc_sts_2012_train_n10  > eval_log/STS_2012_test_wiki2016_glove_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_glove_trans_n10_bsz200_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STS_2013_TEST -m self -p ./gen_log/pc/pc_sts_2012_train_n10  > eval_log/STS_2013_test_wiki2016_glove_trans_n10_bsz200_ep2_0_self
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_glove_trans_n10_bsz200_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STS_2014_TEST -m self -p ./gen_log/pc/pc_sts_2012_train_n10  > eval_log/STS_2014_test_wiki2016_glove_trans_n10_bsz200_ep2_0_self
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_glove_trans_n10_bsz200_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STS_2013_TEST -m load -p ./gen_log/pc/pc_sts_2012_train_n10  > eval_log/STS_2013_test_wiki2016_glove_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_glove_trans_n10_bsz200_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STS_2014_TEST -m load -p ./gen_log/pc/pc_sts_2012_train_n10  > eval_log/STS_2014_test_wiki2016_glove_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_glove_trans_n10_bsz200_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STS_2015_TEST -m load -p ./gen_log/pc/pc_sts_2012_train_n10  > eval_log/STS_2015_test_wiki2016_glove_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_glove_trans_n10_bsz200_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STS_2016_TEST -m load -p ./gen_log/pc/pc_sts_2012_train_n10  > eval_log/STS_2016_test_wiki2016_glove_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_2012-6_test_wiki2016_glove_trans_n10_bsz200_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STS_2012_6_TEST -m load -p ./gen_log/pc/pc_sts_2012_train_n10  > eval_log/STS_2012-6_test_wiki2016_glove_trans_n10_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_glove_trans_n10_bsz200_ep2_0_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_DEV -m load -p ./gen_log/pc/pc_sts_train_n10_final > eval_log/final_STS_dev_wiki2016_glove_trans_n10_bsz200_ep2_0_pc_train
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_glove_trans_n10_bsz200_ep2_0_final_update.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_DEV -m load -p ./gen_log/pc/pc_sts_train_n10_final > eval_log/final_STS_dev_wiki2016_glove_trans_n10_bsz200_ep2_0_update_pc_old
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_glove_trans_n10_bsz200_ep2_0_kmeans.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_DEV -m load -p ./gen_log/pc/pc_sts_train_n10_final > eval_log/final_STS_dev_wiki2016_glove_trans_n10_bsz200_ep2_0_kmeans_pc_old
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_test_wiki2016_glove_trans_n10_bsz200_ep2_0_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TEST -m load -p ./gen_log/pc/pc_sts_train_n10_final  > eval_log/final_STS_test_wiki2016_glove_trans_n10_bsz200_ep2_0_pc_train
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_train_wiki2016_glove_trans_n3_bsz200_ep2_0_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TRAIN -m save -p ./gen_log/pc/pc_sts_train_n3_final  > eval_log/final_STS_train_wiki2016_glove_trans_n3_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_glove_trans_n3_bsz200_ep2_0_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_DEV -m load -p ./gen_log/pc/pc_sts_train_n3_final > eval_log/final_STS_dev_wiki2016_glove_trans_n3_bsz200_ep2_0_pc_train
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_test_wiki2016_glove_trans_n3_bsz200_ep2_0_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TEST -m load -p ./gen_log/pc/pc_sts_train_n3_final  > eval_log/final_STS_test_wiki2016_glove_trans_n3_bsz200_ep2_0_pc_train
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_train_wiki2016_glove_trans_n20_bsz200_ep2_0_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TRAIN -m save -p ./gen_log/pc/pc_sts_train_n20_final  > eval_log/final_STS_train_wiki2016_glove_trans_n20_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_glove_trans_n20_bsz200_ep2_0_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_DEV -m load -p ./gen_log/pc/pc_sts_train_n20_final > eval_log/final_STS_dev_wiki2016_glove_trans_n20_bsz200_ep2_0_pc_train
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_test_wiki2016_glove_trans_n20_bsz200_ep2_0_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TEST -m load -p ./gen_log/pc/pc_sts_train_n20_final  > eval_log/final_STS_test_wiki2016_glove_trans_n20_bsz200_ep2_0_pc_train
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_train_wiki2016_glove_trans_n50_bsz200_ep2_0_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TRAIN -m save -p ./gen_log/pc/pc_sts_train_n50_final  > eval_log/final_STS_train_wiki2016_glove_trans_n50_bsz200_ep1_1
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_glove_trans_n50_bsz200_ep2_0_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_DEV -m load -p ./gen_log/pc/pc_sts_train_n50_final > eval_log/final_STS_dev_wiki2016_glove_trans_n50_bsz200_ep1_1_pc_train
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_test_wiki2016_glove_trans_n50_bsz200_ep2_0_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TEST -m load -p ./gen_log/pc/pc_sts_train_n50_final  > eval_log/final_STS_test_wiki2016_glove_trans_n50_bsz200_ep1_1_pc_train
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_train_wiki2016_glove_trans_n1_bsz200_ep2_2_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TRAIN -m save -p ./gen_log/pc/pc_sts_train_n1_final  > eval_log/final_STS_train_wiki2016_glove_trans_n1_bsz200_ep2_2
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_glove_trans_n1_bsz200_ep2_2_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_DEV -m load -p ./gen_log/pc/pc_sts_train_n1_final > eval_log/final_STS_dev_wiki2016_glove_trans_n1_bsz200_ep2_2_pc_train
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_test_wiki2016_glove_trans_n1_bsz200_ep2_2_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TEST -m load -p ./gen_log/pc/pc_sts_train_n1_final  > eval_log/final_STS_test_wiki2016_glove_trans_n1_bsz200_ep2_2_pc_train
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_train_wiki2016_glove_LSTM_n10_bsz200_ep2_2_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TRAIN -m save -p ./gen_log/pc/pc_sts_train_n10_LSTM_final  > eval_log/final_STS_train_wiki2016_glove_LSTM_n10_bsz200_ep2_2
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_glove_LSTM_n10_bsz200_ep2_2_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_DEV -m load -p ./gen_log/pc/pc_sts_train_n10_LSTM_final > eval_log/final_STS_dev_wiki2016_glove_LSTM_n10_bsz200_ep2_2_pc_train
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t /iesl/canvas/amolagrawal/resultsemnlp/sts_train_words.json  -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TRAIN -m save -p ./gen_log/pc/pc_sts_train_Amol_word  > eval_log/Amol_sts_train_word
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t /iesl/canvas/amolagrawal/resultsemnlp/sts_dev_words.json  -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_DEV -m load -p ./gen_log/pc/pc_sts_train_Amol_word > eval_log/Amol_sts_dev_word
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t /iesl/canvas/amolagrawal/resultsemnlp/sts_test_words.json  -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TEST -m load -p ./gen_log/pc/pc_sts_train_Amol_word > eval_log/Amol_sts_test_word
#-m load -p ./gen_log/pc/pc_sts_train_n10_LSTM_final > eval_log/final_STS_dev_wiki2016_glove_LSTM_n10_bsz200_ep2_2_pc_train
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_test_wiki2016_glove_LSTM_n10_bsz200_ep2_2_final.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TEST -m load -p ./gen_log/pc/pc_sts_train_n10_LSTM_final  > eval_log/final_STS_test_wiki2016_glove_LSTM_n10_bsz200_ep2_2_pc_train

#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_glove_trans_n3_bsz200_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_DEV > eval_log/STS_dev_wiki2016_glove_trans_n3_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_glove_trans_n5_bsz200_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_DEV > eval_log/STS_dev_wiki2016_glove_trans_n5_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_glove_trans_n5_bsz200_no_connect_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_DEV > eval_log/STS_dev_wiki2016_glove_trans_n5_bsz200_no_connect_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_glove_trans_n1_bsz200_ep3_0.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_DEV > eval_log/STS_dev_wiki2016_glove_trans_n1_bsz200_ep3_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_word2vec_trans_n20_bsz200_ep2_0.json -w ./resources/word2vec_wiki2016_min100.txt -d $DICT_UPPER -g $STSB_DEV > eval_log/STS_dev_wiki2016_word2vec_trans_n20_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_glove_trans_n20_bsz200_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_DEV > eval_log/STS_dev_wiki2016_glove_trans_n20_bsz200_ep2_0

WiC_TRAIN_F="dataset_testing/WiC_dataset/train/train.data.txt"
WiC_TRAIN_G="dataset_testing/WiC_dataset/train/train.gold.txt"
WiC_DEV_F="dataset_testing/WiC_dataset/dev/dev.data.txt"
WiC_DEV_G="dataset_testing/WiC_dataset/dev/dev.gold.txt"

#topic_file_name = "./gen_log/WiC_dev_wiki2016_glove_trans_n20_bsz200_ep1_1.json"
#topic_file_name = "./gen_log/WiC_dev_wiki2016_glove_trans_n3_bsz200_ep1_1.json"
#topic_file_name = "./gen_log/WiC_dev_wiki2016_glove_trans_n1_bsz200_ep1_0.json"
#topic_file_name = "./gen_log/WiC_dev_wiki2016_glove_trans_no_connect_bsz200_ep2_0.json"
#w_emb_file_name = "./resources/word2vec_wiki2016_min100.txt"
#w_emb_file_name = "./resources/lexvec_wiki2016_min100"

#~/anaconda3/bin/python src/testing/sim/WiC_eval.py -t <topic_file_name> -w <w_emb_file_name> -d <freq_file_name> -f <feature_file_name> -g <gt_file_name>
#~/anaconda3/bin/python src/testing/sim/WiC_eval.py -t ./gen_log/WiC_dev_wiki2016_glove_trans_n10_bsz200_ep2_1.json -w $GLOVE_UPPER -d $DICT_UPPER -f $WiC_DEV_F -g $WiC_DEV_G > eval_log/WiC_dev_wiki2016_glove_trans_n10_bsz200_ep2_1
#~/anaconda3/bin/python src/testing/sim/WiC_eval.py -t ./gen_log/WiC_dev_wiki2016_glove_trans_n3_bsz200_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -f $WiC_DEV_F -g $WiC_DEV_G > eval_log/WiC_dev_wiki2016_glove_trans_n3_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/WiC_eval.py -t ./gen_log/WiC_dev_wiki2016_glove_trans_n5_bsz200_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -f $WiC_DEV_F -g $WiC_DEV_G > eval_log/WiC_dev_wiki2016_glove_trans_n5_bsz200_ep2_0
#~/anaconda3/bin/python src/testing/sim/WiC_eval.py -t ./gen_log/WiC_dev_wiki2016_glove_trans_n5_bsz200_no_connect_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -f $WiC_DEV_F -g $WiC_DEV_G > eval_log/WiC_dev_wiki2016_glove_trans_n5_bsz200_no_connect_ep2_0
#~/anaconda3/bin/python src/testing/sim/WiC_eval.py -t ./gen_log/WiC_dev_wiki2016_glove_trans_n1_bsz200_ep3_0.json -w $GLOVE_UPPER -d $DICT_UPPER -f $WiC_DEV_F -g $WiC_DEV_G > eval_log/WiC_dev_wiki2016_glove_trans_n1_bsz200_ep3_0
#~/anaconda3/bin/python src/testing/sim/WiC_eval.py -t ./gen_log/WiC_dev_wiki2016_word2vec_trans_n20_bsz200_ep2_0.json -w ./resources/word2vec_wiki2016_min100.txt -d $DICT_UPPER -f $WiC_DEV_F -g $WiC_DEV_G > eval_log/WiC_dev_wiki2016_word2vec_trans_n20_bsz200_ep2_0


WORD2VEC_LOWER="./resources/word2vec_sg_wiki2016_lower_min100_w5_s5.txt"
GLOVE_LOWER="./resources/glove.42B.300d_filtered_wiki2016_nchunk_lower_min100.txt"
DICT_PHRASE_LOWER="./data/processed/wiki2016_nchunk_lower_min100/dictionary_index"

#topic_file_name = "./gen_log/phrase_train_wiki2016_glove_trans_bsz200_ep1_3.json"
#topic_file_name = "./gen_log/phrase_train_wiki2016_glove_trans_bsz200_no_connect_ep1_3.json"
#topic_file_name = "./gen_log/phrase_train_wiki2016_glove_trans_bsz200_no_connect_no_stop_ep1_3.json"
#topic_file_name = "./gen_log/phrase_train_wiki2016_glove_trans_no_stop_bsz200_ep1_3.json"
#topic_file_name = "./gen_log/phrase_train_wiki2016_glove_trans_bsz200_no_connect_ep1_5.json"
#topic_file_name = "./gen_log/phrase_train_wiki2016_glove_trans_n1_bsz200_no_connect_ep1_2.json"
#topic_file_name = "./gen_log/phrase_train_wiki2016_glove_trans_n1_bsz200_no_connect_no_stop_ep1_2.json"
#topic_file_name = "./gen_log/phrase_train_wiki2016_lex_enwiki_trans_d400_bsz400_ep1_5.json"
#embedding_file_name = embedding_dir + "lexvec_enwiki_wiki2016_min100"

#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t <topic_file_name> -w <w_emb_file_name> -d <freq_file_name> -g <train_or_test> -l <upper_emb_to_lower>
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_train_wiki2016_glove_trans5_bsz1000_ep1_29.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g train -l 0 > eval_log/phrase_train_wiki2016_glove_trans5_bsz1000_ep1_29
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_train_wiki2016_glove_trans55_connect_bsz1000_ep1_29.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g train -l 0 > eval_log/phrase_train_wiki2016_glove_trans55_connect_bsz1000_ep1_29
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_train_wiki2016_glove_trans_no_stop_bsz200_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g train -l 0 > eval_log/phrase_train_wiki2016_glove_trans_no_stop_bsz200_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_train_wiki2016_glove_trans_n3_no_stop_no_connect_bsz200_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g train -l 0 > eval_log/phrase_train_wiki2016_glove_trans_n3_no_stop_no_connect_bsz200_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_train_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz200_ep1_2.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g train -l 0 > eval_log/phrase_train_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz200_ep1_2
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_train_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz200_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g train -l 0 > eval_log/phrase_train_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz200_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_train_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g train -l 0 > eval_log/phrase_train_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_test_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz200_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g test -l 0 > eval_log/phrase_test_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz200_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_test_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g test -l 0 > eval_log/phrase_test_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_test_wiki2016_word2vec_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7.json -w $WORD2VEC_LOWER -d $DICT_PHRASE_LOWER -g test -l 0 > eval_log/phrase_test_wiki2016_word2vec_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_test_wiki2016_word2vec_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep1_9.json -w $WORD2VEC_LOWER -d $DICT_PHRASE_LOWER -g test -l 0 > eval_log/phrase_test_wiki2016_word2vec_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep1_9
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_BiRD_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g BiRD -l 0 > eval_log/phrase_BiRD_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_BiRD_wiki2016_word2vec_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7.json -w $WORD2VEC_LOWER -d $DICT_PHRASE_LOWER -g BiRD -l 0 > eval_log/phrase_BiRD_wiki2016_word2vec_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_BiRD_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7_update.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g BiRD -l 0 > eval_log/phrase_BiRD_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7_update
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_BiRD_wiki2016_glove_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep2_1_update.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g BiRD -l 0 > eval_log/phrase_BiRD_wiki2016_glove_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep2_1_update
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_BiRD_wiki2016_word2vec_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep1_9.json -w $WORD2VEC_LOWER -d $DICT_PHRASE_LOWER -g BiRD -l 0 > eval_log/phrase_BiRD_wiki2016_word2vec_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep1_9
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_WikiSRS_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g WikiSRS -l 0 > eval_log/phrase_WikiSRS_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_WikiSRS_wiki2016_word2vec_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7.json -w $WORD2VEC_LOWER -d $DICT_PHRASE_LOWER -g WikiSRS -l 0 > eval_log/phrase_WikiSRS_wiki2016_word2vec_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_WikiSRS_wiki2016_word2vec_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep1_9.json -w $WORD2VEC_LOWER -d $DICT_PHRASE_LOWER -g WikiSRS -l 0 > eval_log/phrase_WikiSRS_wiki2016_word2vec_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep1_9
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_test_wiki2016_glove_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep2_5_correct.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g test -l 0 > eval_log/phrase_test_wiki2016_glove_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep2_5_correct
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_BiRD_wiki2016_glove_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep2_5.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g BiRD -l 0 > eval_log/phrase_BiRD_wiki2016_glove_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep2_5
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_WikiSRS_wiki2016_glove_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep2_5.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g WikiSRS -l 0 > eval_log/phrase_WikiSRS_wiki2016_glove_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep2_5
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_train_wiki2016_glove_trans_n3_stop_bsz200_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g train -l 0 > eval_log/phrase_train_wiki2016_glove_trans_n3_stop_bsz200_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_train_wiki2016_glove_trans_n3_stop_no_connect_bsz200_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g train -l 0 > eval_log/phrase_train_wiki2016_glove_trans_n3_stop_no_connect_bsz200_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_train_wiki2016_glove_trans_n1_bsz200_no_connect_ep1_20.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g train -l 0 > eval_log/phrase_train_wiki2016_glove_trans_n1_bsz200_no_connect_ep1_20
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_train_wiki2016_glove_trans_n1_bsz200_no_connect_no_stop_ep1_17.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g train -l 0 > eval_log/phrase_train_wiki2016_glove_trans_n1_bsz200_no_connect_no_stop_ep1_17
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_train_wiki2016_glove_trans_bsz200_no_connect_no_stop_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g train -l 0 > eval_log/phrase_train_wiki2016_glove_trans_bsz200_no_connect_no_stop_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_train_wiki2016_glove_trans_bsz200_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g train -l 0 > eval_log/phrase_train_wiki2016_glove_trans_bsz200_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_train_wiki2016_glove_trans_bsz200_no_connect_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g train -l 0 > eval_log/phrase_train_wiki2016_glove_trans_bsz200_no_connect_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/phrase_eval.py -t ./gen_log/phrase_train_wiki2016_glove_trans_e5d1_drop05_n1_no_stop_no_connect_bsz1000_ep1_3.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g train -l 0 > eval_log/phrase_train_wiki2016_glove_trans_e5d1_drop05_n1_no_stop_no_connect_bsz1000_ep1_3



#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t <topic_file_name> -w <w_emb_file_name> -d <freq_file_name> -g <train_or_test> -l <upper_emb_to_lower>
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_val_wiki2016_glove_trans2_no_stop_bsz200_ep1_8.json  -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g val -l 0 
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_test_wiki2016_glove_trans2_no_stop_bsz200_ep1_8.json  -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g test -l 0 
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_val_wiki2016_glove_trans_bsz200_no_connect_no_stop_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g val -l 0 > eval_log/phrase_hyper_val_wiki2016_glove_trans_bsz200_no_connect_no_stop_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_val_wiki2016_glove_trans_bsz200_no_connect_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g val -l 0 > eval_log/phrase_hyper_val_wiki2016_glove_trans_bsz200_no_connect_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_val_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz200_ep1_2.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g val -l 0 > eval_log/phrase_hyper_val_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz200_ep1_2

#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_val_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz200_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g val -l 0 > eval_log/phrase_hyper_val_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz200_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_val_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g val -l 0 > eval_log/phrase_hyper_val_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_test_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g test -l 0 > eval_log/phrase_hyper_test_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_test_wiki2016_word2vec_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7.json -w $WORD2VEC_LOWER -d $DICT_PHRASE_LOWER -g test -l 0 > eval_log/phrase_hyper_test_wiki2016_word2vec_trans_e5d2_drop05_n10_no_stop_no_connect_bsz1000_ep1_7
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_val_wiki2016_glove_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep2_5_correct.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g val -l 0 > eval_log/phrase_hyper_val_wiki2016_glove_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep2_5_correct
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_test_wiki2016_glove_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep2_5_correct.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g test -l 0 > eval_log/phrase_hyper_test_wiki2016_glove_trans_e5d2_drop05_n1_no_stop_no_connect_bsz1000_ep2_5_correct

#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_val_wiki2016_glove_trans_n3_stop_no_connect_bsz200_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g val -l 0 > eval_log/phrase_hyper_val_wiki2016_glove_trans_n3_stop_no_connect_bsz200_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_val_wiki2016_glove_trans_n3_no_stop_no_connect_bsz200_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g val -l 0 > eval_log/phrase_hyper_val_wiki2016_glove_trans_n3_no_stop_no_connect_bsz200_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_val_wiki2016_glove_trans5_bsz1000_ep1_29.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g val -l 0 > eval_log/phrase_hyper_val_wiki2016_glove_trans5_bsz1000_ep1_29
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_val_wiki2016_glove_trans_bsz200_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g val -l 0 > eval_log/phrase_hyper_val_wiki2016_glove_trans_bsz200_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_val_wiki2016_glove_trans_n1_bsz200_no_connect_no_stop_ep1_17.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g val -l 0 > eval_log/phrase_hyper_val_wiki2016_glove_trans_n1_bsz200_no_connect_no_stop_ep1_17
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_val_wiki2016_glove_trans_n1_bsz200_no_connect_ep1_20.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g val -l 0 > eval_log/phrase_hyper_val_wiki2016_glove_trans_n1_bsz200_no_connect_ep1_20

#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_word_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz200_ep1_2.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g word -l 0 > eval_log/phrase_hyper_word_wiki2016_glove_trans_e5d2_drop05_n10_no_stop_no_connect_bsz200_ep1_2
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_word_wiki2016_glove_trans2_no_stop_bsz200_ep1_8.json  -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g word -l 0 
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_word_wiki2016_glove_trans2_no_connect_no_stop_bsz200_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g word -l 0 > eval_log/phrase_hyper_word_wiki2016_glove_trans2_no_connect_no_stop_bsz200_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_word_wiki2016_glove_trans5_bsz1000_ep1_29.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g word -l 0 > eval_log/phrase_hyper_word_wiki2016_glove_trans5_bsz1000_ep1_29
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_word_wiki2016_glove_trans_bsz200_no_connect_ep1_8.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g word -l 0 > eval_log/phrase_hyper_word_wiki2016_glove_trans_bsz200_no_connect_ep1_8
#~/anaconda3/bin/python src/testing/sim_word/hyper_eval.py -t ./gen_log/phrase_hyper_word_wiki2016_glove_trans_n1_bsz200_no_connect_no_stop_ep1_17.json -w $GLOVE_LOWER -d $DICT_PHRASE_LOWER -g word -l 0 > eval_log/phrase_hyper_word_wiki2016_glove_trans_n1_bsz200_no_connect_no_stop_ep1_17
