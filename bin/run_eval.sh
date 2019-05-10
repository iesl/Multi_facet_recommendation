#!/bin/bash
module load python3/current
echo "code is running"

GLOVE_UPPER="./resources/glove.840B.300d_filtered_wiki2016.txt"
DICT_UPPER="./data/processed/wiki2016_min100/dictionary_index"

STSB_TRAIN="./dataset_testing/STS/stsbenchmark/sts-train.csv"
STSB_DEV="./dataset_testing/STS/stsbenchmark/sts-dev.csv"

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

#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t <topic_file_name> -w <w_emb_file_name> -d <freq_file_name> -g <gt_file_name>
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_train_wiki2016_glove_trans_n10_bsz200_ep2_1.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_TRAIN > eval_log/STS_train_wiki2016_glove_trans_n10_bsz200_ep2_1
#~/anaconda3/bin/python src/testing/sim/STS_eval.py -t ./gen_log/STS_dev_wiki2016_glove_trans_n3_bsz200_ep2_0.json -w $GLOVE_UPPER -d $DICT_UPPER -g $STSB_DEV > eval_log/STS_dev_wiki2016_glove_trans_n3_bsz200_ep2_0

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

