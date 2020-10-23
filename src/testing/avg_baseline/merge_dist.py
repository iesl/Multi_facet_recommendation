import sys
sys.path.insert(0, sys.path[0]+'/../../')

import utils_testing
import utils
import torch
import numpy as np
import getopt

dist_file_2 = ''
testing_dir = ''

#merge_alpha = 1
#merge_alpha = 0.7
merge_alpha = 0.8
#merge_alpha = 0.9
#merge_alpha = 0.5
#merge_alpha = 0.2
#merge_alpha = 0

eval_target = 'user'
tensor_folder_name = 'tensors_cold'

help_msg = '-i <dist_file_1> -j <dist_file_2> -d <testing_dir> -t <tensor_folder_name> -e <eval_target> -a <merge_alpha>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:j:d:t:e:a:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-i"):
        dist_file_1 = arg
    elif opt in ("-j"):
        dist_file_2 = arg
    elif opt in ("-d"):
        testing_dir = arg
    elif opt in ("-t"):
        tensor_folder_name = arg
    elif opt in ("-e"):
        eval_target = arg
    elif opt in ("-a"):
        merge_alpha = float(arg)

input_dict_path = testing_dir + "/feature/dictionary_index"
if eval_target == 'user':
    user_dict_path = testing_dir + "/user/dictionary_index"
elif eval_target == 'tag':
    tag_dict_path = testing_dir + "/tag/dictionary_index"
submission_data_file = testing_dir + '/'+tensor_folder_name+'/test.pt'

#dist_file_1 = "./gen_log/ICLR2020_spector_avg_dist.np"
#dist_file_1 = "./gen_log/ICLR2020_spector_max_dist.np"
#dist_file_1 = "./gen_log/NeurIPS2019_spector_avg_dist.np"
#dist_file_2 = "./gen_log/NeurIPS2019_avg_cbow_freq_4_dist.np"
#dist_file_1 = "./gen_log/NeurIPS2019_spector_max_dist.np"
#dist_file_2 = "./gen_log/NeurIPS2019_max_cbow_freq_4_dist.np"
#dist_file_1 = "./gen_log/NeurIPS2019_spector_max_dist.np"
#dist_file_1 = "./gen_log/NeurIPS2019_ELMo_dist.np"
#dist_file_1 = "./gen_log/NeurIPS2019_bm25_dist.np"
#dist_file_1 = "./gen_log/NeurIPS2019_TPMS_dist.np"
#dist_file_1 = "./gen_log/ICLR2019_TPMS_dist.np"
#dist_file_1 = "./gen_log/ICLR2019_bid_score_TPMS_dist.np"
#dist_file_1 = "./gen_log/ICLR2019_ELMo_dist.np"
#dist_file_1 = "./gen_log/ICLR2019_bid_score_ELMo_dist.np"
#dist_file_1 = "./gen_log/ICLR2020_Carlos_max.np"
#dist_file_1 = "./gen_log/ICLR2020_bid_score_Carlos_max.np"
#dist_file_1 = "./gen_log/ICLR2020_Carlos_avg.np"
#dist_file_1 = "./gen_log/ICLR2020_ELMo_dist.np"
#dist_file_1 = "./gen_log/ICLR2020_bid_score_ELMo_dist.np"
#dist_file_1 = "./gen_log/ICLR2020_new_max_cbow_freq_4_dist.np"
#dist_file_1 = "./gen_log/ICLR2020_new_avg_cbow_freq_4_dist.np"

#dist_file_2 = "./gen_log/ICLR2020_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4_dist_w_freq_single_right_type.np"
#dist_file_2 = "./gen_log/ICLR2020_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4_dist_w_freq_single_right_type.np"
#dist_file_2 = "./gen_log/ICLR2020_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4_dist_GRU_w_freq_single_right_type.np"
#dist_file_2 = "./gen_log/ICLR2020_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4_dist_GRU_w_freq_single_right_type.np"

#dist_file_2 = "./gen_log/ICLR2020_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4_w_freq.np"
#dist_file_2 = "./gen_log/ICLR2020_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4.np"
#dist_file_2 = "./gen_log/ICLR2020_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4_sim_w_freq.np"
#dist_file_2 = "./gen_log/ICLR2020_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4_sim_w_freq_no_lin.np"
#dist_file_2 = "./gen_log/ICLR2020_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4_no_lin_auto1.np"
#dist_file_2 = "./gen_log/ICLR2020_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4_no_lin_auto1.np"
#dist_file_2 = "./gen_log/ICLR2020_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4_no_lin_w_freq_auto1_alldrop01.np"
#dist_file_2 = "./gen_log/ICLR2020_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4_no_lin_w_freq_auto1_alldrop01.np"
#dist_file_2 = "./gen_log/ICLR2020_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4__w_freq_single_stable_auto_avg1_alldrop01_tfreq.np"
#dist_file_2 = "./gen_log/ICLR2020_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4__w_freq_single_stable_auto_avg1_alldrop01_tfreq.np"
#dist_file_2 = "./gen_log/NeurIPS2019_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4__w_freq_single_stable_auto_avg1_alldrop01_tfreq_bid_score.np"
#dist_file_2 = "./gen_log/NeurIPS2019_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4__w_freq_single_stable_auto_avg1_alldrop01_tfreq_bid_score.np"

#dist_file_1 = "./gen_log/ICLR2020_max_cbow_freq_4_dist_bid_score.np"
#dist_file_1 = "./gen_log/ICLR2020_avg_cbow_freq_4_dist_bid_score.np"
#dist_file_2 = "./gen_log/ICLR2020_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4_no_lin_auto1_bid_score.np"
#dist_file_2 = "./gen_log/ICLR2020_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4_no_lin_auto1_bid_score.np"

#dist_file_1 = "./gen_log/UAI2019_ELMo_dist.np"
#dist_file_1 = "./gen_log/UAI2019_TPMS_dist.np"
#dist_file_1 = "./gen_log/UAI2019_bid_score_ELMo_dist.np"
#dist_file_1 = "./gen_log/UAI2019_bid_score_TPMS_dist.np"
#dist_file_1 = "./gen_log/UAI2019_new_avg_cbow_freq_4_dist.np"
#dist_file_1 = "./gen_log/UAI2019_new_max_cbow_freq_4_dist.np"
#dist_file_2 = "./gen_log/UAI2019_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4_w_freq.np"
#dist_file_2 = "./gen_log/UAI2019_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4.np"
#dist_file_2 = "./gen_log/UAI2019_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4_sim_w_freq.np"
#dist_file_2 = "./gen_log/UAI2019_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4_sim_w_freq_no_lin.np"
#aist_file_2 = "./gen_log/UAI2019_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4_no_lin_auto1.np"
#dist_file_2 = "./gen_log/UAI2019_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4_no_lin_auto1.np"
#dist_file_2 = "./gen_log/UAI2019_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4_no_lin_w_freq_auto1_alldrop01.np"
#dist_file_2 = "./gen_log/UAI2019_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4_no_lin_w_freq_auto1_alldrop01.np"

#dist_file_1 = "./gen_log/UAI2019_avg_cbow_freq_4_dist_bid_score.np"
#dist_file_1 = "./gen_log/UAI2019_max_cbow_freq_4_dist_bid_score.np"
#dist_file_2 = "./gen_log/UAI2019_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4_no_lin_auto1_bid_score.np"
#dist_file_2 = "./gen_log/UAI2019_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4_no_lin_auto1_bid_score.np"


#input_dict_path = "./data/processed/NeurIPS2019_bid_score_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/NeurIPS2019_bid_score_gorc_uncased/user/dictionary_index"
#submission_data_file = './data/processed/NeurIPS2019_bid_score_gorc_uncased/tensors_cold/test.pt'
#input_dict_path = "./data/processed/ICLR2019_bid_score_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/ICLR2019_bid_score_gorc_uncased/user/dictionary_index"
#submission_data_file = './data/processed/ICLR2019_bid_score_gorc_uncased/tensors_cold/test.pt'
#input_dict_path = "./data/processed/ICLR2019_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/ICLR2019_gorc_uncased/user/dictionary_index"
#submission_data_file = './data/processed/ICLR2019_gorc_uncased/tensors_cold/test.pt'
#input_dict_path = "./data/processed/ICLR2020_old_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/ICLR2020_old_gorc_uncased/user/dictionary_index"
#submission_data_file = './data/processed/ICLR2020_old_gorc_uncased/tensors_cold/test.pt'
#input_dict_path = "./data/processed/UAI2019_old_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/UAI2019_old_gorc_uncased/user/dictionary_index"
#submission_data_file = './data/processed/UAI2019_old_gorc_uncased/tensors_cold/test.pt'
#input_dict_path = "./data/processed/ICLR2020_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/ICLR2020_gorc_uncased/user/dictionary_index"
#submission_data_file = './data/processed/ICLR2020_gorc_uncased/tensors_cold/test.pt'
#input_dict_path = "./data/processed/UAI2019_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/UAI2019_gorc_uncased/user/dictionary_index"
#submission_data_file = './data/processed/UAI2019_gorc_uncased/tensors_cold/test.pt'
#input_dict_path = "./data/processed/ICLR2020_bid_score_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/ICLR2020_bid_score_gorc_uncased/user/dictionary_index"
#submission_data_file = './data/processed/ICLR2020_bid_score_gorc_uncased/tensors_cold/test.pt'
#input_dict_path = "./data/processed/UAI2019_bid_score_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/UAI2019_bid_score_gorc_uncased/user/dictionary_index"
#submission_data_file = './data/processed/UAI2019_bid_score_gorc_uncased/tensors_cold/test.pt'
#input_dict_path = "./data/processed/ICLR2020_old_bid_score_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/ICLR2020_old_bid_score_gorc_uncased/user/dictionary_index"
#submission_data_file = './data/processed/ICLR2020_old_bid_score_gorc_uncased/tensors_cold/test.pt'
#input_dict_path = "./data/processed/UAI2019_old_bid_score_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/UAI2019_old_bid_score_gorc_uncased/user/dictionary_index"
#submission_data_file = './data/processed/UAI2019_old_bid_score_gorc_uncased/tensors_cold/test.pt'
#input_dict_path = "./data/processed/UAI2019_bid_score_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/UAI2019_bid_score_gorc_uncased/user/dictionary_index"
#submission_data_file = './data/processed/UAI2019_bid_score_gorc_uncased/tensors_cold/test.pt'

out_f_path = './gen_log/temp'

dist_1 = np.loadtxt(dist_file_1)
if len(dist_file_2) == 0:
    paper_user_dist = dist_1
else:
    dist_2 = np.loadtxt(dist_file_2)
    paper_user_dist = dist_1 * merge_alpha + dist_2 * (1- merge_alpha)

if eval_target == 'tag':
    paper_tag_dist = paper_user_dist
    paper_user_dist = []
#paper_user_dist = np.power(dist_1 + 1, merge_alpha) * np.power(dist_2 + 1, 1- merge_alpha)

eval_bsz = 50

device = torch.device('cpu')

with open(submission_data_file,'rb') as f_in:
    dataloader_test_info = utils.create_data_loader(f_in, eval_bsz, device, want_to_shuffle = False, deduplication = True)

if eval_target == 'user':
    with open(user_dict_path) as f_in:
        user_idx2word_freq = utils.load_idx2word_freq(f_in)
elif eval_target == 'tag':
    with open(tag_dict_path) as f_in:
        tag_idx2word_freq = utils.load_idx2word_freq(f_in)

with open(input_dict_path) as f_in:
    idx2word_freq = utils.load_idx2word_freq(f_in)

most_popular_baseline = False
div_eval = 'openreview'
if eval_target == 'user':
    test_user = True
    test_tag = False
    tag_idx2word_freq = []
    paper_tag_dist = []
elif eval_target == 'tag':
    test_user = False
    test_tag = True
    user_idx2word_freq = []
    paper_user_dist = []
    
with open(out_f_path, 'w') as outf:
    utils_testing.recommend_test_from_all_dist(dataloader_test_info, paper_user_dist, paper_tag_dist, idx2word_freq, user_idx2word_freq, tag_idx2word_freq, test_user, test_tag, outf, device, most_popular_baseline, div_eval)
