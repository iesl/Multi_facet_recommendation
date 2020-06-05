import sys
sys.path.insert(0, sys.path[0]+'/../..')
import utils
import numpy as np



data_folder = './data/processed/NeurIPS2020_final_review_gorc_uncased'
#data_folder = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/processed/NeurIPS2020_final_gorc_uncased'
#data_folder = './data/processed/NeurIPS2019_bid_score_gorc_uncased'
#data_folder = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/processed/NeurIPS2019_bid_score_gorc_uncased'
test_paper_id_file = data_folder + '/paper_id_test'
user_dict_file = data_folder + '/user/dictionary_index'

input_2_file = ''
#input_file = './gen_log/NeurIPS2019_TPMS_dist.np'
#output_file = './gen_log/to_NeurIPS2019/tpms.csv'
#input_file = './gen_log/NeurIPS2019_max_cbow_freq_4_dist.np'
#output_file = './gen_log/to_NeurIPS2019/cbow_max_agg.csv'
#input_file = './gen_log/NeurIPS2019_avg_cbow_freq_4_dist.np'
#output_file = './gen_log/to_NeurIPS2019/cbow_avg_agg.csv'
#input_file = './gen_log/NeurIPS2019_scibert_avg_cbow_freq_4_dist.np'
#output_file = './gen_log/to_NeurIPS2019/scibert_avg_agg.csv'
#input_file = './gen_log/NeurIPS2019_scibert_max_cbow_freq_4_dist.np'
#output_file = './gen_log/to_NeurIPS2019/scibert_max_agg.csv'
#input_file = './/gen_log/NeurIPS2019_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4__sim_w_freq_no_lin_auto_avg1_alldrop01_bid_score.np'
#output_file = './gen_log/to_NeurIPS2019/single_sim.csv'
#input_file = './/gen_log/NeurIPS2019_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4__w_freq_single_stable_auto_avg1_alldrop01_bid_score.np'
#output_file = './gen_log/to_NeurIPS2019/single.csv'
#input_file = './/gen_log/NeurIPS2019_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4__w_freq_single_stable_auto_avg1_alldrop01_bid_score.np'
#output_file = './gen_log/to_NeurIPS2019/multi.csv'
#input_file = './/gen_log/NeurIPS2019_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4__single_stable_fix_auto_avg1_alldrop01_bid_score.np'
#output_file = './gen_log/to_NeurIPS2019/multi_fixed.csv'
#input_file = './/gen_log/NeurIPS2020_final_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4__w_freq_single_stable_fix_auto_avg1_alldrop01.np'
#output_file = './gen_log/to_NeurIPS2020/multi_fixed.csv'
#input_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/NeurIPS2019_TPMS_dist.np'
#output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/to_NeurIPS2019/tpms.csv'
#input_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/NeurIPS2019_max_cbow_freq_4_dist.np'
#output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/to_NeurIPS2019/cbow_max_agg.csv'
#input_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/NeurIPS2020_final_avg_cbow_freq_4_dist.np'
#output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/to_NeurIPS2020/cbow_avg_agg_ac.csv'
#input_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/NeurIPS2019_avg_cbow_freq_4_dist.np'
#output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/to_NeurIPS2019/cbow_avg_agg.csv'
#input_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/NeurIPS2019_scibert_avg_cbow_freq_4_dist.np'
#output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/to_NeurIPS2019/scibert_avg_agg.csv'
#input_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/NeurIPS2019_scibert_max_cbow_freq_4_dist.np'
#output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/to_NeurIPS2019/scibert_max_agg.csv'
#agg_method = 'max'
agg_method = 'avg'
input_file = './gen_log/NeurIPS2020_final_review_'+agg_method+'_cbow_freq_4_dist.np'
input_2_file = './gen_log/NeurIPS2020_final_review_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4__w_freq_single_stable_fix_auto_avg1_alldrop01.np'
output_file = './gen_log/to_NeurIPS2020/cbow_'+agg_method+'_agg_multi.csv'
#input_file = './/gen_log/NeurIPS2019_'+agg_method+'_cbow_freq_4_dist.np'
#input_2_file = './/gen_log/NeurIPS2019_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4__sim_w_freq_no_lin_auto_avg1_alldrop01_bid_score.np'
#output_file = './gen_log/to_NeurIPS2019/cbow_'+agg_method+'_agg_single_sim.csv'
#input_2_file = './/gen_log/NeurIPS2019_rec_test_trans_bsz50_n1_shuffle_uni_max_lr2e-4__w_freq_single_stable_auto_avg1_alldrop01_bid_score.np'
#output_file = './gen_log/to_NeurIPS2019/cbow_'+agg_method+'_agg_single.csv'
#input_2_file = './/gen_log/NeurIPS2019_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4__w_freq_single_stable_auto_avg1_alldrop01_bid_score.np'
#input_2_file = './/gen_log/NeurIPS2019_rec_test_trans_bsz50_n5_shuffle_uni_max_lr2e-4__single_stable_fix_auto_avg1_alldrop01_bid_score.np'
#output_file = './gen_log/to_NeurIPS2019/cbow_'+agg_method+'_agg_multi_fixed.csv'

merge_alpha = 0.8

paper_dist_1 = np.loadtxt(input_file)

if len(input_2_file) == 0:
    paper_dist = paper_dist_1
else:
    paper_dist_2 = np.loadtxt(input_2_file)
    paper_dist = paper_dist_1 * merge_alpha + paper_dist_2 * (1- merge_alpha)

num_special_tok = 3

with open(user_dict_file) as f_in:
    idx2word_freq = utils.load_idx2word_freq(f_in)
    #user_raw_d2_idx_freq, user_idx_max = utils.load_word_dict(f_in)
#user_num = user_idx_max + 1
user_num = len(idx2word_freq)

idx_l2_paper_id = []
with open(test_paper_id_file) as f_in:
    for idx, line in enumerate(f_in):
        paper_id = line.rstrip()
        idx_l2_paper_id.append(paper_id)
        #paper_id_list.append(paper_id)
        #paper_id_d2_idx[paper_id] = idx

paper_num = len(idx_l2_paper_id)


paper_num_np, user_num_np = paper_dist.shape

assert paper_num_np == paper_num, print(paper_num_np, paper_num)
assert user_num_np == user_num

with open(output_file, 'w') as f_out:
    for i in range(num_special_tok, user_num):
        user_raw = idx2word_freq[i][0]
        suffix_start = user_raw.index('|')
        assert suffix_start > 0
        user_name = user_raw[:suffix_start]
        for j in range(paper_num):
            paper_id = idx_l2_paper_id[j]
            score = 1 - paper_dist[j,i]
            f_out.write('{},{},{}\n'.format(paper_id,user_name,score))

