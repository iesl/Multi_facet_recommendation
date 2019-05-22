import sys
sys.path.insert(0, sys.path[0]+'/../..')
import utils_testing
from scipy.stats import spearmanr, pearsonr
import numpy as np
import torch
import utils

import getopt
help_msg = '-t <topic_file_name> -w <w_emb_file_name> -d <freq_file_name> -g <gt_file_name> -m <pc_mode> -p <path_to_pc>'

#pc_mode = 'none'
pc_mode = 'self'
#pc_mode = 'save'
#pc_mode = 'load'

#path_to_pc = './gen_log/pc_sts_train_n10'
path_to_pc = ''

try:
    opts, args = getopt.getopt(sys.argv[1:], "t:w:d:g:m:p:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-t"):
        topic_file_name = arg
    elif opt in ("-w"):
        w_emb_file_name = arg
    elif opt in ("-d"):
        freq_file_name = arg
    elif opt in ("-g"):
        gt_file_name= arg
    elif opt in ("-m"):
        pc_mode = arg
    elif opt in ("-p"):
        path_to_pc = arg

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
#topic_file_name = "./gen_log/STS_dev_wiki2016_glove_trans_n10_bsz200_ep2_1.json"
#topic_file_name = "./gen_log/STS_train_wiki2016_glove_trans_n10_bsz200_ep2_1.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_glove_trans_n3_bsz200_ep1_1.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_glove_trans_n1_bsz200_ep1_0.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_glove_trans_RMSProp_bsz200_ep1_0.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_glove_trans_no_connect_bsz200_ep2_0.json"
#w_emb_file_name = "./resources/glove.840B.300d_filtered_wiki2016.txt"
#topic_file_name = "./gen_log/STS_dev_wiki2016_word2vec_trans_n20_bsz200_ep1_0.json"
#w_emb_file_name = "./resources/word2vec_wiki2016_min100.txt"
#topic_file_name = "./gen_log/STS_dev_wiki2016_lex_crawl_trans_bsz200_ep2_1.json"
#w_emb_file_name = "./resources/lexvec_wiki2016_min100"
#freq_file_name = "./data/processed/wiki2016_min100/dictionary_index"

#topic_file_name = "./gen_log/STS_dev_wiki2016_lex_enwiki_trans_bsz200_ep1_0.json"
#w_emb_file_name = "./resources/lexvec_enwiki_wiki2016_min100"
#topic_file_name = "./gen_log/STS_dev_wiki2016_paragram_trans_bsz200_ep1_1.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_paragram_trans_n20_bsz200_ep1_0.json"
#w_emb_file_name = "./resources/paragram_wiki2016_min100"
#freq_file_name = "./data/processed/wiki2016_lower_min100/dictionary_index"

#topic_file_name = "./gen_log/STS_dev_wiki2016_glove_maxlc_bsz200_ep2_0.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_updated_glove_maxlc_bsz200_ep3_0.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_word2vec_maxlc_bsz200_ep2_0.json"

#gt_file_name = "./dataset_testing/STS/stsbenchmark/sts-dev.csv"
#gt_file_name = "./dataset_testing/STS/stsbenchmark/sts-train.csv"

print(topic_file_name)
print(w_emb_file_name)
sys.stdout.flush()


#device = 'cpu'
device = 'cuda'
bsz = 100
L1_losss_B = 0.2

with open(freq_file_name) as f_in:
    word_d2_idx_freq, max_ind = utils.load_word_dict(f_in)

#def compute_freq_prob(word_d2_idx_freq):
#    all_idx, all_freq= list( zip(*word_d2_idx_freq.values()) )
#    freq_sum = float(sum(all_freq))
#    for w in word_d2_idx_freq:
#        idx, freq = word_d2_idx_freq[w]
#        word_d2_idx_freq[w].append(freq/freq_sum)

utils_testing.compute_freq_prob(word_d2_idx_freq)

#def load_emb_file(emb_file):
#    word2emb = {}
#    with open(emb_file) as f_in:
#        for line in f_in:
#            word_val = line.rstrip().split(' ')
#            if len(word_val) < 3:
#                continue
#            word = word_val[0]
#            val = np.array([float(x) for x in  word_val[1:]])
#            word2emb[word] = val
#            emb_size = len(val)
#    return word2emb

def load_STS_file(f_in):
    output_list = []
    for line in f_in:
        #print(line.rstrip().split('\t'))
        fields = line.rstrip().split('\t')
        genre, source, source_year, org_idx, score, sent_1, sent_2 = fields[:7]
        output_list.append([sent_1.rstrip(), sent_2.rstrip(), float(score), genre+'-'+source, len(sent_1) + len(sent_2)])
    return output_list

print("load", gt_file_name)
with open(gt_file_name) as f_in:
    testing_list = load_STS_file(f_in)

print("load", topic_file_name)
with open(topic_file_name) as f_in:
    sent_d2_topics = utils_testing.load_prediction_from_json(f_in)

print("load", w_emb_file_name)
word2emb, emb_size  = utils.load_emb_file_to_dict(w_emb_file_name)

testing_pair_loader, other_info = utils_testing.build_loader_from_pairs(testing_list, sent_d2_topics, bsz, device)

with torch.no_grad():
    #pred_scores, method_names = utils_testing.predict_sim_scores(testing_pair_loader, L1_losss_B, device, word2emb, other_info, word_d2_idx_freq, OOV_sim_zero = True, compute_WMD = False)
    #pred_scores, method_names = utils_testing.predict_sim_scores(testing_pair_loader, L1_losss_B, device, word2emb, other_info, word_d2_idx_freq, OOV_sim_zero = True, compute_WMD = True, pc_mode = pc_mode, path_to_pc = path_to_pc)
    pred_scores, method_names = utils_testing.predict_sim_scores(testing_pair_loader, L1_losss_B, device, word2emb, other_info, word_d2_idx_freq, OOV_sim_zero = True, compute_WMD = False, pc_mode = pc_mode, path_to_pc = path_to_pc)

def get_lower_half(score_list):
    sorted_ind = np.argsort(score_list)
    #print(score_list[sorted_ind[0]])
    lower_idx_set = set(sorted_ind[:int(len(sorted_ind)/2)])
    #print( len(lower_idx_set) )
    return lower_idx_set

print(len(pred_scores))
def summarize_prediction_STS(pred_scores, testing_list):
    method_d2_genre_score = {}
    method_paper_order = ['SC_rmsprop', 'avg_en_word_emb', 'baseline', 'baseline_freq4', 'baseline_freq_4_pc1', 'w_imp_sim', 'w_imp_sim_freq_4', 'w_imp_sim_freq_4_pc1']
    for method in method_paper_order:
        method_d2_genre_score[method] = []
    #method_paper_set = set(method_paper_order)
    genre_paper_order = ['all', 'lower', 'higher']
    #genre_paper_set = set(genre_paper_order)
    paper_num_format = '{0:.1f}'
    genre_s_d2_scores = {'all': [], 'lower': [], 'higher': [], 'short': [], 'long': []}
    field_lists = list(zip(*testing_list))
    score_list = field_lists[2]
    lower_idx_set = get_lower_half(score_list)
    sent_len_list = field_lists[4]
    shorter_idx_set = get_lower_half(sent_len_list)
    for i in range(len(testing_list)):
        genre_s = testing_list[i][3]
        if genre_s not in genre_s_d2_scores:
            genre_s_d2_scores[genre_s] = []
        genre_s_d2_scores[genre_s].append( pred_scores[i] + [testing_list[i][2]] )
        genre_s_d2_scores['all'].append( pred_scores[i] + [testing_list[i][2]] )
        if i in lower_idx_set:
            genre_s_d2_scores['lower'].append(pred_scores[i] + [testing_list[i][2]] )
        else:
            genre_s_d2_scores['higher'].append(pred_scores[i] + [testing_list[i][2]] )
        if i in shorter_idx_set:
            genre_s_d2_scores['short'].append(pred_scores[i] + [testing_list[i][2]] )
        else:
            genre_s_d2_scores['long'].append(pred_scores[i] + [testing_list[i][2]] )
    for genre_s in genre_s_d2_scores:
        pred_and_gt = list( zip(*genre_s_d2_scores[genre_s]) ) 
        print("genre_source: ", genre_s)
        #print(pred_and_gt)
        for m in range(len(pred_and_gt)-1):
            print("method ", method_names[m],pearsonr( pred_and_gt[m], pred_and_gt[-1] ), spearmanr( pred_and_gt[m], pred_and_gt[-1] ))
    for genre_s in genre_paper_order:
        pred_and_gt = list( zip(*genre_s_d2_scores[genre_s]) )
        for m in range(len(pred_and_gt)-1):
            if method_names[m] in method_d2_genre_score:
                p_r, conf = pearsonr( pred_and_gt[m], pred_and_gt[-1])
                method_d2_genre_score[method_names[m]].append( paper_num_format.format( 100 * p_r  ) )
    for method in method_paper_order:
        print(method + ' & ' + ' & '.join( method_d2_genre_score[method]) )

summarize_prediction_STS(pred_scores, testing_list)
