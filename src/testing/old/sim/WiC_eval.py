import sys
sys.path.insert(0, sys.path[0]+'/../..')
import utils_testing
from scipy.stats import spearmanr, pearsonr
import numpy as np
import torch
import utils

import getopt
help_msg = '-t <topic_file_name> -w <w_emb_file_name> -d <freq_file_name> -f <feature_file_name> -g <gt_file_name>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "t:w:d:f:g:")
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
    elif opt in ("-f"):
        test_file_name= arg
    elif opt in ("-g"):
        gt_file_name= arg

#topic_file_name = "./gen_log/WiC_dev_wiki2016_glove_trans_n20_bsz200_ep1_1.json"
#topic_file_name = "./gen_log/WiC_dev_wiki2016_glove_trans_n3_bsz200_ep1_1.json"
#topic_file_name = "./gen_log/WiC_dev_wiki2016_glove_trans_n1_bsz200_ep1_0.json"
#topic_file_name = "./gen_log/WiC_dev_wiki2016_glove_trans_n10_bsz200_ep2_1.json"
#topic_file_name = "./gen_log/WiC_dev_wiki2016_glove_trans_no_connect_bsz200_ep2_0.json"
#w_emb_file_name = "./resources/word2vec_wiki2016_min100.txt"
#w_emb_file_name = "./resources/glove.840B.300d_filtered_wiki2016.txt"
#w_emb_file_name = "./resources/lexvec_wiki2016_min100"
#freq_file_name = "./data/processed/wiki2016_min100/dictionary_index"

#w_emb_file_name = "./resources/lexvec_enwiki_wiki2016_min100"
#w_emb_file_name = "./resources/paragram_wiki2016_min100"
#freq_file_name = "./data/processed/wiki2016_lower_min100/dictionary_index"

print(topic_file_name)
print(w_emb_file_name)
sys.stdout.flush()

#test_file_name = "dataset_testing/WiC_dataset/train/train.data.txt"
#gt_file_name = "dataset_testing/WiC_dataset/train/train.gold.txt"
#test_file_name = "dataset_testing/WiC_dataset/dev/dev.data.txt"
#gt_file_name = "dataset_testing/WiC_dataset/dev/dev.gold.txt"

#device = 'cpu'
device = 'cuda'
bsz = 100
L1_losss_B = 0.2

with open(freq_file_name) as f_in:
    word_d2_idx_freq, max_ind = utils.load_word_dict(f_in)

utils_testing.compute_freq_prob(word_d2_idx_freq)

def load_WiC_files(test_file_name, gt_file_name):
    gt_list = []
    with open(gt_file_name) as f_in:
        for line in f_in:
            gt_list.append( line.rstrip() )
    output_list = []
    with open(test_file_name) as f_in:
        for i, line in enumerate(f_in):
            fields = line.rstrip().split('\t')
            target_word, POS, span, sent1, sent2 = fields
            gt = gt_list[i]
            output_list.append([ sent1, sent2, gt, POS, len(sent1) + len(sent2) ])
    return output_list

print("load", test_file_name)
print("load", gt_file_name)
testing_list = load_WiC_files(test_file_name, gt_file_name)

print("load", topic_file_name)
with open(topic_file_name) as f_in:
    sent_d2_topics = utils_testing.load_prediction_from_json(f_in)

print("load", w_emb_file_name)
word2emb, emb_size  = utils.load_emb_file_to_dict(w_emb_file_name)

testing_pair_loader, other_info = utils_testing.build_loader_from_pairs(testing_list, sent_d2_topics, bsz, device)

with torch.no_grad():
    pred_scores, method_names = utils_testing.predict_sim_scores(testing_pair_loader, L1_losss_B, device, word2emb, other_info, word_d2_idx_freq)

def get_lower_half(score_list):
    sorted_ind = np.argsort(score_list)
    #print(score_list[sorted_ind[0]])
    lower_idx_set = set(sorted_ind[:int(len(sorted_ind)/2)])
    #print( len(lower_idx_set) )
    return lower_idx_set


print(len(pred_scores))
def summarize_prediction_WiC(pred_scores, testing_list):
    genre_s_d2_scores = {'all': [], 'short': [], 'long': []}
    field_lists = list(zip(*testing_list))
    sent_len_list = field_lists[4]
    shorter_idx_set = get_lower_half(sent_len_list)
    for i in range(len(testing_list)):
        genre_s = testing_list[i][3]
        if genre_s not in genre_s_d2_scores:
            genre_s_d2_scores[genre_s] = []
        genre_s_d2_scores[genre_s].append( pred_scores[i] + [testing_list[i][2]] )
        genre_s_d2_scores['all'].append( pred_scores[i] + [testing_list[i][2]] )
        if i in shorter_idx_set:
            genre_s_d2_scores['short'].append(pred_scores[i] + [testing_list[i][2]] )
        else:
            genre_s_d2_scores['long'].append(pred_scores[i] + [testing_list[i][2]] )
    for genre_s in genre_s_d2_scores:
        pred_and_gt = list( zip(*genre_s_d2_scores[genre_s]) ) 
        print("genre_source: ", genre_s)
        #print(pred_and_gt)
        for m in range(len(pred_and_gt)-1):
            AP, F1_best, acc_best = utils_testing.compute_AP_best_F1_acc(pred_and_gt[m], pred_and_gt[-1], correct_label = 'T')
            print("method ", method_names[m], ': AP@all '+ str(AP) + ', Best F1 ' + str(F1_best) + ', Best Accuracy ' + str(acc_best)) 


summarize_prediction_WiC(pred_scores, testing_list)
