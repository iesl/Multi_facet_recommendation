import sys
sys.path.insert(0, sys.path[0]+'/../..')
import utils_testing
from scipy.stats import spearmanr, pearsonr
import numpy as np
import json
from scipy.spatial import distance
#import torch
#import utils

import getopt

sent_emb_file_name = "./gen_log/BERT_WiC-dev_cased.json"

print(sent_emb_file_name)
sys.stdout.flush()

#test_file_name = "dataset_testing/WiC_dataset/train/train.data.txt"
#gt_file_name = "dataset_testing/WiC_dataset/train/train.gold.txt"
test_file_name = "dataset_testing/WiC_dataset/dev/dev.data.txt"
gt_file_name = "dataset_testing/WiC_dataset/dev/dev.gold.txt"


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


print("loading ", sent_emb_file_name)
#word2emb, emb_size = utils.load_emb_file_to_dict(embedding_file_name)
with open(sent_emb_file_name) as f_in:
    embedding_list = json.load(f_in)

sent2emb = {}
for sent, proc_idx, avg_emb, cls_emb in embedding_list:
    sent2emb[sent] = [avg_emb, cls_emb]

method_names = ['BERT_avg_emb','BERT_cls_emb']

pred_scores = []
for fields in testing_list:
    sent_1 = fields[0]
    sent_2 = fields[1]

    score_pred = [0] * len(method_names)
    for j in range(len(method_names)):
        sent_emb_1 = sent2emb[sent_1][j]
        sent_emb_2 = sent2emb[sent_2][j]
        score_pred[j] = 1 - distance.cosine(sent_emb_1, sent_emb_2)
    pred_scores.append(score_pred)


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
