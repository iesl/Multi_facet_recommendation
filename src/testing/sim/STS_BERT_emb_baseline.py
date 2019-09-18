import sys
sys.path.insert(0, sys.path[0]+'/../..')
#import utils_testing
from scipy.stats import spearmanr, pearsonr
import numpy as np
import json
from scipy.spatial import distance
#import torch
#import utils

import getopt

method = "BERT"
#method = "ST"

#sent_emb_file_name = "./gen_log/BERT_large_sts-dev_cased.json"
#sent_emb_file_name = "./gen_log/BERT_sts-train_cased.json"
sent_emb_file_name = "./gen_log/BERT_large_sts-test_cased.json"
#sent_emb_file_name = "./gen_log/ST_d300_sts-dev.json"
#sent_emb_file_name = "./gen_log/ST_d600_sts-dev_36k.json"
#sent_emb_file_name = "./gen_log/ST_d600_sts-dev_final.json"
#sent_emb_file_name = "./gen_log/ST_d600_sts-test_36k.json"

#gt_file_name = "./dataset_testing/STS/stsbenchmark/sts-dev.csv"
#gt_file_name = "./dataset_testing/STS/stsbenchmark/sts-train.csv"
gt_file_name = "./dataset_testing/STS/stsbenchmark/sts-test.csv"

print(sent_emb_file_name)
sys.stdout.flush()


#with open(freq_file_name) as f_in:
#    word_d2_idx_freq, max_ind = utils.load_word_dict(f_in)

#utils_testing.compute_freq_prob(word_d2_idx_freq)


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


print("loading ", sent_emb_file_name)
#word2emb, emb_size = utils.load_emb_file_to_dict(embedding_file_name)
with open(sent_emb_file_name) as f_in:
    embedding_list = json.load(f_in)

if method == "BERT":
    sent2emb = {}
    for sent, proc_idx, avg_emb, cls_emb in embedding_list:
        sent2emb[sent] = [avg_emb, cls_emb]

    method_names = ['BERT_avg_emb','BERT_cls_emb']
elif method == "ST":
    sent2emb = {}
    for sent, emb in embedding_list:
        sent2emb[sent] = [emb]

    method_names = ['ST_emb']
    

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
def summarize_prediction_STS(pred_scores, testing_list):
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


summarize_prediction_STS(pred_scores, testing_list)
