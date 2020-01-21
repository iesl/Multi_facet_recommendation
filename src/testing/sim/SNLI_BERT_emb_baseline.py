import sys
sys.path.insert(0, sys.path[0]+'/../..')
#import utils_testing
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
import numpy as np
import json
from scipy.spatial import distance
#import torch
#import utils

import getopt

#method = "BERT"
method = "ELMo"
#method = "ST"

#sent_emb_file_name = "./gen_log/ELMo_snli-dev_cased.json"
sent_emb_file_name = "./gen_log/ELMo_snli-test_cased.json"
#sent_emb_file_name = "./gen_log/ELMo_large_sts-test_cased.json"
#sent_emb_file_name = "./gen_log/BERT_large_sts-dev_cased.json"
#sent_emb_file_name = "./gen_log/BERT_sts-train_cased.json"
#sent_emb_file_name = "./gen_log/BERT_large_sts-test_cased.json"
#sent_emb_file_name = "./gen_log/BERT_large_snli_dev_cased.json"
#sent_emb_file_name = "./gen_log/BERT_base_snli_test_cased.json"
#sent_emb_file_name = "./gen_log/BERT_large_snli_test_cased.json"
#sent_emb_file_name = "./gen_log/BERT_base_sts_2012_train_cased.json"
#sent_emb_file_name = "./gen_log/BERT_base_sts_2012-6_test_cased.json"
#sent_emb_file_name = "./gen_log/ST_d300_sts-dev.json"
#sent_emb_file_name = "./gen_log/ST_d600_sts-dev_36k.json"
#sent_emb_file_name = "./gen_log/ST_d600_sts-dev_final.json"
#sent_emb_file_name = "./gen_log/ST_d600_sts-test_36k.json"
#sent_emb_file_name = "./gen_log/ST_d600_sts_2012-6_test_36k.json"

gt_file_name = "./dataset_testing/SNLI/snli_1.0_dev_useful.txt"
#gt_file_name = "./dataset_testing/SNLI/snli_1.0_test_useful.txt"
#gt_file_name = "./dataset_testing/STS/stsbenchmark/sts-dev.csv"
#gt_file_name = "./dataset_testing/STS/stsbenchmark/sts-train.csv"
#gt_file_name = "./dataset_testing/STS/stsbenchmark/sts-test.csv"
#gt_file_name = "./dataset_testing/STS/sts_2012_train"
#gt_file_name = "./dataset_testing/STS/sts_all_years_test"
#gt_file_name = "./dataset_testing/STS/sts_test_year_2012"

print(sent_emb_file_name)
sys.stdout.flush()

#with open(freq_file_name) as f_in:
#    word_d2_idx_freq, max_ind = utils.load_word_dict(f_in)

#utils_testing.compute_freq_prob(word_d2_idx_freq)

def load_entail(file_name, all_pairs):
    not_noun_count = 0
    dataset = []
    with open(file_name) as f_in:
        for line_idx, line in enumerate(f_in):
            #if line_idx == 0:
            #    continue
            fields = line.rstrip().split('\t')
            hypo_candidate = fields[1].rstrip()
            hyper_candidate = fields[2].rstrip()
            if hypo_candidate == 'sentence1':
                continue
            label = 0
            if fields[0] == 'entailment':
                label = 1
            dataset.append( [hypo_candidate, hyper_candidate, label] )
            all_pairs.append( [hypo_candidate, hyper_candidate, 'entail'] )
    return dataset

def load_STS_file(f_in):
    output_list = []
    for line in f_in:
        #print(line.rstrip().split('\t'))
        fields = line.rstrip().split('\t')
        genre, source, source_year, org_idx, score, sent_1, sent_2 = fields[:7]
        output_list.append([sent_1.rstrip(), sent_2.rstrip(), float(score), genre+'-'+source, len(sent_1) + len(sent_2)])
    return output_list

all_pairs = []
print("load", gt_file_name)
testing_list = load_entail(gt_file_name, all_pairs)


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
elif method == "ELMo":
    sent2emb = {}
    for sent, proc_idx, avg_emb, cls_emb in embedding_list:
        #sent2emb[' '.join(sent)] = [avg_emb, cls_emb]
        sent2emb[sent] = [avg_emb, cls_emb]

    method_names = ['ELMo_avg_emb','ELMo_proc_emb']
    

pair_d2_scores = {}
for fields in testing_list:
    sent_1 = fields[0]
    sent_2 = fields[1]

    score_pred = [0] * len(method_names)
    for j in range(len(method_names)):
        sent_emb_1 = sent2emb[sent_1][j]
        sent_emb_2 = sent2emb[sent_2][j]
        score_pred[j] = 1 - distance.cosine(sent_emb_1, sent_emb_2)
    pair_d2_scores[(sent_1,sent_2)] = score_pred


def test_entail_single(dataset, method_idx, method_name, pair_d2_score):
    sent1_d2_score_label = {}
    #uni_OOV_count = 0
    #bi_OOV_count = 0
    direction_list = []
    for sent1, sent2, label in dataset:
        #score_pred, uni_OOV_count, bi_OOV_count = output_sim_score(bigram, unigram, word2emb, uni_OOV_count, bi_OOV_count)
        score_pred = pair_d2_score[ (sent1, sent2) ][method_idx]
        if sent1 not in sent1_d2_score_label:
            sent1_d2_score_label[sent1] = []
        sent1_d2_score_label[sent1].append( [score_pred, label] )

        if label == 1 and 'diff' in method_name:
            if score_pred == OOV_value or score_pred == 0:
                direction_list.append(0.5)
            elif score_pred > 0:
                direction_list.append(1)
            else:
                direction_list.append(0)
    pred_list = []
    gt_list = []
    for sent1 in sent1_d2_score_label:
        score_label = sent1_d2_score_label[sent1]
        score_list, label_list = zip(*score_label)
        max_idx = np.argmax(score_list)
        for idx, label in enumerate(label_list):
            if idx == max_idx:
                pred_list.append( 1 )
            else:
                pred_list.append( 0 )
            gt_list.append(label)

    F1 = f1_score(gt_list, pred_list)
    acc = accuracy_score(gt_list, pred_list)
    print(method_name + ': F1 ' + str(F1) + ', Accuracy ' + str(acc), ', Direction Accuracy ' + str(np.mean(direction_list)) )

for j in range(len(method_names)):
    test_entail_single(testing_list, j, method_names[j], pair_d2_scores)

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


