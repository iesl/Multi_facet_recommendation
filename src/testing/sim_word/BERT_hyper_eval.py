import random
import numpy as np
from sklearn.metrics import average_precision_score, f1_score
import sys
sys.path.insert(0, sys.path[0]+'/../..')
import utils
import utils_testing
import torch

import json
from scipy.spatial import distance

train_or_test = 'test'

#embedding_file_name = "gen_log/BERT_HypeNet_WordNet_phrase_val.json"
embedding_file_name = "gen_log/BERT_HypeNet_WordNet_phrase_test.json"
#embedding_file_name = embedding_dir + "glove.840B.300d_filtered_wiki2016_min100.txt"

#lower
#lowercase_emb = False
#embedding_file_name = embedding_dir + "paragram_wiki2016_min100"
#topic_file_name = "./gen_log/phrase_train_wiki2016_glove_trans_bsz200_ep1_3.json"
#topic_file_name = "./gen_log/phrase_train_wiki2016_glove_trans_bsz200_no_connect_ep1_3.json"
#topic_file_name = "./gen_log/phrase_train_wiki2016_glove_trans_bsz200_no_connect_no_stop_ep1_3.json"
#topic_file_name = "./gen_log/phrase_train_wiki2016_glove_trans_no_stop_bsz200_ep1_3.json"
#topic_file_name = "./gen_log/phrase_train_wiki2016_glove_trans_bsz200_no_connect_ep1_5.json"
#topic_file_name = "./gen_log/phrase_train_wiki2016_glove_trans5_bsz1000_ep1_29.json"
#topic_file_name = "./gen_log/phrase_train_wiki2016_glove_trans_n1_bsz200_no_connect_ep1_2.json"
#topic_file_name = "./gen_log/phrase_train_wiki2016_glove_trans_n1_bsz200_no_connect_no_stop_ep1_2.json"
#topic_file_name = "./gen_log/phrase_hyper_val_wiki2016_glove_trans5_bsz1000_ep1_29.json"
#embedding_file_name = embedding_dir + "glove.42B.300d_filtered_wiki2016_nchunk_lower_min100.txt"
#topic_file_name = "./gen_log/phrase_train_wiki2016_lex_enwiki_trans_d400_bsz400_ep1_5.json"
#embedding_file_name = embedding_dir + "lexvec_enwiki_wiki2016_min100"

#freq_file_name = "./data/processed/wiki2016_nchunk_lower_min100/dictionary_index"

#w2v = gensim.models.KeyedVectors.load_word2vec_format(emb_file_path, binary=False)

dataset_dir = "./dataset_testing/phrase/"
if train_or_test == 'train':
    dataset_list = [ [ dataset_dir + 'HypeNet/rnd/train.tsv', 'raw' , "hyper"], [ dataset_dir + 'WordNet/wordnet_train.txt' , "POS",  "hyper" ] ]
elif train_or_test == 'val':
    dataset_list = [ [ dataset_dir + 'HypeNet/rnd/val.tsv', 'raw' , "hyper"], [ dataset_dir + 'WordNet/wordnet_valid.txt' , "POS",  "hyper" ] ]
elif train_or_test == 'test':
    dataset_list = [ [ dataset_dir + 'HypeNet/rnd/test.tsv', 'raw' , "hyper"], [ dataset_dir + 'WordNet/wordnet_test.txt' , "POS",  "hyper" ] ]
elif train_or_test == 'val_test':
    dataset_list = [ [ dataset_dir + 'HypeNet/rnd/val.tsv', 'raw' , "hyper"], [ dataset_dir + 'HypeNet/rnd/test.tsv', 'raw' , "hyper"], [ dataset_dir + 'WordNet/wordnet_valid.txt' , "POS",  "hyper" ], [ dataset_dir + 'WordNet/wordnet_test.txt' , "POS",  "hyper" ] ]
elif train_or_test == 'word':
    dataset_list = [ [ dataset_dir + 'word_hyper/BLESS.all', 'POS' , "hyper"], [ dataset_dir + 'word_hyper/EVALution.all' , "POS",  "hyper" ], [ dataset_dir + 'word_hyper/LenciBenotto.all' , "POS",  "hyper" ], [ dataset_dir + 'word_hyper/Weeds.all' , "POS",  "hyper" ] ]


def processing_phrase(phrase_raw, POS_suffix):
    words = phrase_raw.split(',')
    output_list = []
    for i, w in enumerate(words):
        if POS_suffix and i == len(words) - 1:
            if w[-1] != 'n':
                return []
            else:
                output_list.append(w[:-2])
        else:
            output_list.append(w)
    return ' '.join(output_list)

def load_hyper(file_name, POS_suffix, all_pairs):
    not_noun_count = 0
    dataset = []
    with open(file_name) as f_in:
        for line in f_in:
            fields = line.rstrip().split('\t')
            hypo_candidate = processing_phrase(fields[0], POS_suffix)
            hyper_candidate = processing_phrase(fields[1], POS_suffix)
            if len(hypo_candidate) == 0 or len(hyper_candidate) == 0:
                #one of the phrase contains words which are not nouns
                not_noun_count += 1
                continue
            label = 0
            if fields[3] == 'hyper':
                label = 1
            dataset.append( [hypo_candidate, hyper_candidate, label] )
            all_pairs.append( [hypo_candidate, hyper_candidate, 'hyper'] )
    print("Throw away "+str(not_noun_count)+" pairs which are not nouns and keep "+str(len(all_pairs))+" pairs")
    return dataset

dataset_arr = []
all_pairs = []
for file_info in dataset_list:
    file_type = file_info[-1]
    print("loading ", file_info)
    if file_type == "hyper":
        POS_suffix = False
        if file_info[1] == "POS":
            POS_suffix = True
        dataset_arr.append( load_hyper(file_info[0], POS_suffix, all_pairs) )

print("loading ", embedding_file_name)
#word2emb, emb_size = utils.load_emb_file_to_dict(embedding_file_name)
with open(embedding_file_name) as f_in:
    embedding_list = json.load(f_in)

word2emb = {}
for w, proc_idx, avg_emb, cls_emb in embedding_list:
    #word2emb[w] = cls_emb
    word2emb[w] = [avg_emb , cls_emb]

pair_d2_score = {}
for i in range(len(all_pairs)):
    bigram, unigram, dataset = all_pairs[i]
    pred_score = []
    for j in range(2):
        bi_emb = word2emb[bigram][j]
        uni_emb = word2emb[unigram][j]
        score_pred = 1 - distance.cosine(uni_emb, bi_emb)
        pred_score.append(score_pred)
    pair_d2_score[ (bigram, unigram) ] = pred_score

method_names = ['avg' , 'cls']

def test_hyper_single(dataset, method_idx, method_name, pair_d2_score):
    score_list = []
    gt_list = []
    #uni_OOV_count = 0
    #bi_OOV_count = 0
    direction_list = []
    for bigram, unigram, label in dataset:
        #score_pred, uni_OOV_count, bi_OOV_count = output_sim_score(bigram, unigram, word2emb, uni_OOV_count, bi_OOV_count)
        score_pred = pair_d2_score[ (bigram, unigram) ][method_idx]
        score_list.append( score_pred )
        gt_list.append(label)
        if label == 1 and 'diff' in method_name:
            if score_pred == OOV_value or score_pred == 0:
                direction_list.append(0.5)
            elif score_pred > 0:
                direction_list.append(1)
            else:
                direction_list.append(0)
    AP, F1_best, acc_best = utils_testing.compute_AP_best_F1_acc(score_list, gt_list, correct_label = 1)
    print(method_name + ': AP@all '+ str(AP) + ', Best F1 ' + str(F1_best) + ', Best Accuracy ' + str(acc_best), ', Direction Accuracy ' + str(np.mean(direction_list)) )

def test_all_methods(dataset, pair_d2_score, method_names, file_type):
    num_methods = len(method_names)
    for method_idx, method_name in enumerate(method_names):
        if file_type == "hyper":
            test_hyper_single(dataset, method_idx, method_name, pair_d2_score)
            

for file_info, dataset in zip(dataset_list, dataset_arr):
    file_type = file_info[-1]
    print("testing ", file_info)
    test_all_methods(dataset, pair_d2_score, method_names, file_type)
    #if file_type == "SemEval2013":
    #    test_SemEval(dataset, pair_d2_score, method_names)
    #elif file_type == "Turney":
    #    test_Turney(dataset, pair_d2_score, method_names)

