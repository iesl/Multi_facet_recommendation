import random
import numpy as np
from sklearn.metrics import average_precision_score, f1_score
import sys
sys.path.insert(0, sys.path[0]+'/../..')
import utils
import utils_testing
import torch

import getopt
help_msg = '-t <topic_file_name> -w <w_emb_file_name> -d <freq_file_name> -g <train_or_test> -l <upper_emb_to_lower>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "t:w:d:g:l:")
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
        embedding_file_name = arg
    elif opt in ("-d"):
        freq_file_name = arg
    elif opt in ("-g"):
        train_or_test= int(arg)
    elif opt in ("-l"):
        lowercase_emb= int(arg)

#embedding_dir = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/resources/"
#embedding_dir = "./resources/"


#lowercase_emb = True
#embedding_file_name = embedding_dir + "glove.840B.300d_filtered_wiki2016_min100.txt"
#embedding_file_name = embedding_dir + "lexvec_wiki2016_min100"
#embedding_file_name = embedding_dir + "word2vec_wiki2016_min100.txt"

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
#embedding_file_name = embedding_dir + "glove.42B.300d_filtered_wiki2016_nchunk_lower_min100.txt"
#topic_file_name = "./gen_log/phrase_train_wiki2016_lex_enwiki_trans_d400_bsz400_ep1_5.json"
#embedding_file_name = embedding_dir + "lexvec_enwiki_wiki2016_min100"

#freq_file_name = "./data/processed/wiki2016_nchunk_lower_min100/dictionary_index"

#w2v = gensim.models.KeyedVectors.load_word2vec_format(emb_file_path, binary=False)

dataset_dir = "./dataset_testing/phrase/"
#dataset_list = [ [dataset_dir + "SemEval2013/en.trainSet", "SemEval2013" ], [dataset_dir + "SemEval2013/en.testSet", "SemEval2013"], [dataset_dir + "Turney2012/jair.data", "Turney"] ]
if train_or_test == 'train':
    dataset_list = [ [dataset_dir + "SemEval2013/train/en.trainSet.negativeInstances-v2", dataset_dir + "SemEval2013/train/en.trainSet.positiveInstances-v2", "SemEval2013" ], [dataset_dir + "Turney2012/Turney_train.txt", "Turney"] ]
elif train_or_test == 'test'::
    dataset_list = [ [dataset_dir + "SemEval2013/test/en.testSet.negativeInstances-v2", dataset_dir + "SemEval2013/test/en.testSet.positiveInstances-v2", "SemEval2013" ], [dataset_dir + "Turney2012/Turney_test.txt", "Turney"] ]

bsz = 100

device = 'cuda'
L1_losss_B = 0.2

print(topic_file_name)

#def load_emb_file(emb_file):
#    with open(emb_file) as f_in:
#        word2emb = {}
#        for line in f_in:
#            word_val = line.rstrip().split(' ')
#            if len(word_val) < 3:
#                continue
#            word = word_val[0]
#            val = np.array([float(x) for x in  word_val[1:]])
#            if lowercase_emb:
#                word_lower = word.lower()
#                if word_lower not in word2emb:
#                    word2emb[word_lower] = val
#                else:
#                    if word == word_lower:
#                        word2emb[word_lower] = val
#            else:
#                word2emb[word] = val
#            emb_size = len(val)
#    return word2emb, emb_size

def reverse_bigram(bigram):
    w_list = bigram.split()
    return w_list[1]+' '+w_list[0]

def load_Turney(file_name, all_pairs):
    dataset = []
    with open(file_name) as f_in:
        for line in f_in:
            line = line.rstrip().replace(' | ', '|')
            fields = line.split('|')
            #bigram = fields[0].split()
            bigram = fields[0]
            candidates = [ (fields[1],1) ] + [ (x,0) for x in fields[4:] ] #assume that 1 is correct, and 2, 3 are the candidate we want to remove
            random.shuffle(candidates)
            dataset.append( [bigram, candidates] )
            for unigram in fields[1:]:
                #all_pairs.append( [fields[0], unigram, "Turney"] )
                all_pairs.append( [bigram, unigram, "Turney"] )
                all_pairs.append( [reverse_bigram(bigram), unigram, "Turney"] )
    return dataset

def load_pairs(file_name, label, all_pairs):
    pair_list = []
    with open(file_name) as f_in:
        for line in f_in:
            unigram, bigram = line.rstrip().split('\t')
            #pair_list.append( [bigram.split(), unigram, label])
            pair_list.append( [bigram, unigram, label])
            all_pairs.append( [bigram, unigram, "SemEval"] )
    return pair_list

def load_SemEval(neg_file, pos_file, all_pairs):
    neg_pairs = load_pairs(neg_file, 0, all_pairs)
    pos_pairs = load_pairs(pos_file, 1, all_pairs)
    dataset = neg_pairs + pos_pairs
    random.shuffle(dataset)
    return dataset

dataset_arr = []
all_pairs = []
for file_info in dataset_list:
    file_type = file_info[-1]
    print("loading ", file_info)
    if file_type == "SemEval2013":
        dataset_arr.append( load_SemEval( file_info[0], file_info[1], all_pairs ) )
    elif file_type == "Turney":
        dataset_arr.append( load_Turney( file_info[0], all_pairs ) )

with open(freq_file_name) as f_in:
    word_d2_idx_freq, max_ind = utils.load_word_dict(f_in)

utils_testing.compute_freq_prob(word_d2_idx_freq)

print("load ", topic_file_name)
with open(topic_file_name) as f_in:
    sent_d2_topics = utils_testing.load_prediction_from_json(f_in)

print("build dataloader")
testing_pair_loader, other_info = utils_testing.build_loader_from_pairs(all_pairs, sent_d2_topics, bsz, device)

print("loading ", embedding_file_name)
word2emb, emb_size = utils.load_emb_file_to_dict(embedding_file_name)

with torch.no_grad():
    pred_scores, method_names = utils_testing.predict_sim_scores(testing_pair_loader, L1_losss_B, device, word2emb, other_info, word_d2_idx_freq)

pair_d2_score = {}
for i in range(len(all_pairs)):
    bigram, unigram, dataset = all_pairs[i]
    pair_d2_score[ (bigram, unigram) ] = pred_scores[i]

def add_scores(bigram, candidates, score_list, pair_d2_score, method_idx):
    for ele in candidates:
        if type(ele) == str:
            unigram = ele
        elif len(ele) == 2:
            unigram = ele[0]
        #score_pred, uni_OOV_count, bi_OOV_count = output_sim_score(bigram, unigram, word2emb, uni_OOV_count, bi_OOV_count)
        score_pred = pair_d2_score[ (bigram, unigram) ][method_idx]
        score_list.append(score_pred)


def max_break_tie_randomly(input_list):
    max_val = -100000000000
    max_buffer = []
    for i, ele in enumerate(input_list):
        if ele > max_val:
            max_val = ele
            max_buffer = [i]
        elif ele == max_val:
            max_buffer.append(i)
    return random.choice(max_buffer)

def update_correct_count(score_list, candidates, correct_count):
    max_idx = max_break_tie_randomly(score_list)
    if max_idx < len(candidates) and candidates[max_idx][1] == 1:
        correct_count += 1
    return correct_count

def test_Turney_single(dataset, method_idx, method_name, pair_d2_score):
    correct_count = 0
    correct_count_10 = 0
    correct_count_7 = 0
    correct_count_14 = 0
    total_count = 0
    #uni_OOV_count = 0
    #bi_OOV_count = 0
    for bigram, candidates in dataset:
        total_count += 1
        
        score_list = []
        add_scores(bigram, candidates, score_list, pair_d2_score, method_idx)
        correct_count = update_correct_count(score_list, candidates, correct_count)
        
        score_list_10 = score_list.copy()
        add_scores(reverse_bigram(bigram), candidates, score_list_10, pair_d2_score, method_idx)
        correct_count_10 = update_correct_count(score_list_10, candidates, correct_count_10)

        #for unigram in bigram:
        score_5_7_list = []
        add_scores(bigram, bigram.split(), score_5_7_list, pair_d2_score, method_idx)
        #for unigram in bigram.split():
        #    score_pred = pair_d2_score[ (bigram, unigram) ][method_idx]
        #    score_list.append(score_pred)
        score_list_7 = score_list + score_5_7_list
        correct_count_7 = update_correct_count(score_list_7, candidates, correct_count_7)        
        #max_idx = np.argmax(score_list_7)
        #if max_idx < len(candidates) and candidates[max_idx][1] == 1:
        #    correct_count_7 += 1
        
        score_list_14 = score_list_10 + score_5_7_list
        add_scores(reverse_bigram(bigram), bigram.split(), score_list_14, pair_d2_score, method_idx)
        correct_count_14 = update_correct_count(score_list_14, candidates, correct_count_14)
    
    print(method_name + ': acc (5 c) ' + str(correct_count/float(total_count)) + ', acc (7 c) ' + str(correct_count_7/float(total_count)) + 'acc (10 c) ' + str(correct_count_10/float(total_count)) + ', acc (14 c) ' + str(correct_count_14/float(total_count)) )
    #print(method_name)
    #print("accuracy 5 candidates", correct_count/float(total_count))
    #print("accuracy 7 candidates", correct_count_7/float(total_count))
    #print("total ", len(candidates)*total_count, ", unigram OOV ", uni_OOV_count, ", bigram OOV ", bi_OOV_count )


    
def test_SemEval_single(dataset, method_idx, method_name, pair_d2_score):
    score_list = []
    gt_list = []
    #uni_OOV_count = 0
    #bi_OOV_count = 0
    for bigram, unigram, label in dataset:
        #score_pred, uni_OOV_count, bi_OOV_count = output_sim_score(bigram, unigram, word2emb, uni_OOV_count, bi_OOV_count)
        score_pred = pair_d2_score[ (bigram, unigram) ][method_idx]
        score_list.append( score_pred )
        gt_list.append(label)
    AP, F1_best, acc_best = utils_testing.compute_AP_best_F1_acc(score_list, gt_list, correct_label = 1)
    print(method_name + ': AP@all '+ str(AP) + ', Best F1 ' + str(F1_best) + ', Best Accuracy ' + str(acc_best) )
    #sorted_idx = np.argsort(score_list)
    #sorted_idx = sorted_idx.tolist()[::-1]
    ##print(sorted_idx)
    #total_correct = sum(gt_list)
    #correct_count = 0
    #total_count = 0
    #precision_list = []
    #F1_list = []
    #for idx in sorted_idx:
    #    total_count += 1
    #    if gt_list[idx] == 1:
    #        correct_count += 1
    #        precision = correct_count/float(total_count)
    #        precision_list.append( precision )
    #        recall = correct_count / float(total_correct) 
    #        F1_list.append(  2*(precision*recall)/(recall+precision) )
    
    #print("AP@all ", np.mean(precision_list))
    #print(average_precision_score(gt_list, score_list) )
    #print("Best F1 ", np.max(F1_list))
    #print("F1 ", f1_score(gt_list, score_list) )
    #print("total ", len(gt_list), ", unigram OOV ", uni_OOV_count, ", bigram OOV ", bi_OOV_count )

def test_all_methods(dataset, pair_d2_score, method_names, file_type):
    num_methods = len(method_names)
    for method_idx, method_name in enumerate(method_names):
        if file_type == "SemEval2013":
            test_SemEval_single(dataset, method_idx, method_name, pair_d2_score)
        elif file_type == "Turney":
            test_Turney_single(dataset, method_idx, method_name, pair_d2_score)
            

for file_info, dataset in zip(dataset_list, dataset_arr):
    file_type = file_info[-1]
    print("testing ", file_info)
    test_all_methods(dataset, pair_d2_score, method_names, file_type)
    #if file_type == "SemEval2013":
    #    test_SemEval(dataset, pair_d2_score, method_names)
    #elif file_type == "Turney":
    #    test_Turney(dataset, pair_d2_score, method_names)

