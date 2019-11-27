import gensim
from scipy.spatial import distance
import random
import numpy as np
from sklearn.metrics import average_precision_score, f1_score
import sys
sys.path.insert(0, sys.path[0]+'/../..')
import utils
import json
from scipy import stats

#embedding_file_name = "gen_log/BERT_SemEval2013_Turney2012_phrase_train.json"
#embedding_file_name = "gen_log/BERT_SemEval2013_Turney2012_phrase_test.json"
#embedding_file_name = "gen_log/BERT_large_BiRD_phrase_test.json"
#embedding_file_name = "gen_log/BERT_base_BiRD_phrase_test.json"
embedding_file_name = "gen_log/BERT_base_BiRD_phrase_test_cased.json"
#embedding_file_name = "gen_log/BERT_large_WikiSRS_rel_sim_phrase_test.json"
#embedding_file_name = "gen_log/BERT_base_WikiSRS_rel_sim_phrase_test.json"
#embedding_file_name = "gen_log/BERT_base_WikiSRS_rel_sim_phrase_test_cased.json"
#embedding_file_name = "gen_log/ELMo_WikiSRS_rel_sim_phrase_test.json"
#embedding_file_name = "gen_log/BERT_large_SemEval2013_Turney2012_phrase_test.json"

#dataset_dir = "/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/phrase/"
dataset_dir = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/phrase/"
#dataset_list = [ [dataset_dir + "SemEval2013/en.trainSet", "SemEval2013" ], [dataset_dir + "SemEval2013/en.testSet", "SemEval2013"], [dataset_dir + "Turney2012/jair.data", "Turney"] ]
#dataset_list = [ [dataset_dir + "SemEval2013/train/en.trainSet.negativeInstances-v2", dataset_dir + "SemEval2013/train/en.trainSet.positiveInstances-v2", "SemEval2013" ], [dataset_dir + "Turney2012/Turney_train.txt", "Turney"] ]
#dataset_list = [ [dataset_dir + "SemEval2013/test/en.testSet.negativeInstances-v2", dataset_dir + "SemEval2013/test/en.testSet.positiveInstances-v2", "SemEval2013" ], [dataset_dir + "Turney2012/Turney_test.txt", "Turney"] ]
dataset_list = [ [dataset_dir + "BiRD/BiRD.txt", 'BiRD'] ]
#dataset_list = [ [dataset_dir + "WikiSRS/WikiSRS_relatedness.csv", 'WikiSRS'], [dataset_dir + "WikiSRS/WikiSRS_similarity.csv", 'WikiSRS'] ]

def reverse_bigram(bigram):
    #w_list = bigram.split()
    #return w_list[1]+' '+w_list[0]
    return [bigram[1],bigram[0]]

def load_Turney(file_name):
    dataset = []
    with open(file_name) as f_in:
        for line in f_in:
            line = line.rstrip().replace(' | ', '|')
            fields = line.split('|')
            bigram = fields[0].split()
            candidates = [ (fields[1],1) ] + [ (x,0) for x in fields[4:] ] #assume that 1 is correct, and 2, 3 are the candidate we want to remove
            random.shuffle(candidates)
            dataset.append( [bigram, candidates] )
    return dataset

def load_BiRD(file_name):
    pair_list = []
    with open(file_name) as f_in:
        for i, line in enumerate(f_in):
            if i == 0:
                continue
            fields = line.rstrip().split('\t')
            pair_list.append( [ tuple(fields[1].split()), tuple(fields[2].split()), float(fields[6])])
    return pair_list

def load_WikiSRS(file_name):
    pair_list = []
    with open(file_name) as f_in:
        for i, line in enumerate(f_in):
            if i == 0:
                continue
            fields = line.rstrip().split('\t')
            pair_list.append( [ tuple(fields[2].split()), tuple(fields[3].split()), float(fields[4])])
            #pair_list.append( [fields[2].lower(), fields[3].lower(), float(fields[4])])
    return pair_list

def load_pairs(file_name, label):
    pair_list = []
    with open(file_name) as f_in:
        for line in f_in:
            unigram, bigram = line.rstrip().split('\t')
            pair_list.append( [bigram.split(), unigram, label])
    return pair_list

def load_SemEval(neg_file, pos_file):
    neg_pairs = load_pairs(neg_file, 0)
    pos_pairs = load_pairs(pos_file, 1)
    dataset = neg_pairs + pos_pairs
    random.shuffle(dataset)
    return dataset

dataset_arr = []
for file_info in dataset_list:
    file_type = file_info[-1]
    print("loading ", file_info)
    if file_type == "SemEval2013":
        dataset_arr.append( load_SemEval( file_info[0], file_info[1] ) )
    elif file_type == "Turney":
        dataset_arr.append( load_Turney( file_info[0] ) )
    elif file_type == "BiRD":
        dataset_arr.append( load_BiRD( file_info[0] ) )
    elif file_type == "WikiSRS":
        dataset_arr.append( load_WikiSRS( file_info[0] ) )

print("loading ", embedding_file_name)
#word2emb, emb_size = utils.load_emb_file_to_dict(embedding_file_name)
with open(embedding_file_name) as f_in:
    embedding_list = json.load(f_in)

word2emb = {}
for w, proc_idx, avg_emb, cls_emb in embedding_list:
    #word2emb[w] = cls_emb
    word2emb[w] = avg_emb


def output_sim_score(bigram, unigram, word2emb, uni_OOV_count, bi_OOV_count):
    #if unigram not in word2emb:
    #    uni_OOV_count += 1
    #    return 0, uni_OOV_count, bi_OOV_count
    #else:
    #    uni_emb = word2emb[unigram]
    
    #bi_emb = np.zeros( emb_size )
    #count = 0
    #for w in bigram:
    #    if w in word2emb:
    #        count += 1
    #        bi_emb += word2emb[w]
    #if count == 0:
    #    bi_OOV_count += 1
    #    return 0, uni_OOV_count, bi_OOV_count
    #else:
    #    bi_emb = bi_emb / count
    uni_emb = word2emb[unigram]
    bi_emb = word2emb[' '.join(bigram)]
    score_pred = 1 - distance.cosine(uni_emb, bi_emb)
    return score_pred, uni_OOV_count, bi_OOV_count

def output_sim_score_BiRD(bigram, unigram, word2emb, uni_OOV_count, bi_OOV_count):
    uni_emb = word2emb[' '.join(unigram)]
    bi_emb = word2emb[' '.join(bigram)]
    score_pred = 1 - distance.cosine(uni_emb, bi_emb)
    return score_pred, uni_OOV_count, bi_OOV_count

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


def test_BiRD(dataset, word2emb):
    score_list = []
    gt_list = []
    bigram_d2_pred = {}
    bigram_d2_gt = {}
    uni_OOV_count = 0
    bi_OOV_count = 0
    for bigram, unigram, label in dataset:
        score_pred, uni_OOV_count, bi_OOV_count = output_sim_score_BiRD(bigram, unigram, word2emb, uni_OOV_count, bi_OOV_count)
        score_list.append( score_pred )
        gt_list.append(label)
        if bigram not in bigram_d2_pred:
            bigram_d2_pred[bigram] = []
            bigram_d2_gt[bigram] = []
        bigram_d2_pred[bigram].append(score_pred)
        bigram_d2_gt[bigram].append(label)
    pearson_score_sum = 0
    rank_score_sum = 0
    weight_sum = 0.0
    for bigram in bigram_d2_pred:
        if len(bigram_d2_pred[bigram]) < 2:
            continue
        weight = len(bigram_d2_gt[bigram])
        weight_sum += weight
        pearson_score_sum += weight * stats.pearsonr(bigram_d2_pred[bigram], bigram_d2_gt[bigram])[0]
        rank_score_sum += weight * stats.spearmanr(bigram_d2_pred[bigram], bigram_d2_gt[bigram])[0]

    global_rank_results = stats.spearmanr(score_list, gt_list)
    global_linear_results = stats.pearsonr(score_list, gt_list)
    print('Pearson '+ str(pearson_score_sum/weight_sum)+ ', Spearman rank '+ str(rank_score_sum/weight_sum) + ', Global Pearson '+ str(global_linear_results[0])+ ', Spearman rank '+ str(global_rank_results[0]) )


def test_Turney(dataset, word2emb):
    correct_count = 0
    correct_count_7 = 0
    correct_count_10 = 0
    correct_count_14 = 0
    total_count = 0
    uni_OOV_count = 0
    bi_OOV_count = 0
    for bigram, candidates in dataset:
        total_count += 1
        score_list = []
        for unigram, label in candidates:
            score_pred, uni_OOV_count, bi_OOV_count = output_sim_score(bigram, unigram, word2emb, uni_OOV_count, bi_OOV_count)
            score_list.append(score_pred)
        correct_count = update_correct_count(score_list, candidates, correct_count)        

        score_list_10 = score_list.copy()
        for unigram, label in candidates:
            score_pred, uni_OOV_count, bi_OOV_count = output_sim_score( reverse_bigram(bigram), unigram, word2emb, uni_OOV_count, bi_OOV_count)
            score_list_10.append(score_pred)
        correct_count_10 = update_correct_count(score_list_10, candidates, correct_count_10)

        #max_idx = np.argmax(score_list)
        #if candidates[max_idx][1] == 1:
        #    correct_count += 1
        score_5_7_list = []
        for unigram in bigram:
            score_pred, dummy_1, dummy_2 = output_sim_score(bigram, unigram, word2emb, uni_OOV_count, bi_OOV_count)
            score_5_7_list.append(score_pred)
        correct_count_7 = update_correct_count(score_list+score_5_7_list, candidates, correct_count_7)

        score_list_14 = score_list_10 + score_5_7_list
        for unigram in bigram:
            score_pred, dummy_1, dummy_2 = output_sim_score(reverse_bigram(bigram), unigram, word2emb, uni_OOV_count, bi_OOV_count)
            score_list_14.append(score_pred)
        correct_count_14 = update_correct_count(score_list_14, candidates, correct_count_14)
        #max_idx = np.argmax(score_list)
        #if max_idx < len(candidates) and candidates[max_idx][1] == 1:
        #    correct_count_7 += 1
    
    print(': acc (5 c) ' + str(correct_count/float(total_count)) + ', acc (7 c) ' + str(correct_count_7/float(total_count)) + 'acc (10 c) ' + str(correct_count_10/float(total_count)) + ', acc (14 c) ' + str(correct_count_14/float(total_count)) )
    #print("accuracy 5 candidates", correct_count/float(total_count))
    #print("accuracy 7 candidates", correct_count_7/float(total_count))
    print("total ", len(candidates)*total_count, ", unigram OOV ", uni_OOV_count, ", bigram OOV ", bi_OOV_count )

def test_SemEval(dataset, word2emb):
    score_list = []
    gt_list = []
    uni_OOV_count = 0
    bi_OOV_count = 0
    for bigram, unigram, label in dataset:
        score_pred, uni_OOV_count, bi_OOV_count = output_sim_score(bigram, unigram, word2emb, uni_OOV_count, bi_OOV_count)
        score_list.append( score_pred)
        gt_list.append(label)
    sorted_idx = np.argsort(score_list)
    sorted_idx = sorted_idx.tolist()[::-1]
    #print(sorted_idx)
    total_correct = sum(gt_list)
    correct_count = 0
    total_count = 0
    precision_list = []
    F1_list = []
    for idx in sorted_idx:
        total_count += 1
        if gt_list[idx] == 1:
            correct_count += 1
            precision = correct_count/float(total_count)
            precision_list.append( precision )
            recall = correct_count / float(total_correct) 
            F1_list.append(  2*(precision*recall)/(recall+precision) )
    print("AP@all ", np.mean(precision_list))
    print(average_precision_score(gt_list, score_list) )
    print("Best F1 ", np.max(F1_list))
    #print("F1 ", f1_score(gt_list, score_list) )
    print("total ", len(gt_list), ", unigram OOV ", uni_OOV_count, ", bigram OOV ", bi_OOV_count )
            

for file_info, dataset in zip(dataset_list, dataset_arr):
    file_type = file_info[-1]
    print("testing ", file_info)
    if file_type == "SemEval2013":
        test_SemEval(dataset, word2emb)
    elif file_type == "Turney":
        test_Turney(dataset, word2emb)
    elif file_type == "BiRD":
        test_BiRD(dataset, word2emb)
    elif file_type == "WikiSRS":
        test_BiRD(dataset, word2emb)
