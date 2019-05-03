import gensim
from scipy.spatial import distance
import random
import numpy as np
from sklearn.metrics import average_precision_score, f1_score

embedding_dir = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/resources/"

#lowercase_emb = True
#embedding_file_name = embedding_dir + "glove.840B.300d_filtered_wiki2016_min100.txt"
#embedding_file_name = embedding_dir + "lexvec_wiki2016_min100"
#embedding_file_name = embedding_dir + "word2vec_wiki2016_min100.txt"

#lower
lowercase_emb = False
#embedding_file_name = embedding_dir + "lexvec_enwiki_wiki2016_min100"
#embedding_file_name = embedding_dir + "paragram_wiki2016_min100"
embedding_file_name = embedding_dir + "glove.42B.300d_filtered_wiki2016_nchunk_lower_min100.txt"

#w2v = gensim.models.KeyedVectors.load_word2vec_format(emb_file_path, binary=False)

dataset_dir = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/"
#dataset_list = [ [dataset_dir + "SemEval2013/en.trainSet", "SemEval2013" ], [dataset_dir + "SemEval2013/en.testSet", "SemEval2013"], [dataset_dir + "Turney2012/jair.data", "Turney"] ]
dataset_list = [ [dataset_dir + "SemEval2013/train/en.trainSet.negativeInstances-v2", dataset_dir + "SemEval2013/train/en.trainSet.positiveInstances-v2", "SemEval2013" ], [dataset_dir + "Turney2012/Turney_train.txt", "Turney"] ]
#dataset_list = [ [dataset_dir + "SemEval2013/test/en.testSet.negativeInstances-v2", dataset_dir + "SemEval2013/test/en.testSet.positiveInstances-v2", "SemEval2013" ], [dataset_dir + "Turney2012/Turney_train.txt", "Turney"] ]

def load_emb_file(emb_file):
    with open(emb_file) as f_in:
        word2emb = {}
        for line in f_in:
            word_val = line.rstrip().split(' ')
            if len(word_val) < 3:
                continue
            word = word_val[0]
            val = np.array([float(x) for x in  word_val[1:]])
            if lowercase_emb:
                word_lower = word.lower()
                if word_lower not in word2emb:
                    word2emb[word_lower] = val
                else:
                    if word == word_lower:
                        word2emb[word_lower] = val
            else:
                word2emb[word] = val
            emb_size = len(val)
    return word2emb, emb_size

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

print("loading ", embedding_file_name)
word2emb, emb_size = load_emb_file(embedding_file_name)

def output_sim_score(bigram, unigram, word2emb, uni_OOV_count, bi_OOV_count):
    if unigram not in word2emb:
        uni_OOV_count += 1
        return 0, uni_OOV_count, bi_OOV_count
    else:
        uni_emb = word2emb[unigram]
    
    bi_emb = np.zeros( emb_size )
    count = 0
    for w in bigram:
        if w in word2emb:
            count += 1
            bi_emb += word2emb[w]
    if count == 0:
        bi_OOV_count += 1
        return 0, uni_OOV_count, bi_OOV_count
    else:
        bi_emb = bi_emb / count
    score_pred = 1 - distance.cosine(uni_emb, bi_emb)
    return score_pred, uni_OOV_count, bi_OOV_count

def test_Turney(dataset, word2emb):
    correct_count = 0
    correct_count_7 = 0
    total_count = 0
    uni_OOV_count = 0
    bi_OOV_count = 0
    for bigram, candidates in dataset:
        total_count += 1
        score_list = []
        for unigram, label in candidates:
            score_pred, uni_OOV_count, bi_OOV_count = output_sim_score(bigram, unigram, word2emb, uni_OOV_count, bi_OOV_count)
            score_list.append(score_pred)
        max_idx = np.argmax(score_list)
        if candidates[max_idx][1] == 1:
            correct_count += 1
        
        for unigram in bigram:
            score_pred, dummy_1, dummy_2 = output_sim_score(bigram, unigram, word2emb, uni_OOV_count, bi_OOV_count)
            score_list.append(score_pred)
        max_idx = np.argmax(score_list)
        if max_idx < len(candidates) and candidates[max_idx][1] == 1:
            correct_count_7 += 1
    
    print("accuracy 5 candidates", correct_count/float(total_count))
    print("accuracy 7 candidates", correct_count_7/float(total_count))
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

