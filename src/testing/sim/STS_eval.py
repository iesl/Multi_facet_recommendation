import sys
sys.path.insert(0, sys.path[0]+'/../..')
import utils_testing
from scipy.stats import spearmanr, pearsonr
import numpy as np
import torch
#import utils

#topic_file_name = "./gen_log/STS_dev_updated_glove_lc_elayer2_bsz200_ep6_linear_cosine.json"
#topic_file_name = "./gen_log/STS_dev_glove_lc_elayer2_bsz200_ep7_linear_cosine.json"
#topic_file_name = "./gen_log/STS_dev_glove_lc_elayer2_bsz200_ep7_posi_cosine.json"
#topic_file_name = "./gen_log/STS_dev_word2vec_lc_elayer1_bsz200_ep4.json"
#topic_file_name = "./gen_log/STS_dev_word2vec_lc_elayer1_bsz200_ep7_linear_cosine.json"
topic_file_name = "./gen_log/STS_dev_wiki2016_glove_lc_bsz200_ep2_0.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_glove_maxlc_bsz200_ep2_0.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_updated_glove_maxlc_bsz200_ep3_0.json"
#topic_file_name = "./gen_log/STS_dev_wiki2016_word2vec_maxlc_bsz200_ep2_0.json"

print(topic_file_name)

gt_file_name = "./dataset_testing/STS/stsbenchmark/sts-dev.csv"

device = 'cpu'
bsz = 100
L1_losss_B = 0.2

def load_STS_file(f_in):
    output_list = []
    for line in f_in:
        #print(line.rstrip().split('\t'))
        fields = line.rstrip().split('\t')
        genre, source, source_year, org_idx, score, sent_1, sent_2 = fields[:7]
        output_list.append([sent_1, sent_2, float(score), genre+'-'+source])
    return output_list

with open(gt_file_name) as f_in:
    testing_list = load_STS_file(f_in)

with open(topic_file_name) as f_in:
    sent_d2_topics = utils_testing.load_prediction_from_json(f_in)

testing_pair_loader = utils_testing.build_loader_from_pairs(testing_list, sent_d2_topics, bsz, device)

with torch.no_grad():
    pred_scores = utils_testing.predict_sim_scores(testing_pair_loader, L1_losss_B, device)

print(len(pred_scores))
def summarize_prediction_STS(pred_scores, testing_list):
    genre_s_d2_scores = {'all': [], 'lower': [], 'higher': []}
    field_lists = list(zip(*testing_list))
    score_list = field_lists[2]
    sorted_ind = np.argsort(score_list)
    print(score_list[sorted_ind[0]])
    lower_idx_set = set(sorted_ind[:int(len(sorted_ind)/2)])
    print( len(lower_idx_set) )
    for i in range(len(testing_list)):
        genre_s = testing_list[i][-1]
        if genre_s not in genre_s_d2_scores:
            genre_s_d2_scores[genre_s] = []
        genre_s_d2_scores[genre_s].append( pred_scores[i] + [testing_list[i][2]] )
        genre_s_d2_scores['all'].append( pred_scores[i] + [testing_list[i][2]] )
        if i in lower_idx_set:
            genre_s_d2_scores['lower'].append(pred_scores[i] + [testing_list[i][2]] )
        else:
            genre_s_d2_scores['higher'].append(pred_scores[i] + [testing_list[i][2]] )
    for genre_s in genre_s_d2_scores:
        pred_and_gt = list( zip(*genre_s_d2_scores[genre_s]) ) 
        print("genre_source: ", genre_s)
        #print(pred_and_gt)
        for m in range(len(pred_and_gt)-1):
            print("method", m, spearmanr( pred_and_gt[m], pred_and_gt[-1] ))
            print("method", m, pearsonr( pred_and_gt[m], pred_and_gt[-1] ))


summarize_prediction_STS(pred_scores, testing_list)
