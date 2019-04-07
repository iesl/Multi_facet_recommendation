import sys
sys.path.insert(0, sys.path[0]+'/../..')
import utils_testing
#import utils

topic_file_name = "./gen_log/STS_dev_updated_glove_lc_elayer2_bsz200_ep5_linear.json"
#topic_file_name = "./gen_log/STS_dev_glove_lc_elayer2_bsz200_ep7_linear.json"
#topic_file_name = "./gen_log/STS_dev_glove_lc_elayer2_bsz200_ep7_posi.json"
#topic_file_name = "./gen_log/STS_dev_word2vec_lc_elayer1_bsz200_ep4.json"

gt_file_name = "./dataset_testing/STS/stsbenchmark/sts-dev.csv"

device = 'cpu'
bsz = 100
L1_losss_B = 0.2

def load_STS_file(f_in):
    output_list = []
    for line in f_in:
        genre, source, source_year, org_idx, score, sent_1, sent_2 = line.rstrip().split('\t')
        output_list.append([sent_1, sent_2, float(score), genre+'-'+source])
    return output_list

with open(gt_file_name) as f_in:
    testing_list = load_STS_file(f_in)

with open(topic_file_name) as f_in:
    sent_d2_topics = utils_testing.load_prediction_from_json(f_in)

testing_pair_loader = utils_testing.build_loader_from_pairs(testing_list, sent_d2_topics, bsz, device)

pred_scores = utils_testing.predict_sim_scores(testing_pair_loader, L1_losss_B, device)

def summarize_prediction_STS(pred_scores, testing_list):
    

summarize_prediction_STS(pred_scores, testing_list)
