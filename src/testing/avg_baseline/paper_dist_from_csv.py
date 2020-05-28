import sys
sys.path.insert(0, sys.path[0]+'/../..')
import utils
import numpy as np

from unicodedata import normalize
#from chardet import detect

# get file encoding type
#def get_encoding_type(file):
#    with open(file, 'rb') as f:
#        rawdata = f.read()
#    return detect(rawdata)['encoding']


data_folder = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/processed/NeurIPS2019_bid_score_gorc_uncased'
#data_folder = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/processed/UAI2019_gorc_uncased'
#data_folder = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/processed/UAI2019_bid_score_gorc_uncased'
#data_folder = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/processed/ICLR2020_gorc_uncased'
#data_folder = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/processed/ICLR2020_bid_score_gorc_uncased'
#data_folder = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/processed/ICLR2019_gorc_uncased'
#data_folder = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/processed/ICLR2019_bid_score_gorc_uncased'
test_paper_id_file = data_folder + '/paper_id_test'
user_dict_file = data_folder + '/user/dictionary_index'
score_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/source_data/OpenReviewTestData/tpms.csv'
output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/NeurIPS2019_TPMS_dist.np'
#score_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/ELMo-TPMS-scores/iclr2019TPMS.csv'
#output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/ICLR2019_TPMS_dist.np'
#output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/ICLR2019_bid_score_TPMS_dist.np'
#score_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/ELMo-TPMS-scores/ICLR2019_ELMo_scores.csv'
#output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/ICLR2019_ELMo_dist.np'
#output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/ICLR2019_bid_score_ELMo_dist.np'
#score_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/ELMo-TPMS-scores/ICLR2020_ELMo_scores.csv'
#output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/ICLR2020_ELMo_dist.np'
#output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/ICLR2020_bid_score_ELMo_dist.np'
#score_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/ELMo-TPMS-scores/UAI2019_ELMo_scores.csv'
#score_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/ELMo-TPMS-scores/UAI2019TPMS.csv'
#output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/UAI2019_ELMo_dist.np'
#output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/UAI2019_TPMS_dist.np'
#output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/UAI2019_bid_score_ELMo_dist.np'
#output_file = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/UAI2019_bid_score_TPMS_dist.np'

#UAI2019_ELMo_scores.csv  
#UAI2019TPMS.csv
#from_codec = get_encoding_type(user_dict_file)
#print(from_codec)
#from_codec = get_encoding_type(score_file)
#print(from_codec)

special_word_set = set(['[null]','<unk>','<eos>'])

with open(user_dict_file) as f_in:
    user_raw_d2_idx_freq, user_idx_max = utils.load_word_dict(f_in)
user_num = user_idx_max + 1

user_d2_idx = {}
for user_raw in user_raw_d2_idx_freq:
    if user_raw in special_word_set:
        continue
    suffix_start = user_raw.index('|')
    assert suffix_start > 0
    user = user_raw[:suffix_start]
    user_d2_idx[normalize('NFC',user)] = user_raw_d2_idx_freq[user_raw][0]

#paper_id_list = []
paper_id_d2_idx = {}
with open(test_paper_id_file) as f_in:
    for idx, line in enumerate(f_in):
        paper_id = line.rstrip()
        #paper_id_list.append(paper_id)
        paper_id_d2_idx[paper_id] = idx

paper_num = len(paper_id_d2_idx)

num_filled_val = 0
paper_user_dist  = np.ones((paper_num, user_num), dtype=np.float16)
false_pos_user = set()
true_pos_user = set()
false_pos_paper = set()
true_pos_paper = set()
with open(score_file) as f_in:
    for line in f_in:
        paper_id, user, score = line.rstrip().split(',')
        #paper_id = paper_id.encode('ascii', "ignore")
        #user = unicode(user)
        user = normalize('NFC', user)
        if paper_id not in paper_id_d2_idx:
            false_pos_paper.add(paper_id)
            continue
        true_pos_paper.add(paper_id)
        paper_idx = paper_id_d2_idx[paper_id]
        if user not in user_d2_idx:
            false_pos_user.add(user)
            continue
        true_pos_user.add(user)
        user_idx = user_d2_idx[user]
        paper_user_dist[paper_idx,user_idx] = 1 - float(score)
        num_filled_val += 1
print("1 - density of the distance matrix {}".format(1 - num_filled_val/float(paper_num * user_num)))
print("User in score file but not in corpus (might include the cases where the author does not write any paper): {}".format(false_pos_user))
print("Paper in score file but not in corpus: {}".format(false_pos_paper))
print("User in corpus but not in score file: {}".format(set(user_d2_idx.keys()) - true_pos_user))
print("Ppaer in corpus but not in score file: {}".format(set(paper_id_d2_idx.keys()) - true_pos_paper))

np.savetxt(output_file, paper_user_dist)
