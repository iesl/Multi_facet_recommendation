#import numpy as np
import torch
import json
import sys
#dist_option = 'sim_avg'
dist_option = 'sim_max'

import getopt

help_msg = '-s test_emb_file -r train_emb_file -p paper_to_reviewer_file -d dist_option -o output_file'

try:
    opts, args = getopt.getopt(sys.argv[1:], "s:r:p:o:d:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-s"):
        test_emb_file = arg
    elif opt in ("-r"):
        train_emb_file = arg
    elif opt in ("-p"):
        paper_to_reviewer_file = arg
    elif opt in ("-d"):
        dist_option = arg
    elif opt in ("-o"):
        output_file = arg

#test_emb_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/NeurIPS2019_emb_spector_raw.jsonl"
#train_emb_file ="/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/NeurIPS2019_emb_spector_raw_train_duplicate.jsonl"
#paper_to_reviewer_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/all_reviewer_paper_data"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/NeurIPS2019_spector_"+dist_option+".csv"

#test_emb_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/ICLR2020_emb_spector_raw.jsonl"
#train_emb_file ="/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/ICLR2020_emb_spector_raw_train_duplicate.jsonl"
#paper_to_reviewer_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020/all_reviewer_paper_data"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/ICLR2020_spector_"+dist_option+".csv"


device = torch.device('cpu')

def load_emb_file(f_in):
    #id_d2_emb = {}
    paper_emb_size_default = 768
    id_list = []
    emb_list = []
    bad_id_set = set()
    for line in f_in:
        paper_data = json.loads(line.rstrip())
        paper_id = paper_data['paper_id']
        paper_emb_size = len(paper_data['embedding'])
        assert paper_emb_size == 0 or paper_emb_size == paper_emb_size_default
        if paper_emb_size == 0:
            paper_emb = [0]*paper_emb_size_default
            bad_id_set.add(paper_id)
            #paper_emb_norm = np.zeros(paper_emb_size_default)
        else:
            #paper_emb = np.array(paper_data['embedding'])
            #paper_emb_norm = paper_emb/(np.linalg.norm(paper_emb)+0.000000000001)
            paper_emb = paper_data['embedding']
        #id_d2_emb[paper_id] = paper_emb_norm
        id_list.append(paper_id)
        emb_list.append(paper_emb)
    emb_tensor = torch.tensor(emb_list, device = device)
    emb_tensor = emb_tensor / (emb_tensor.norm(dim = 1, keepdim=True)+0.000000000001)
    print(len(bad_id_set))
    return emb_tensor, id_list, bad_id_set

with open(test_emb_file) as f_in:
    paper_emb_test, test_id_list, test_bad_id_set = load_emb_file(f_in)
    paper_num_test = len(test_id_list)

with open(train_emb_file) as f_in:
    paper_emb_train, train_id_list, train_bad_id_set = load_emb_file(f_in)
    paper_num_train = len(train_id_list)

    paper_id_d2_train_idx = {}
    for idx, paper_id in enumerate(train_id_list):
        paper_id_d2_train_idx[paper_id] = idx

#sys.exit()

user_d2_paper_idx = {}
    
with open(paper_to_reviewer_file) as f_in:
    for line in f_in:
        feature, feature_type, user_raw_list, all_authors, paper_id = line.rstrip().split('\t')
        if paper_id in train_bad_id_set:
            continue
        for user_raw in user_raw_list.split(','):
            suffix_start = user_raw.index('|')
            assert suffix_start > 0
            user = user_raw[:suffix_start]
            paper_idx_list = user_d2_paper_idx.get(user,[])
            paper_idx_list.append(paper_id_d2_train_idx[paper_id])
            user_d2_paper_idx[user] = paper_idx_list

#user_num = len(user_d2_paper_idx)

p2p_aff = torch.empty( (paper_num_test, paper_num_train) ,device=device)

for i in range(paper_num_test):
    p2p_aff[i,:] = torch.sum(paper_emb_test[i,:].unsqueeze(dim=0) * paper_emb_train, dim=1)

with open(output_file,'w') as f_out:
    #paper_user_dist = np.empty( (paper_num_test, user_num), dtype=np.float16 )
    #for j in range(user_num):
    for user in user_d2_paper_idx:
        train_paper_id_list = user_d2_paper_idx[user]
        if len(train_paper_id_list) == 0:
            continue
        train_paper_aff_j = p2p_aff[:, train_paper_id_list]
    #for i in range(paper_num_test):
        if dist_option == 'sim_avg':
            all_paper_aff = train_paper_aff_j.mean(dim = 1)
        elif dist_option == 'sim_max':
            all_paper_aff = train_paper_aff_j.max(dim = 1)[0]
        for j in range(paper_num_test):
            f_out.write( ','.join( [ test_id_list[j], user, str(all_paper_aff[j].item())]) + '\n'  )

