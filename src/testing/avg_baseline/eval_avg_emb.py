import sys
sys.path.insert(0, sys.path[0]+'/../../')
import utils
import utils_testing
import torch
import numpy as np



#input_dict_path = "./data/processed/UAI2019_bid_score_scibert_gorc_uncased/feature/word_freq"
#user_dict_path = "./data/processed/UAI2019_bid_score_scibert_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/UAI2019_scibert_gorc_uncased/feature/word_freq"
#user_dict_path = "./data/processed/UAI2019_scibert_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/UAI2019_bid_score_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/UAI2019_bid_score_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/UAI2019_old_bid_score_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/UAI2019_old_bid_score_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/UAI2019_bid_high_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/UAI2019_bid_high_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/UAI2019_bid_low_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/UAI2019_bid_low_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/UAI2019_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/UAI2019_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/UAI2019_old_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/UAI2019_old_gorc_uncased/user/dictionary_index"

input_dict_path = "./data/processed/ICLR2018_bid_score_scibert_gorc_uncased/feature/word_freq"
user_dict_path = "./data/processed/ICLR2018_bid_score_scibert_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/ICLR2018_scibert_gorc_uncased/feature/word_freq"
#user_dict_path = "./data/processed/ICLR2018_scibert_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/ICLR2019_bid_score_scibert_gorc_uncased/feature/word_freq"
#user_dict_path = "./data/processed/ICLR2019_bid_score_scibert_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/ICLR2019_scibert_gorc_uncased/feature/word_freq"
#user_dict_path = "./data/processed/ICLR2019_scibert_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/ICLR2020_bid_score_scibert_gorc_uncased/feature/word_freq"
#user_dict_path = "./data/processed/ICLR2020_bid_score_scibert_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/ICLR2020_scibert_gorc_uncased/feature/word_freq"
#user_dict_path = "./data/processed/ICLR2020_scibert_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/ICLR2020_old_bid_score_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/ICLR2020_old_bid_score_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/ICLR2020_bid_score_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/ICLR2020_bid_score_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/ICLR2020_bid_high_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/ICLR2020_bid_high_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/ICLR2020_bid_low_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/ICLR2020_bid_low_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/ICLR2020_old_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/ICLR2020_old_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/ICLR2020_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/ICLR2020_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/ICLR2019_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/ICLR2019_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/ICLR2018_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/ICLR2018_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/ICLR2019_bid_score_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/ICLR2019_bid_score_gorc_uncased/user/dictionary_index"
#input_dict_path = "./data/processed/ICLR2018_bid_score_gorc_uncased/feature/dictionary_index"
#user_dict_path = "./data/processed/ICLR2018_bid_score_gorc_uncased/user/dictionary_index"

#submission_data_file = './data/processed/UAI2019_bid_score_scibert_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/UAI2019_new_bid_score_scibert_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#submission_data_file = './data/processed/UAI2019_scibert_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/UAI2019_scibert_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#submission_emb_file = './gen_log/UAI2019_scibert_submission_paper_emb_uni_avg_cbow_ACM_dim200.txt'
#submission_data_file = './data/processed/UAI2019_bid_score_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/UAI2019_new_bid_score_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#submission_data_file = './data/processed/UAI2019_bid_high_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/UAI2019_bid_high_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#submission_data_file = './data/processed/UAI2019_bid_low_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/UAI2019_bid_low_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#submission_emb_file = './gen_log/UAI2019_bid_low_submission_paper_emb_uni_avg_cbow_ACM_dim200.txt'
#submission_data_file = './data/processed/UAI2019_gorc_uncased/tensors_cold/test.pt'
#submission_data_file = './data/processed/UAI2019_old_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/UAI2019_new_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#submission_emb_file = './gen_log/UAI2019_submission_paper_emb_uni_avg_cbow_ACM_dim200.txt'
#submission_data_file = './data/processed/ICLR2020_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/ICLR2020_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#submission_emb_file = './gen_log/ICLR2020_submission_paper_emb_uni_avg_cbow_ACM_dim200.txt'

#submission_data_file = './data/processed/ICLR2020_bid_score_scibert_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/ICLR2020_new_bid_score_scibert_submission_paper_emb_freq_4_avg.txt'
#submission_data_file = './data/processed/ICLR2020_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/ICLR2020_new_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#submission_data_file = './data/processed/ICLR2020_scibert_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/ICLR2020_scibert_submission_paper_emb_freq_4_avg.txt'
#submission_emb_file = './gen_log/ICLR2020_scibert_submission_paper_emb_uni_avg.txt'
#submission_data_file = './data/processed/ICLR2020_bid_score_gorc_uncased/tensors_cold/test.pt'
#submission_data_file = './data/processed/ICLR2020_old_bid_score_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/ICLR2020_new_bid_score_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#submission_data_file = './data/processed/ICLR2020_bid_high_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/ICLR2020_bid_high_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#submission_data_file = './data/processed/ICLR2020_bid_low_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/ICLR2020_bid_low_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#submission_emb_file = './gen_log/ICLR2020_bid_low_submission_paper_emb_uni_avg_cbow_ACM_dim200.txt'
#submission_data_file = './data/processed/ICLR2019_bid_score_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/ICLR2019_bid_score_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#submission_data_file = './data/processed/ICLR2018_bid_score_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/ICLR2018_bid_score_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#submission_data_file = './data/processed/ICLR2019_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/ICLR2019_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#submission_data_file = './data/processed/ICLR2018_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/ICLR2018_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#submission_data_file = './data/processed/ICLR2019_scibert_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/ICLR2019_scibert_submission_paper_emb_freq_4_avg.txt'
#submission_data_file = './data/processed/ICLR2019_bid_score_scibert_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/ICLR2019_bid_score_scibert_submission_paper_emb_freq_4_avg.txt'
#submission_data_file = './data/processed/ICLR2018_scibert_gorc_uncased/tensors_cold/test.pt'
#submission_emb_file = './gen_log/ICLR2018_scibert_submission_paper_emb_freq_4_avg.txt'
submission_data_file = './data/processed/ICLR2018_bid_score_scibert_gorc_uncased/tensors_cold/test.pt'
submission_emb_file = './gen_log/ICLR2018_bid_score_scibert_submission_paper_emb_freq_4_avg.txt'

#reviewer_data_file = './data/processed/UAI2019_bid_score_scibert_gorc_uncased/tensors_cold/train.pt'
#reviewer_emb_file = './gen_log/UAI2019_bid_score_scibert_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#reviewer_data_file = './data/processed/UAI2019_scibert_gorc_uncased/tensors_cold/train.pt'
#reviewer_emb_file = './gen_log/UAI2019_scibert_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#reviewer_emb_file = './gen_log/UAI2019_scibert_reviewer_paper_emb_uni_avg_cbow_ACM_dim200.txt'
#reviewer_data_file = './data/processed/UAI2019_old_bid_score_gorc_uncased/tensors_cold/train.pt'
#reviewer_data_file = './data/processed/UAI2019_bid_score_gorc_uncased/tensors_cold/train.pt'
#reviewer_data_file = './data/processed/UAI2019_bid_high_gorc_uncased/tensors_cold/train.pt'
#reviewer_data_file = './data/processed/UAI2019_bid_low_gorc_uncased/tensors_cold/train.pt'
#reviewer_data_file = './data/processed/UAI2019_gorc_uncased/tensors_cold/train.pt'
#reviewer_data_file = './data/processed/UAI2019_old_gorc_uncased/tensors_cold/train.pt'
#reviewer_emb_file = './gen_log/UAI2019_new_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#reviewer_emb_file = './gen_log/UAI2019_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#reviewer_emb_file = './gen_log/UAI2019_reviewer_paper_emb_uni_avg_cbow_ACM_dim200.txt'
#reviewer_emb_file = './gen_log/UAI2019_bid_high_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#reviewer_emb_file = './gen_log/UAI2019_bid_low_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'

#reviewer_data_file = './data/processed/ICLR2020_gorc_uncased/tensors_cold/train.pt'
#reviewer_data_file = './data/processed/ICLR2020_bid_score_scibert_gorc_uncased/tensors_cold/train.pt'
#reviewer_emb_file = './gen_log/ICLR2020_bid_score_scibert_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#reviewer_data_file = './data/processed/ICLR2020_old_gorc_uncased/tensors_cold/train.pt'
#reviewer_emb_file = './gen_log/ICLR2020_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#reviewer_data_file = './data/processed/ICLR2020_scibert_gorc_uncased/tensors_cold/train.pt'
#reviewer_emb_file = './gen_log/ICLR2020_scibert_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#reviewer_emb_file = './gen_log/ICLR2020_scibert_reviewer_paper_emb_uni_avg_cbow_ACM_dim200.txt'
#reviewer_data_file = './data/processed/ICLR2020_bid_score_gorc_uncased/tensors_cold/train.pt'
#reviewer_data_file = './data/processed/ICLR2020_old_bid_score_gorc_uncased/tensors_cold/train.pt'
#reviewer_emb_file = './gen_log/ICLR2020_new_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#reviewer_data_file = './data/processed/ICLR2020_bid_high_gorc_uncased/tensors_cold/train.pt'
#reviewer_data_file = './data/processed/ICLR2020_bid_low_gorc_uncased/tensors_cold/train.pt'
#reviewer_emb_file = './gen_log/ICLR2020_reviewer_paper_emb_uni_avg_cbow_ACM_dim200.txt'
#reviewer_data_file = './data/processed/ICLR2019_gorc_uncased/tensors_cold/train.pt'
#reviewer_emb_file = './gen_log/ICLR2019_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#reviewer_data_file = './data/processed/ICLR2018_gorc_uncased/tensors_cold/train.pt'
#reviewer_emb_file = './gen_log/ICLR2018_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#reviewer_data_file = './data/processed/ICLR2019_scibert_gorc_uncased/tensors_cold/train.pt'
#reviewer_emb_file = './gen_log/ICLR2019_scibert_reviewer_paper_emb_freq_4_avg.txt'
reviewer_data_file = './data/processed/ICLR2018_scibert_gorc_uncased/tensors_cold/train.pt'
reviewer_emb_file = './gen_log/ICLR2018_scibert_reviewer_paper_emb_freq_4_avg.txt'

#mode = 'save_dist'
mode = 'run_eval'

if mode == 'save_dist':
    #out_dist_path = './gen_log/UAI2019_new_avg_cbow_freq_4_dist.np'
    #out_dist_path = './gen_log/UAI2019_new_max_cbow_freq_4_dist.np'
    #out_dist_path = './gen_log/ICLR2020_new_avg_cbow_freq_4_dist.np'
    #out_dist_path = './gen_log/ICLR2020_new_max_cbow_freq_4_dist.np'
    #out_dist_path = './gen_log/ICLR2020_new_avg_cbow_freq_4_dist_bid_score.np'
    #out_dist_path = './gen_log/ICLR2020_new_max_cbow_freq_4_dist_bid_score.np'
    #out_dist_path = './gen_log/UAI2019_new_avg_cbow_freq_4_dist_bid_score.np'
    #out_dist_path = './gen_log/UAI2019_new_max_cbow_freq_4_dist_bid_score.np'
    #out_dist_path = './gen_log/ICLR2018_scibert_avg_cbow_freq_4_dist.np'
    #out_dist_path = './gen_log/ICLR2018_scibert_max_cbow_freq_4_dist.np'
    #out_dist_path = './gen_log/ICLR2018_scibert_avg_cbow_freq_4_dist_bid_score.np'
    out_dist_path = './gen_log/ICLR2018_scibert_max_cbow_freq_4_dist_bid_score.np'
    #out_dist_path = './gen_log/ICLR2019_scibert_avg_cbow_freq_4_dist.np'
    #out_dist_path = './gen_log/ICLR2019_scibert_max_cbow_freq_4_dist.np'
    #out_dist_path = './gen_log/ICLR2019_scibert_avg_cbow_freq_4_dist_bid_score.np'
    #out_dist_path = './gen_log/ICLR2019_scibert_max_cbow_freq_4_dist_bid_score.np'
    #out_dist_path = './gen_log/ICLR2019_avg_cbow_freq_4_dist.np'
    #out_dist_path = './gen_log/ICLR2019_max_cbow_freq_4_dist.np'
    #out_dist_path = './gen_log/ICLR2019_avg_cbow_freq_4_dist_bid_score.np'
    #out_dist_path = './gen_log/ICLR2019_max_cbow_freq_4_dist_bid_score.np'
    #out_dist_path = './gen_log/ICLR2018_avg_cbow_freq_4_dist.np'
    #out_dist_path = './gen_log/ICLR2018_max_cbow_freq_4_dist.np'
    #out_dist_path = './gen_log/ICLR2018_avg_cbow_freq_4_dist_bid_score.np'
    #out_dist_path = './gen_log/ICLR2018_max_cbow_freq_4_dist_bid_score.np'
elif mode == 'run_eval':
    #out_f_path = './gen_log/UAI2019_bid_high_avg_cbow_freq_4_baseline'
    #out_f_path = './gen_log/UAI2019_bid_low_avg_cbow_freq_4_baseline'
    #out_f_path = './gen_log/UAI2019_bid_low_avg_cbow_uni_baseline'
    #out_f_path = './gen_log/UAI2019_bid_low_max_cbow_uni_baseline'
    #out_f_path = './gen_log/UAI2019_avg_cbow_freq_4_baseline'
    out_f_path = './gen_log/temp'
    #out_f_path = './gen_log/UAI2019_avg_cbow_uni_baseline'
    #out_f_path = './gen_log/UAI2019_max_cbow_uni_baseline'
    #out_f_path = './gen_log/ICLR2020_avg_cbow_freq_4_baseline'
    #out_f_path = './gen_log/ICLR2020_avg_cbow_uni_baseline'
    #out_f_path = './gen_log/ICLR2020_max_cbow_uni_baseline'
    #out_f_path = './gen_log/ICLR2020_bid_low_avg_cbow_freq_4_baseline'
    #out_f_path = './gen_log/ICLR2020_bid_low_max_cbow_freq_4_baseline'
    #out_f_path = './gen_log/ICLR2020_bid_low_avg_cbow_uni_baseline'
    #out_f_path = './gen_log/ICLR2020_bid_low_max_cbow_uni_baseline'

#dist_option = 'emb_avg'
dist_option = 'sim_avg'
#dist_option = 'sim_max'

def load_idx_d2_word_freq(f_in):
    idx_d2_word_freq = {}
    for i, line in enumerate(f_in):
        fields = line.rstrip().split('\t')
        if len(fields) == 3:
            idx_d2_word_freq[int(fields[2])] = [fields[0],int(fields[1])]

    return idx_d2_word_freq

device = torch.device('cpu')

eval_bsz = 50

def load_emb_file(f_in, device):
    emb_list = []
    for line in f_in:
        word_val = line.rstrip().split(' ')
        val = [float(x) for x in  word_val]
        emb_list.append(val)
    return torch.tensor(emb_list, device = device)

with torch.no_grad():

    with open(reviewer_data_file, 'rb') as f_in:
        dataloader_train_info = utils.create_data_loader(f_in, eval_bsz, device, want_to_shuffle = False, deduplication = True)
        #feature, feature_type, user, tag, repeat_num, user_len, tag_len = torch.load(f_in, map_location='cpu')
        #dataset, all_user_tag = utils.create_uniq_paper_data(feature, feature_type, user, tag, device, user_subsample_idx = [], tag_subsample_idx= [])
    all_user_tag = dataloader_train_info[1]
    #all_user_train, all_tag_train = zip(*all_user_tag)
    fields = list(zip(*all_user_tag))
    all_user_train = fields[0]
    #all_tag_train = fields[1]

    with open(submission_data_file,'rb') as f_in:
        dataloader_test_info = utils.create_data_loader(f_in, eval_bsz, device, want_to_shuffle = False, deduplication = True)
    all_user_tag = dataloader_test_info[1]
    #all_user_test, all_tag_test = zip(*all_user_tag)
    fields = list(zip(*all_user_tag))
    all_user_test = fields[0]
    #all_tag_test = fields[1]

    with open(reviewer_emb_file) as f_in:
        paper_emb_train = load_emb_file(f_in, device)

    with open(submission_emb_file) as f_in:
        paper_emb_test = load_emb_file(f_in, device)

    paper_num_train = paper_emb_train.size(0)
    assert paper_num_train == len(all_user_train)
    paper_num_test = paper_emb_test.size(0)
    assert paper_num_test == len(all_user_test)

    with open(input_dict_path) as f_in:
        #idx2word_freq = utils.load_idx2word_freq(f_in)
        idx2word_freq = load_idx_d2_word_freq(f_in)

    with open(user_dict_path) as f_in:
        user_idx2word_freq = utils.load_idx2word_freq(f_in)
    
    user_num = len(user_idx2word_freq)
    user_l2_paper_id = [[] for i in range(user_num)]
    for i in range(paper_num_train):
        for user_id in all_user_train[i]:
            user_l2_paper_id[user_id].append(i)
    #with open(tag_dict_path) as f_in:
    #    tag_idx2word_freq = utils.load_idx2word_freq(f_in)

    #TODO: batchify if slow
    if dist_option[:3] == 'sim':
        p2p_dist = torch.empty( (paper_num_test, paper_num_train) ,device=device)
        for i in range(paper_num_test):
            p2p_dist[i,:] = 1 - torch.sum(paper_emb_test[i,:].unsqueeze(dim=0) * paper_emb_train, dim=1)
            #assume that the embedding has been normalized
        paper_user_dist = np.empty( (paper_num_test, user_num), dtype=np.float16 )
        for j in range(user_num):
            train_paper_id_list = user_l2_paper_id[j]
            if len(train_paper_id_list) == 0:
                paper_user_dist[:,j] = 1
                continue
            train_paper_dist_j = p2p_dist[:, train_paper_id_list]
        #for i in range(paper_num_test):
            if dist_option == 'sim_avg':
                paper_user_dist[:,j] = train_paper_dist_j.mean(dim = 1)
            elif dist_option == 'sim_max':
                #print(train_paper_dist_j.size())
                #print(train_paper_dist_j.min(dim = 1).size())
                #print(torch.min(train_paper_dist_j,dim = 1).size())
                #paper_user_dist[:,j] = train_paper_dist_j.min(dim = 1)
                #torch.min(train_paper_dist_j, dim = 1)
                paper_user_dist[:,j] = torch.min(train_paper_dist_j, dim = 1)[0]
    
    if mode == 'save_dist':
        np.savetxt(out_dist_path, paper_user_dist)
        #with open(out_dist_path, 'w') as outf:
        #    pickle.dump( paper_user_dist, outf)
    elif mode == 'run_eval':
        test_user = True
        test_tag = False
        most_popular_baseline = False
        div_eval = 'openreview'
        tag_idx2word_freq = []
        paper_tag_dist = []
        with open(out_f_path, 'w') as outf:
            utils_testing.recommend_test_from_all_dist(dataloader_test_info, paper_user_dist, paper_tag_dist, idx2word_freq, user_idx2word_freq, tag_idx2word_freq, test_user, test_tag, outf, device, most_popular_baseline, div_eval)
