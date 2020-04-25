import sys
sys.path.insert(0, sys.path[0]+'/../../')
import utils
import torch

def compute_freq_prob_idx2word(idx2word_freq):
    all_word, all_freq= list( zip(*idx2word_freq) )
    freq_sum = float(sum(all_freq))
    for i, (w, freq) in enumerate(idx2word_freq):
        idx2word_freq[i].append(freq/freq_sum)

use_freq_w = True
#use_freq_w = False
emb_file = './resources/cbow_ACM_dim200_gorc_uncased_min_5.txt'

input_dict = "./data/processed/UAI2019_bid_score_gorc_uncased/feature/dictionary_index"
#input_dict = "./data/processed/UAI2019_bid_high_gorc_uncased/feature/dictionary_index"
#input_dict = "./data/processed/UAI2019_bid_low_gorc_uncased/feature/dictionary_index"
#input_dict = "./data/processed/UAI2019_gorc_uncased/feature/dictionary_index"
#input_dict = "./data/processed/ICLR2020_gorc_uncased/feature/dictionary_index"
#input_dict = "./data/processed/ICLR2020_bid_score_gorc_uncased/feature/dictionary_index"
#input_dict = "./data/processed/ICLR2020_bid_high_gorc_uncased/feature/dictionary_index"
#input_dict = "./data/processed/ICLR2020_bid_low_gorc_uncased/feature/dictionary_index"

data_file = './data/processed/UAI2019_bid_score_gorc_uncased/tensors_cold/test.pt'
output_file = './gen_log/UAI2019_bid_score_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/UAI2019_bid_high_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/UAI2019_bid_high_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/UAI2019_bid_low_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/UAI2019_bid_low_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#output_file = './gen_log/UAI2019_bid_low_submission_paper_emb_uni_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/UAI2019_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/UAI2019_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#output_file = './gen_log/UAI2019_submission_paper_emb_uni_avg_cbow_ACM_dim200.txt'

#data_file = './data/processed/ICLR2020_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/ICLR2020_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#output_file = './gen_log/ICLR2020_submission_paper_emb_uni_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/ICLR2020_bid_score_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/ICLR2020_bid_score_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#output_file = './gen_log/ICLR2020_bid_score_submission_paper_emb_uni_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/ICLR2020_bid_low_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/ICLR2020_bid_low_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#output_file = './gen_log/ICLR2020_bid_low_submission_paper_emb_uni_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/ICLR2020_bid_high_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/ICLR2020_bid_high_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#output_file = './gen_log/ICLR2020_bid_high_submission_paper_emb_uni_avg_cbow_ACM_dim200.txt'

#data_file = './data/processed/UAI2019_bid_high_gorc_uncased/tensors_cold/train.pt'
#output_file = './gen_log/UAI2019_bid_high_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/UAI2019_bid_low_gorc_uncased/tensors_cold/train.pt'
#output_file = './gen_log/UAI2019_bid_low_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/UAI2019_gorc_uncased/tensors_cold/train.pt'
#output_file = './gen_log/UAI2019_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#output_file = './gen_log/UAI2019_reviewer_paper_emb_uni_avg_cbow_ACM_dim200.txt'

#data_file = './data/processed/ICLR2020_gorc_uncased/tensors_cold/train.pt'
#output_file = './gen_log/ICLR2020_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#output_file = './gen_log/ICLR2020_reviewer_paper_emb_uni_avg_cbow_ACM_dim200.txt'

device = torch.device('cpu')

with open(input_dict) as f_in:
    idx2word_freq = utils.load_idx2word_freq(f_in)

compute_freq_prob_idx2word(idx2word_freq)

with torch.no_grad():
    source_emb, emb_size, oov_list = utils.load_emb_file_to_tensor(emb_file, device, idx2word_freq)
    source_emb = source_emb / (0.000000000001 + source_emb.norm(dim = 1, keepdim=True))
    source_emb[0,:] = 0

    with open(data_file, 'rb') as f_in:
        #feature_raw, feature_type, user, tag, repeat_num, user_len, tag_len = torch.load(f_in, map_location='cpu')
        fields = torch.load(f_in, map_location='cpu')
        if len(fields) == 7:
            feature_raw, feature_type, user, tag, repeat_num, user_len, tag_len = fields #torch.load(f_in, map_location='cpu')
            bid_score = torch.zeros(0)
        else:
            feature_raw, feature_type, user, tag, repeat_num, user_len, tag_len, bid_score = fields

    dataset, all_user_tag = utils.create_uniq_paper_data(feature_raw, feature_type, user, tag, device, user_subsample_idx = [], tag_subsample_idx= [], bid_score=bid_score)
    feature = dataset.feature
    num_paper = feature.size(0)
    all_emb_sum = torch.empty( (num_paper, emb_size), device = device)
    for i in range(num_paper):
        feature_i = feature[i,:].to(dtype = torch.long, device = device)
        word_emb_paper_i = source_emb[feature_i,:]
        if use_freq_w:
            feature_i_list = feature_i.tolist()
            alpha = 0.0001
            prob_i = torch.tensor([idx2word_freq[x][2] for x in feature_i_list], device = device)
            weight = alpha / (alpha + prob_i)
            weight = weight.unsqueeze(dim=-1)
            #print(weight.size())
        else:
            weight = 1
        #print(word_emb_paper_i.size())
        word_emb_sum = (weight * word_emb_paper_i).sum(dim=0)
        all_emb_sum[i,:] = word_emb_sum
    all_emb_norm = all_emb_sum / (0.000000000001 + all_emb_sum.norm(dim = 1, keepdim=True))
    all_emb_norm = all_emb_norm.tolist()
    #print(len(all_emb_norm))
    #print(len(all_emb_norm[0]))

    with open(output_file, 'w') as f_out:
        for i in range(num_paper):
            f_out.write(' '.join(map(str,all_emb_norm[i])) + '\n')
