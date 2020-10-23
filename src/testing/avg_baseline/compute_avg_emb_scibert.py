import sys
sys.path.insert(0, sys.path[0]+'/../../')
import utils
import torch

from scibert.modeling_bert import BertModel
from scibert.configuration_bert import BertConfig

model_name = 'scibert-scivocab-uncased'
encoder = BertModel.from_pretrained(model_name)
bert_config = BertConfig.from_pretrained(model_name)
bert_vocab_size = bert_config.vocab_size
print(bert_vocab_size)
bert_emb_size = bert_config.hidden_size
print(bert_emb_size)

def print_parameters(model, excluding_prefix = None):
    parameter_sum = 0
    for name, param in model.named_parameters():
        if excluding_prefix is not None and excluding_prefix == name[:len(excluding_prefix)]:
            print("Skipping ", name, param.numel())
            continue
        if param.requires_grad:
            print(name, param.numel())
            parameter_sum += param.numel()
    return parameter_sum

#print("SciBERT after filtering ", print_parameters(encoder))
#exit()

CLS_idx = 102
SEP_idx = 103

def compute_freq_prob_idx_d2_word(idx_d2_word_freq):
    all_word, all_freq= list( zip(*idx_d2_word_freq.values()) )
    freq_sum = float(sum(all_freq))
    for idx, (w, freq) in idx_d2_word_freq.items():
        idx_d2_word_freq[idx].append(freq/freq_sum)

bsz = 30

use_freq_w = True
#use_freq_w = False
#emb_file = './resources/cbow_ACM_dim200_gorc_uncased_min_5.txt'

#input_dict = "./data/processed/ICLR2018_scibert_gorc_uncased/feature/word_freq"
#input_dict = "./data/processed/ICLR2018_bid_score_scibert_gorc_uncased/feature/word_freq"
#input_dict = "./data/processed/ICLR2019_scibert_gorc_uncased/feature/word_freq"
#input_dict = "./data/processed/ICLR2019_bid_score_scibert_gorc_uncased/feature/word_freq"
#input_dict = "./data/processed/ICLR2020_scibert_gorc_uncased/feature/word_freq"
#input_dict = "./data/processed/ICLR2020_bid_score_scibert_gorc_uncased/feature/word_freq"
#input_dict = "./data/processed/UAI2019_scibert_gorc_uncased/feature/word_freq"
#input_dict = "./data/processed/UAI2019_bid_score_scibert_gorc_uncased/feature/word_freq"
#input_dict = "./data/processed/NeurIPS2019_scibert_gorc_uncased/feature/word_freq"
input_dict = "./data/processed/NeurIPS2019_bid_score_scibert_gorc_uncased/feature/word_freq"

#data_file = './data/processed/NeurIPS2019_bid_score_scibert_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/NeurIPS2019_bid_score_scibert_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/UAI2019_bid_score_scibert_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/UAI2019_bid_score_scibert_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/UAI2019_scibert_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/UAI2019_scibert_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#output_file = './gen_log/UAI2019_scibert_submission_paper_emb_uni_avg_cbow_ACM_dim200.txt'

#data_file = './data/processed/ICLR2018_bid_score_scibert_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/ICLR2018_bid_score_scibert_submission_paper_emb_freq_4_avg.txt'
#data_file = './data/processed/ICLR2018_scibert_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/ICLR2018_scibert_submission_paper_emb_freq_4_avg.txt'
#data_file = './data/processed/ICLR2019_bid_score_scibert_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/ICLR2019_bid_score_scibert_submission_paper_emb_freq_4_avg.txt'
#data_file = './data/processed/ICLR2019_scibert_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/ICLR2019_scibert_submission_paper_emb_freq_4_avg.txt'
#data_file = './data/processed/ICLR2020_bid_score_scibert_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/ICLR2020_bid_score_scibert_submission_paper_emb_freq_4_avg.txt'
#data_file = './data/processed/ICLR2020_scibert_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/ICLR2020_scibert_submission_paper_emb_freq_4_avg.txt'
#output_file = './gen_log/ICLR2020_scibert_submission_paper_emb_uni_avg.txt'
#data_file = './data/processed/ICLR2020_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/ICLR2020_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#output_file = './gen_log/ICLR2020_submission_paper_emb_uni_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/ICLR2020_bid_score_gorc_uncased/tensors_cold/test.pt'
#output_file = './gen_log/ICLR2020_bid_score_submission_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#output_file = './gen_log/ICLR2020_bid_score_submission_paper_emb_uni_avg_cbow_ACM_dim200.txt'

data_file = './data/processed/NeurIPS2019_bid_score_scibert_gorc_uncased/tensors_cold/train.pt'
output_file = './gen_log/NeurIPS2019_bid_score_scibert_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/UAI2019_bid_score_scibert_gorc_uncased/tensors_cold/train.pt'
#output_file = './gen_log/UAI2019_bid_score_scibert_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/UAI2019_scibert_gorc_uncased/tensors_cold/train.pt'
#output_file = './gen_log/UAI2019_scibert_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/UAI2019_scibert_gorc_uncased/tensors_cold/train.pt'
#output_file = './gen_log/UAI2019_scibert_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#output_file = './gen_log/UAI2019_scibert_reviewer_paper_emb_uni_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/UAI2019_gorc_uncased/tensors_cold/train.pt'
#output_file = './gen_log/UAI2019_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#output_file = './gen_log/UAI2019_reviewer_paper_emb_uni_avg_cbow_ACM_dim200.txt'

#data_file = './data/processed/ICLR2018_scibert_gorc_uncased/tensors_cold/train.pt'
#output_file = './gen_log/ICLR2018_scibert_reviewer_paper_emb_freq_4_avg.txt'
#data_file = './data/processed/ICLR2019_bid_score_scibert_gorc_uncased/tensors_cold/train.pt'
#output_file = './gen_log/ICLR2019_bid_score_scibert_reviewer_paper_emb_freq_4_avg.txt'
#data_file = './data/processed/ICLR2019_scibert_gorc_uncased/tensors_cold/train.pt'
#output_file = './gen_log/ICLR2019_scibert_reviewer_paper_emb_freq_4_avg.txt'
#data_file = './data/processed/ICLR2020_bid_score_scibert_gorc_uncased/tensors_cold/train.pt'
#output_file = './gen_log/ICLR2020_bid_score_scibert_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/ICLR2020_scibert_gorc_uncased/tensors_cold/train.pt'
#output_file = './gen_log/ICLR2020_scibert_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#output_file = './gen_log/ICLR2020_scibert_reviewer_paper_emb_uni_avg_cbow_ACM_dim200.txt'
#data_file = './data/processed/ICLR2020_gorc_uncased/tensors_cold/train.pt'
#output_file = './gen_log/ICLR2020_reviewer_paper_emb_freq_4_avg_cbow_ACM_dim200.txt'
#output_file = './gen_log/ICLR2020_reviewer_paper_emb_uni_avg_cbow_ACM_dim200.txt'

#device = torch.device('cpu')
device = torch.device('cuda')
encoder = encoder.cuda()

def load_idx_d2_word_freq(f_in):
    idx_d2_word_freq = {}
    for i, line in enumerate(f_in):
        fields = line.rstrip().split('\t')
        if len(fields) == 3:
            idx_d2_word_freq[int(fields[2])] = [fields[0],int(fields[1])]

    return idx_d2_word_freq

with open(input_dict) as f_in:
    idx_d2_word_freq = load_idx_d2_word_freq(f_in)

compute_freq_prob_idx_d2_word(idx_d2_word_freq)

alpha = 0.0001
feature_to_weights = torch.zeros( (bert_vocab_size), device = device)
for idx in idx_d2_word_freq:
    feature_to_weights[idx] = alpha / (alpha + idx_d2_word_freq[idx][2])


with torch.no_grad():
    #source_emb, emb_size, oov_list = utils.load_emb_file_to_tensor(emb_file, device, idx2word_freq)
    #source_emb = source_emb / (0.000000000001 + source_emb.norm(dim = 1, keepdim=True))
    #source_emb[0,:] = 0

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
    all_emb_sum = torch.empty( (num_paper, bert_emb_size), device = device)
    #testing_dataloader = torch.utils.data.DataLoader(dataset, batch_size = bsz, shuffle = False, drop_last=False)
    #for i_batch, sample_batched in enumerate(testing_dataloader):
    #    feature, feature_type, paper_id_tensor = sample_batched
    #    last_hidden_state, pooler_output = ecoder(feature)
    #    if use_freq_w:
    #    else:
    for i in range(num_paper):
        if i % 10 == 0:
            sys.stdout.write( "{:2.2f} ".format(i/float(num_paper)) )
            sys.stdout.flush()
        start_arr = [ j for j in range(feature.size(1)) if feature[i,j] == CLS_idx ]
        end_arr = [ j for j in range(feature.size(1)) if feature[i,j] == SEP_idx ]
        if len(end_arr) == len(start_arr) - 1:
            end_arr.append(feature.size(1))
        
        word_emb_sum = torch.zeros((bert_emb_size), device = device)
        for start, end in zip(start_arr, end_arr):
            if start == end:
                continue
            feature_ij = feature[i,start:end+1].to(dtype = torch.long, device = device)
            last_hidden_state, pooler_output = encoder(feature_ij.unsqueeze(dim=0))
            w_emb = last_hidden_state.squeeze(dim=0)
            
            #TODO
            if use_freq_w:
                weight = feature_to_weights[feature_ij]
                weight = weight.unsqueeze(dim=-1)
                #print(weight.size())
            else:
                weight = 1
            #print(word_emb_papder_i.size())
            word_emb_sum += (weight * w_emb).sum(dim=0)
        all_emb_sum[i,:] = word_emb_sum
    all_emb_norm = all_emb_sum / (0.000000000001 + all_emb_sum.norm(dim = 1, keepdim=True))
    all_emb_norm = all_emb_norm.tolist()
    #print(len(all_emb_norm))
    #print(len(all_emb_norm[0]))

    with open(output_file, 'w') as f_out:
        for i in range(num_paper):
            f_out.write(' '.join(map(str,all_emb_norm[i])) + '\n')
