import torch
import nsd_loss
import numpy as np
import gc
import sys
import torch.utils.data
import json
import torch.nn.functional as F

def add_model_arguments(parser):
    ###encoder
    parser.add_argument('--en_model', type=str, default='LSTM',
                        help='type of encoder model (LSTM)')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=600,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--encode_trans_layers', type=int, default=2,
                        help='How many layers we have in transformer. Do not have effect if de_model is LSTM')
    parser.add_argument('--trans_nhid', type=int, default=-1,
                        help='number of hidden units per layer in transformer')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to the output layer (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.4,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')

    ###decoder
    parser.add_argument('--de_model', type=str, default='LSTM',
                        help='type of decoder model (LSTM)')
    parser.add_argument('--trans_layers', type=int, default=2,
                        help='How many layers we have in transformer. Do not have effect if de_model is LSTM')
    parser.add_argument('--de_en_connection', type=bool, default=True,
                        help='If True, using Transformer decoder in our decoder. Otherwise, using Transformer encoder')
    parser.add_argument('--nhidlast2', type=int, default=-1,
                        help='hidden embedding size of the second LSTM')
    parser.add_argument('--n_basis', type=int, default=10,
                        help='number of basis we want to predict')
    parser.add_argument('--linear_mapping_dim', type=int, default=0,
                        help='map the input embedding by linear transformation')
    #parser.add_argument('--postional_option', type=str, default='linear',
    #                help='options of encode positional embedding into models (linear, cat, add)')
    parser.add_argument('--dropoutp', type=float, default=0.5,
                        help='dropout of positional embedding or input embedding after linear transformation (when linear_mapping_dim != 0)')

def predict_batch(feature, parallel_encoder, parallel_decoder, word_norm_emb, n_basis, top_k):
    #output_emb, hidden, output_emb_last = parallel_encoder(feature.t())
    output_emb_last = parallel_encoder(feature)
    basis_pred, coeff_pred = nsd_loss.predict_basis(parallel_decoder, n_basis, output_emb_last, predict_coeff_sum = True )

    coeff_sum = coeff_pred.cpu().numpy()
    coeff_sum_diff = coeff_sum[:,:,0] - coeff_sum[:,:,1]
    coeff_order = np.argsort(coeff_sum_diff, axis = 1)
    coeff_order = np.flip( coeff_order, axis = 1 )

    basis_pred = basis_pred.permute(0,2,1)
    #basis_pred should have dimension (n_batch, emb_size, n_basis)
    basis_norm_pred = basis_pred / (0.000000000001 + basis_pred.norm(dim = 1, keepdim=True) )
    #word_norm_emb should have dimension (ntokens, emb_size)
    sim_pairwise = torch.matmul(word_norm_emb.unsqueeze(dim = 0), basis_norm_pred)
    #print(sim_pairwise.size())
    #sim_pairwise should have dimension (n_batch, ntokens, emb_size)
    top_value, top_index= torch.topk(sim_pairwise, top_k, dim = 1, sorted=True)

    return basis_norm_pred, coeff_order, coeff_sum, top_value, top_index, output_emb_last

def convert_feature_to_text(feature, idx2word_freq):
    feature_list = feature.tolist()
    feature_text = []
    for i in range(feature.size(0)):
        current_sent = []
        for w_ind in feature_list[i]:
            if w_ind != 0:
                w = idx2word_freq[w_ind][0]
                current_sent.append(w)
        feature_text.append(current_sent)
    return feature_text

def print_basis_text(feature, idx2word_freq, coeff_order, coeff_sum, top_value, top_index, i_batch, outf):
    n_basis = coeff_order.shape[1]
    top_k = top_index.size(1)
    feature_text = convert_feature_to_text(feature, idx2word_freq)
    for i_sent in range(len(feature_text)):
        outf.write('{} batch, {}th sent: '.format(i_batch, i_sent)+' '.join(feature_text[i_sent])+'\n')

        for j in range(n_basis):
            org_ind = coeff_order[i_sent, j]
            outf.write(str(j)+', org '+str(org_ind)+', '+str( coeff_sum[i_sent,org_ind,0] )+' - '+str( coeff_sum[i_sent,org_ind,1] )+': ')

            for k in range(top_k):
                word_nn = idx2word_freq[top_index[i_sent,k,org_ind].item()][0]
                outf.write( word_nn+' {:5.3f}'.format(top_value[i_sent,k,org_ind].item())+' ' )
            outf.write('\n')
        outf.write('\n')

def dump_prediction_to_json(feature, basis_norm_pred, idx2word_freq, coeff_order, coeff_sum, top_value, top_index, basis_json, org_sent_list, encoded_emb):
    n_basis = coeff_order.shape[1]
    top_k = top_index.size(1)
    feature_text = convert_feature_to_text(feature, idx2word_freq)
    for i_sent in range(len(feature_text)):
        current_idx = len(basis_json)
        proc_sent = feature_text[i_sent]
        org_sent = org_sent_list[current_idx]
        topic_list = []
        for j in range(n_basis):
            org_ind = coeff_order[i_sent, j]
            topic_info = {}
            topic_info['idx'] = str(j)
            topic_info['org_idx'] = str(org_ind)
            topic_info['coeff_pos'] = str( coeff_sum[i_sent,org_ind,0] )
            topic_info['coeff_neg'] = str( coeff_sum[i_sent,org_ind,1] )
            topic_info['v'] = [str(x) for x in basis_norm_pred[i_sent,:,j].tolist() ]
            topic_vis = []
            for k in range(top_k):
                word_nn = idx2word_freq[top_index[i_sent,k,org_ind].item()][0]
                topic_vis.append([word_nn, ' {:5.3f}'.format(top_value[i_sent,k,org_ind].item())])
            topic_info['topk'] = topic_vis
            topic_list.append(topic_info)
        output_dict = {}
        output_dict['idx'] = str(current_idx)
        output_dict['org_sent'] = org_sent
        output_dict['sent_emb'] = [str(x) for x in encoded_emb[i_sent,:].tolist()]
        output_dict['proc_sent'] = ' '.join(proc_sent)
        output_dict['topics'] = topic_list
        basis_json.append(output_dict)
        #basis_json.append([current_idx, org_sent, ' '.join(proc_sent)])

def load_prediction_from_json(f_in):
    sent_d2_topics = {}
    input_json= json.load(f_in)
    for input_dict in input_json:
        org_sent = input_dict['org_sent']
        topic_list = []
        topic_weight_list = []
        sent_emb = [float(x) for x in input_dict['sent_emb']]
        for topic in input_dict['topics']:
            weight = max( 0, float(topic['coeff_pos']) - float(topic['coeff_neg']))
            vector = [float(x) for x in topic['v']]
            topic_weight_list.append(weight)
            topic_list.append(vector)
        sent_d2_topics[org_sent] = [topic_list, topic_weight_list, sent_emb]
    return sent_d2_topics

def output_sent_basis(dataloader, org_sent_list, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, n_basis, outf_vis):
    basis_json = []
    top_k = 5
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            if i_batch % 100 == 0:
                sys.stdout.write('b'+str(i_batch)+' ')
                sys.stdout.flush()
            feature, target = sample_batched
            basis_norm_pred, coeff_order, coeff_sum, top_value, top_index, encoded_emb = predict_batch(feature, parallel_encoder, parallel_decoder, word_norm_emb, n_basis, top_k)
            print_basis_text(feature, idx2word_freq, coeff_order, coeff_sum, top_value, top_index, i_batch, outf_vis)
            dump_prediction_to_json(feature, basis_norm_pred, idx2word_freq, coeff_order, coeff_sum, top_value, top_index, basis_json, org_sent_list, encoded_emb)
    return basis_json

def visualize_topics_val(dataloader, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, n_basis, max_batch_num):
    #topics_num = 0
    top_k = 5
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            feature, target = sample_batched

            basis_norm_pred, coeff_order, coeff_sum, top_value, top_index, encoded_emb = predict_batch(feature, parallel_encoder, parallel_decoder, word_norm_emb, n_basis, top_k)
            print_basis_text(feature, idx2word_freq, coeff_order, coeff_sum, top_value, top_index, i_batch, outf)

            if i_batch >= max_batch_num:
                break

class Set2SetDataset(torch.utils.data.Dataset):
    def __init__(self, source, source_w, source_sent_emb, target, target_w, target_sent_emb):
        self.source = source
        self.source_w = source_w
        self.source_sent_emb = source_sent_emb
        self.target = target
        self.target_w = target_w
        self.target_sent_emb = target_sent_emb

    def __len__(self):
        return self.source.size(0)

    def __getitem__(self, idx):
        source = self.source[idx, :, :]
        source_w = self.source_w[idx, :]
        source_sent_emb = self.source_sent_emb[idx, :]
        target = self.target[idx, :, :]
        target_w = self.target_w[idx, :]
        target_sent_emb = self.target_sent_emb[idx, :]
        #debug target[-1] = idx
        return [source, source_w, source_sent_emb, target, target_w, target_sent_emb]

def build_loader_from_pairs(testing_list, sent_d2_topics, bsz, device):
    def store_topics(sent, sent_d2_topics, topic_v_tensor, topic_w_tensor, sent_emb_tensor, i_pairs, device):
        topic_v, topic_w, sent_emb = sent_d2_topics[sent]
        topic_v_tensor[i_pairs, :, :] = torch.tensor(topic_v, device = device)
        topic_w_tensor[i_pairs, :] = torch.tensor(topic_w, device = device)
        sent_emb_tensor[i_pairs, :] = torch.tensor(sent_emb, device = device)
    
    corpus_size = len(testing_list)
    first_sent_info = list(sent_d2_topics.values())[0]
    first_topic = first_sent_info[0]
    n_basis = len(first_topic)
    emb_size = len(first_topic[0])
    first_sent_emb = first_sent_info[2]
    encoder_emsize= len(first_sent_emb)

    topic_v_tensor_1 = torch.empty(corpus_size, n_basis, emb_size, device = device)
    topic_w_tensor_1 = torch.empty(corpus_size, n_basis, device = device)
    sent_emb_tensor_1 = torch.empty(corpus_size, encoder_emsize, device = device)
    topic_v_tensor_2 = torch.empty(corpus_size, n_basis, emb_size, device = device)
    topic_w_tensor_2 = torch.empty(corpus_size, n_basis, device = device)
    sent_emb_tensor_2 = torch.empty(corpus_size, encoder_emsize, device = device)
    for i_pairs, fields in enumerate(testing_list):
        sent_1 = fields[0]
        sent_2 = fields[1]
        store_topics(sent_1, sent_d2_topics, topic_v_tensor_1, topic_w_tensor_1, sent_emb_tensor_1, i_pairs, device)
        store_topics(sent_2, sent_d2_topics, topic_v_tensor_2, topic_w_tensor_2, sent_emb_tensor_2, i_pairs, device)

    dataset = Set2SetDataset(topic_v_tensor_1, topic_w_tensor_1, sent_emb_tensor_1, topic_v_tensor_2, topic_w_tensor_2, sent_emb_tensor_2)
    use_cuda = False
    if device == 'cude':
        use_cuda = True
    testing_pair_loader = torch.utils.data.DataLoader(dataset, batch_size = bsz, shuffle = False, pin_memory=use_cuda, drop_last=False)
    return testing_pair_loader



def predict_sim_scores(testing_pair_loader, L1_losss_B, device):
    def safe_normalization(weight):
        weight_sum = torch.sum(weight)
        assert weight_sum>0
        return weight / weight_sum

    def weighted_average(cosine_sim, weight):
        #assume that weight are normalized
        return (cosine_sim * weight).mean(dim = 1)

    def max_cosine_sim(target, source, target_w, device):
        cosine_sim_s_to_t, nn_source_idx = nsd_loss.estimate_coeff_mat_batch_max_cos(target, source, device)
        #cosine_sim should have dimension (n_batch,n_set)
        sim_avg_st = cosine_sim_s_to_t.mean(dim = 1)
        sim_w_avg_st = weighted_average(cosine_sim_s_to_t, target_w)
        return sim_avg_st, sim_w_avg_st
    
    def lc_pred_dist(target, source, target_w, L1_losss_B, device):
        coeff_mat_s_to_t = nsd_loss.estimate_coeff_mat_batch(target, source, L1_losss_B, device, max_iter = 1000)
        pred_embeddings = torch.bmm(coeff_mat_s_to_t, source)
        dist_st = torch.pow( torch.norm( pred_embeddings - target, dim = 2 ), 2)
        dist_avg_st = dist_st.mean(dim = 1)
        dist_w_avg_st = weighted_average(dist_st, target_w)
        return dist_avg_st, dist_w_avg_st

    pred_scores = []
    for source, source_w, source_sent_emb, target, target_w, target_sent_emb in testing_pair_loader:
        #normalize w
        source_w = safe_normalization(source_w)
        target_w = safe_normalization(target_w)
        
        #Kmeans loss
        sim_avg_st, sim_w_avg_st = max_cosine_sim(target, source, target_w, device)
        sim_avg_ts, sim_w_avg_ts = max_cosine_sim(source, target, source_w, device)
        sim_avg = (sim_avg_st + sim_avg_ts)/2
        sim_w_avg = (sim_w_avg_st + sim_w_avg_ts)/2

        dist_avg_st, dist_w_avg_st = lc_pred_dist(target, source, target_w, L1_losss_B, device)
        dist_avg_ts, dist_w_avg_ts = lc_pred_dist(source, target, source_w, L1_losss_B, device)
        dist_avg = (dist_avg_st + dist_avg_ts)/2
        dist_w_avg = (dist_w_avg_st + dist_w_avg_ts)/2

        cosine_sim = F.cosine_similarity(source_sent_emb, target_sent_emb, dim = 1)

        for i in range(sim_avg.size(0)):
        #coeff_mat should have dimension (n_batch,n_set,n_basis)
            pred_scores.append([sim_avg[i].item(), sim_w_avg[i].item(), dist_avg[i].item(), dist_w_avg[i].item(), cosine_sim[i].item() ])
    return pred_scores
