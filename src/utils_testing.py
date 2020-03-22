import torch
import nsd_loss
import numpy as np
from scipy.spatial import distance
import scipy.stats as ss
import gc
import sys
import torch.utils.data
import json
import torch.nn.functional as F
from utils import str2bool
sys.path.insert(0, sys.path[0]+'/testing/sim')
import SIF_pc_removal as SIF
import math
import Sinkhorn
import ot

def add_model_arguments(parser):
    ###encoder
    #parser.add_argument('--en_model', type=str, default='LSTM',
    parser.add_argument('--en_model', type=str, default='TRANS',
                        help='type of encoder model (LSTM)')
    #parser.add_argument('--emsize', type=int, default=300,
    #                    help='size of word embeddings')
    parser.add_argument('--source_emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--target_emsize', type=int, default=0,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--encode_trans_layers', type=int, default=3,
                        help='How many layers we have in transformer. Do not have effect if de_model is LSTM')
    parser.add_argument('--trans_nhid', type=int, default=-1,
                        help='number of hidden units per layer in transformer')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout applied to the output layer (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0,
                        help='dropout to remove words from embedding layer (0 = no dropout)')

    ###decoder
    #parser.add_argument('--de_model', type=str, default='LSTM',
    parser.add_argument('--de_model', type=str, default='TRANS',
                        help='type of decoder model (LSTM, LSTM+TRANS, TRANS+LSTM, TRANS)')
    #parser.add_argument('--de_coeff_model', type=str, default='LSTM',
    parser.add_argument('--de_coeff_model', type=str, default='TRANS',
                    help='type of decoder model to predict coefficients (LSTM, TRANS)')
    parser.add_argument('--trans_layers', type=int, default=3,
                        help='How many layers we have in transformer. Do not have effect if de_model is LSTM')
    parser.add_argument('--de_en_connection', type=str3bool, nargs='?', default=True,
                        help='If True, using Transformer decoder in our decoder. Otherwise, using Transformer encoder')
    parser.add_argument('--nhidlast2', type=int, default=-1,
                        help='hidden embedding size of the second LSTM')
    parser.add_argument('--n_basis', type=int, default=3,
                        help='number of basis we want to predict')
    parser.add_argument('--positional_option', type=str, default='linear',
                        help='options of encode positional embedding into models (linear, cat, add)')
    parser.add_argument('--linear_mapping_dim', type=int, default=0,
                        help='map the input embedding by linear transformation')
    #parser.add_argument('--postional_option', type=str, default='linear',
    #                help='options of encode positional embedding into models (linear, cat, add)')
    parser.add_argument('--dropoutp', type=float, default=0,
                        help='dropout of positional embedding or input embedding after linear transformation (when linear_mapping_dim != 0)')
    parser.add_argument('--dropout_prob_trans', type=float, default=0,
                    help='hidden_dropout_prob and attention_probs_dropout_prob in Transformer')
    parser.add_argument('--dropout_prob_lstm', type=float, default=0,
                    help='dropout_prob in LSTM')

    ##scoring
    parser.add_argument('--coeff_opt', type=str, default='max',
                        help='Could be prod or lc or max')
    parser.add_argument('--loss_type', type=str, default='sim',
                        help='Could be sim or dist')

def predict_batch_simple(feature, feature_type, parallel_encoder, parallel_decoder, normalize_emb = True):
    #output_emb, hidden, output_emb_last = parallel_encoder(feature.t())
    #output_emb_last = parallel_encoder(feature)
    output_emb_last, output_emb = parallel_encoder(feature, feature_type)
    basis_pred =  parallel_decoder(output_emb_last, output_emb)
    #basis_pred, coeff_pred = nsd_loss.predict_basis(parallel_decoder, n_basis, output_emb_last, predict_coeff_sum = True )
    
    if normalize_emb:
        basis_norm_pred = basis_pred / (0.000000000001 + basis_pred.norm(dim = 2, keepdim=True) )
    else:
        basis_norm_pred = basis_pred
    return basis_norm_pred, output_emb_last, output_emb
    

def predict_batch(feature, feature_type, parallel_encoder, parallel_decoder, word_norm_emb, top_k):
    
    basis_norm_pred, output_emb_last, output_emb = predict_batch_simple(feature, feature_type, parallel_encoder, parallel_decoder)
    #output_emb_last, output_emb = parallel_encoder(feature)
    #basis_pred, coeff_pred =  parallel_decoder(output_emb_last, output_emb, predict_coeff_sum = True)

    #coeff_sum = coeff_pred.cpu().numpy()
    
    #coeff_sum_diff = coeff_pred[:,:,0] - coeff_pred[:,:,1]
    #coeff_sum_diff_pos = coeff_sum_diff.clamp(min = 0)
    #coeff_sum_diff_cpu = coeff_sum_diff.cpu().numpy()
    #coeff_order = np.argsort(coeff_sum_diff_cpu, axis = 1)
    #coeff_order = np.flip( coeff_order, axis = 1 )

    #basis_pred = basis_pred.permute(0,2,1)
    #basis_norm_pred = basis_pred / (0.000000000001 + basis_pred.norm(dim = 1, keepdim=True) )
    basis_norm_pred = basis_norm_pred.permute(0,2,1)
    #basis_norm_pred should have dimension (n_batch, emb_size, n_basis)
    #word_norm_emb should have dimension (ntokens, emb_size)
    sim_pairwise = torch.matmul(word_norm_emb.unsqueeze(dim = 0), basis_norm_pred)
    #print(sim_pairwise.size())
    #sim_pairwise should have dimension (n_batch, ntokens, emb_size)
    top_value, top_index = torch.topk(sim_pairwise, top_k, dim = 1, sorted=True)
    
    bsz, max_sent_len, emb_size = output_emb.size()
    avg_out_emb = torch.empty(bsz, emb_size)
    for i in range(bsz):
        #print(feature[i,:])
        sent_len = (feature[i,:] != 0).sum()
        avg_out_emb[i,:] = output_emb[i,-sent_len:,:].mean(dim = 0)

    #return basis_norm_pred, coeff_order, coeff_sum, top_value, top_index, output_emb_last, avg_out_emb 
    return basis_norm_pred, top_value, top_index, output_emb_last, avg_out_emb 

def convert_feature_to_text(feature, idx2word_freq):
    if type(feature) is list:
        feature_list = feature
    else:
        feature_list = feature.tolist()
    feature_text = []
    for i in range( len(feature_list) ):
        current_sent = []
        for w_ind in feature_list[i]:
            if w_ind != 0:
                w = idx2word_freq[w_ind][0]
                current_sent.append(w)
        feature_text.append(current_sent)
    return feature_text

def print_basis_text(feature, idx2word_freq, tag_idx2word_freq, top_value, top_index, i_batch, outf, word_imp_sim = None):
    #n_basis = coeff_order.shape[1]
    n_basis = top_index.shape[2]
    top_k = top_index.size(1)
    feature_text = convert_feature_to_text(feature, idx2word_freq)
    for i_sent in range(len(feature_text)):
        #outf.write('{} batch, {}th sent: '.format(i_batch, i_sent)+' '.join(feature_text[i_sent])+'\n')
        if word_imp_sim is not None:
            #outf.write(' '.join('{0:.2f}'.format(x/4.0) for x in word_imp_sim[i_sent]])+'\n')
            outf.write(' '.join(['\\colorbox{{c{0:02}}}{{{1}}}'.format(int(100-y*100/4.0),x ) for x, y in zip( feature_text[i_sent], word_imp_sim[i_sent] )])+'\n')
        else:
            outf.write(' '.join(feature_text[i_sent])+'\n')

        for j in range(n_basis):
            #org_ind = coeff_order[i_sent, j]
            org_ind = j
            #outf.write(str(j)+', org '+str(org_ind)+', '+str( coeff_sum[i_sent,org_ind,0] )+' - '+str( coeff_sum[i_sent,org_ind,1] )+': ')

            for k in range(top_k):
                word_nn = tag_idx2word_freq[top_index[i_sent,k,org_ind].item()][0]
                outf.write( word_nn+' {:5.3f}'.format(top_value[i_sent,k,org_ind].item())+' ' )
            outf.write('\n')
        outf.write('\n')




def visualize_topics_val(dataloader, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, tag_idx2word_freq, outf, max_batch_num):
    #topics_num = 0
    top_k = 5
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            feature, feature_type, user, tag, repeat_num, user_len, tag_len = sample_batched

            basis_norm_pred, top_value, top_index, encoded_emb, avg_encoded_emb= predict_batch(feature, feature_type, parallel_encoder, parallel_decoder, word_norm_emb, top_k)
            #print_basis_text(feature, idx2word_freq, coeff_order, coeff_sum, top_value, top_index, i_batch, outf, word_imp_sim)
            print_basis_text(feature, idx2word_freq, tag_idx2word_freq, top_value, top_index, i_batch, outf)

            if i_batch >= max_batch_num:
                break

num_special_token = 3
def compute_all_dist(feature, feature_type, parallel_encoder, parallel_decoder, user_norm_emb, tag_norm_emb, coeff_opt, loss_type, device):
    def compute_dist(user_norm_emb, basis_norm_pred, coeff_opt, loss_type, device):
        lr_coeff = 0.05
        iter_coeff = 60
        coeff_opt_algo = "rmsprop"
        L1_losss_B = 0.2
        bsz = basis_norm_pred.size(0)
        vocab_size, emb_size = user_norm_emb.size()
        target_embeddings = user_norm_emb.expand( (bsz, vocab_size, emb_size) )
        with torch.enable_grad():
            if coeff_opt == 'prod':
                coeff_mat_user = nsd_loss.estimate_prod_coeff_mat_batch_opt(target_embeddings.detach(), basis_norm_pred.detach(), L1_losss_B, device, coeff_opt_algo, lr_coeff, iter_coeff)
            elif coeff_opt == 'lc':
                coeff_mat_user = nsd_loss.estimate_coeff_mat_batch_opt(target_embeddings.detach(), basis_norm_pred.detach(), L1_losss_B, device, coeff_opt_algo, lr_coeff, iter_coeff)
            elif coeff_opt == 'max':
                coeff_mat_user = nsd_loss.estimate_coeff_mat_batch_max(target_embeddings.detach(), basis_norm_pred.detach(), device, loss_type)
        
        pred_embeddings_user = torch.bmm(coeff_mat_user, basis_norm_pred)
        if loss_type == 'sim':
            dist_all_user_tensor = - torch.sum( pred_embeddings_user * target_embeddings, dim = 2 )
        elif loss_type == 'dist':
            dist_all_user_tensor = torch.pow( torch.norm( pred_embeddings_user - target_embeddings, dim = 2 ), 2)
        #To exclude <null>, <eos>, <unk>, just give it a super high distance
        dist_all_user_tensor[:,:num_special_token] = 100
        return dist_all_user_tensor.tolist()
    
    normalize_emb = True
    if loss_type != 'dist':
        normalize_emb = False
        
    basis_norm_pred, output_emb_last, output_emb = predict_batch_simple(feature, feature_type, parallel_encoder, parallel_decoder, normalize_emb)
    all_dist_user = compute_dist(user_norm_emb, basis_norm_pred, coeff_opt, loss_type, device)
    all_dist_tag = compute_dist(tag_norm_emb, basis_norm_pred, coeff_opt, loss_type, device)

    return all_dist_user, all_dist_tag

def print_rank(feature_text_j, user_text_j, tag_text_j, gt_rank_user_j, gt_rank_tag_j, top_user_text_j, top_tag_text_j, top_values_user_j, top_values_tag_j, paper_id, outf):
    outf.write(str(paper_id) + ', ' + ' '.join(feature_text_j)+'\n')
    
    for k in range(len(user_text_j)):
        outf.write( user_text_j[k] + ':r' + str(gt_rank_user_j[k]) + ' ' )
    outf.write('\n')
    
    for k in range(len(tag_text_j)):
        outf.write( tag_text_j[k] + ':r' + str(gt_rank_tag_j[k]) + ' ' )
    outf.write('\n\n')
    
    for k in range(len(top_user_text_j)):
        outf.write( top_user_text_j[k] + ':v' + str(top_values_user_j[k])+ ' ' )
    outf.write('\n')
    
    for k in range(len(top_tag_text_j)):
        outf.write( top_tag_text_j[k] + ':v' + str(top_values_tag_j[k])+ ' ' )
    outf.write('\n\n\n')


def recommend_test(dataloader_info, parallel_encoder, parallel_decoder, user_norm_emb, tag_norm_emb, idx2word_freq, user_idx2word_freq, tag_idx2word_freq, coeff_opt, loss_type, outf, device):
    def update_user_dict(user_batch_list, user_d2_paper_id, paper_id):
        for k in range(len(user_batch_list)):
            user_id = user_batch_list[k]
            if user_id < num_special_token:
                #break
                continue
            paper_id_list = user_d2_paper_id.get(user_id, [])
            paper_id_list.append(paper_id)
            user_d2_paper_id[user_id] = paper_id_list
     
    def gt_rank_from_list(all_dist_user_j, user_batch_list_j, num_special_token):
        user_rank = ss.rankdata(all_dist_user_j)
        gt_rank_j = []
        for user_id in user_batch_list_j:
            if user_id < num_special_token:
                #break
                continue
            else:
                gt_rank_j.append(user_rank[user_id])
        return gt_rank_j

    def compute_AP(gt_rank_j):
        gt_rank_j_sorted = sorted(gt_rank_j)
        AP_list = []
        for i, x in enumerate(gt_rank_j_sorted):
            AP_list.append( (i+1) / float(x) )
        return np.mean(AP_list)

    def pred_per_paper(all_dist_user, user_batch_list, recall_at_th):
        bsz = len(user_batch_list)
        gt_rank_user = []
        top_prediction_user = []
        top_values_user = []
        recall_list_user = []
        MAP_list_user = []
        for j in range(bsz):
            #pred_user_rank = np.argsort(all_dist_user[j])
            #gt_rank_j = [user_rank[x] for x in user_batch_list[j]]
            gt_rank_j = gt_rank_from_list(all_dist_user[j], user_batch_list[j], num_special_token = num_special_token)
            top_pred_j = np.argsort(all_dist_user[j])[:recall_at_th]
            top_prediction_user.append( top_pred_j )
            top_values_user.append( [all_dist_user[j][x] for x in top_pred_j] )
            gt_rank_user.append(gt_rank_j)
            if len(gt_rank_j) > 0:
                recall_list_user.append( np.sum([int(x <= recall_at_th) for x in gt_rank_j ]) / min(len(gt_rank_j), recall_at_th) )
                AP = compute_AP(gt_rank_j)
                MAP_list_user.append( AP )

        return gt_rank_user, recall_list_user, top_prediction_user, top_values_user, MAP_list_user
    
    def paper_recall_per_user(user_d2_paper_id, paper_user_dist, recall_at_th):
        user_paper_dist = list(zip(*paper_user_dist))
        recall_list_paper = []
        MAP_list_paper = []
        for user_id in user_d2_paper_id:
            paper_dist = user_paper_dist[user_id]
            paper_id_list = user_d2_paper_id[user_id]
            gt_rank_paper = gt_rank_from_list(paper_dist, paper_id_list, num_special_token = -1)
            recall_list_paper.append( np.sum([int(x <= recall_at_th) for x in gt_rank_paper ]) / min(len(gt_rank_paper), recall_at_th) )
            AP = compute_AP(gt_rank_paper)
            MAP_list_paper.append( AP )


        #print(recall_list_paper)
        recall_avg = np.mean(recall_list_paper)
        MAP = np.mean(MAP_list_paper)
        return recall_avg, MAP
    
    with torch.no_grad():
        paper_user_dist = []
        paper_tag_dist = []
        user_d2_paper_id = {}
        tag_d2_paper_id = {}
        recall_at_th = 50
        recall_all_user = []
        recall_all_tag = []
        MAP_all_user = []
        MAP_all_tag = []

        dataloader, all_user_tag = dataloader_info
        for i_batch, sample_batched in enumerate(dataloader):
            #feature, user, tag = sample_batched
            sys.stdout.write( str(i_batch) + ' ' )
            sys.stdout.flush()

            feature, feature_type, paper_id_tensor = sample_batched
            paper_id_list = paper_id_tensor.tolist()
            user_batch_list = [all_user_tag[paper_id][0] for paper_id in paper_id_list]
            tag_batch_list = [all_user_tag[paper_id][1] for paper_id in paper_id_list]
            
            all_dist_user, all_dist_tag = compute_all_dist(feature, feature_type, parallel_encoder, parallel_decoder, user_norm_emb, tag_norm_emb, coeff_opt, loss_type, device)
            #user_batch_list = user.tolist()
            #tag_batch_list = tag.tolist()
            gt_rank_user, recall_list_user, top_prediction_user, top_values_user, MAP_list_user = pred_per_paper(all_dist_user, user_batch_list, recall_at_th)
            gt_rank_tag, recall_list_tag, top_prediction_tag, top_values_tag, MAP_list_tag = pred_per_paper(all_dist_tag, tag_batch_list, recall_at_th)
            recall_all_user += recall_list_user
            recall_all_tag += recall_list_tag
            MAP_all_user += MAP_list_user
            MAP_all_tag += MAP_list_tag

            bsz = feature.size(0)
            #user_max = user.size(1)
            #tag_max = tag.size(1)

            feature_text = convert_feature_to_text(feature, idx2word_freq)
            user_text = convert_feature_to_text(user_batch_list, user_idx2word_freq)
            tag_text = convert_feature_to_text(tag_batch_list, tag_idx2word_freq)
            top_user_text = convert_feature_to_text(top_prediction_user, user_idx2word_freq)
            top_tag_text = convert_feature_to_text(top_prediction_tag, tag_idx2word_freq)
            for j in range(bsz):
                #paper_id = len(paper_user_dist)
                paper_id = paper_id_list[j]
                update_user_dict(user_batch_list[j], user_d2_paper_id, paper_id)
                update_user_dict(tag_batch_list[j], tag_d2_paper_id, paper_id)
                paper_user_dist.append(all_dist_user[j])
                paper_tag_dist.append(all_dist_tag[j])

                print_rank(feature_text[j], user_text[j], tag_text[j], gt_rank_user[j], gt_rank_tag[j], top_user_text[j], top_tag_text[j], top_values_user[j], top_values_tag[j], paper_id, outf)
            #if i_batch > 3:
            #    break
            #print_basis_text(feature, idx2word_freq, tag_idx2word_freq, coeff_order, coeff_sum, top_value, top_index, i_batch, outf)
        print("\nUser recall per paper at {} is {}, MAP is {}".format(recall_at_th,np.mean(recall_all_user), np.mean(MAP_all_user)))
        print("Tag recall per paper at {} is {}, MAP is {}".format(recall_at_th,np.mean(recall_all_tag),  np.mean(MAP_all_tag)))
        recall_avg_user, MAP_user = paper_recall_per_user(user_d2_paper_id, paper_user_dist, recall_at_th)
        print("Paper recall per user at {} is {}, MAP is {}".format(recall_at_th, recall_avg_user, MAP_user))
        recall_avg_tag, MAP_tag = paper_recall_per_user(tag_d2_paper_id, paper_tag_dist, recall_at_th)
        print("Paper recall per tag at {} is {}, MAP is {}".format(recall_at_th, recall_avg_tag, MAP_tag))
            #if i_batch >= max_batch_num:
            #    break

class Set2SetDataset(torch.utils.data.Dataset):
    #def __init__(self, source, source_w, source_sent_emb, source_avg_word_emb, source_w_imp_list, source_proc_sent, target, target_w, target_sent_emb, target_avg_word_emb, target_w_imp_list, target_proc_sent):
    def __init__(self, source, source_w, source_sent_emb, source_avg_word_emb, target, target_w, target_sent_emb, target_avg_word_emb):
        self.source = source
        self.source_w = source_w
        self.source_sent_emb = source_sent_emb
class Set2SetDataset(torch.utils.data.Dataset):
    #def __init__(self, source, source_w, source_sent_emb, source_avg_word_emb, source_w_imp_list, source_proc_sent, target, target_w, target_sent_emb, target_avg_word_emb, target_w_imp_list, target_proc_sent):
    def __init__(self, source, source_w, source_sent_emb, source_avg_word_emb, target, target_w, target_sent_emb, target_avg_word_emb):
        self.source = source
        self.source_w = source_w
        self.source_sent_emb = source_sent_emb
        self.source_avg_word_emb = source_avg_word_emb
        #self.source_w_imp_list = source_w_imp_list
        #self.source_proc_sent = source_proc_sent
        self.target = target
        self.target_w = target_w
        self.target_sent_emb = target_sent_emb
        self.target_avg_word_emb = target_avg_word_emb
        #self.target_w_imp_list = target_w_imp_list
        #self.target_proc_sent = target_proc_sent

    def __len__(self):
        return self.source.size(0)

    def __getitem__(self, idx):
        source = self.source[idx, :, :]
        source_w = self.source_w[idx, :]
        source_sent_emb = self.source_sent_emb[idx, :]
        source_avg_word_emb = self.source_avg_word_emb[idx, :]
        #source_w_imp_list = self.source_w_imp_list[idx]
        #source_proc_sent = self.source_proc_sent[idx]
        target = self.target[idx, :, :]
        target_w = self.target_w[idx, :]
        target_sent_emb = self.target_sent_emb[idx, :]
        target_avg_word_emb = self.target_avg_word_emb[idx, :]
        #target_w_imp_list = self.target_w_imp_list[idx]
        #target_proc_sent = self.target_proc_sent[idx]
        #debug target[-1] = idx
        #return [source, source_w, source_sent_emb, source_avg_word_emb, source_w_imp_list, source_proc_sent, target, target_w, target_sent_emb, target_avg_word_emb, target_w_imp_list, target_proc_sent]
        return [source, source_w, source_sent_emb, source_avg_word_emb, target, target_w, target_sent_emb, target_avg_word_emb, idx]

def build_loader_from_pairs(testing_list, sent_d2_topics, bsz, device):
    def store_topics(sent, sent_d2_topics, topic_v_tensor, topic_w_tensor, sent_emb_tensor, avg_word_emb_tensor, w_imp_list, proc_sent_list, i_pairs, device):
        topic_v, topic_w, sent_emb, avg_word_emb, w_imp_arr, proc_sent = sent_d2_topics[sent]
        topic_v_tensor[i_pairs, :, :] = torch.tensor(topic_v, device = device)
        topic_w_tensor[i_pairs, :] = torch.tensor(topic_w, device = device)
        sent_emb_tensor[i_pairs, :] = torch.tensor(sent_emb, device = device)
        avg_word_emb_tensor[i_pairs, :] = torch.tensor(avg_word_emb, device = device)
        w_imp_list[i_pairs] = w_imp_arr
        proc_sent_list[i_pairs] = proc_sent
    
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
    avg_word_emb_tensor_1 = torch.empty(corpus_size, encoder_emsize, device = device)
    w_imp_list_1 = [0]*corpus_size
    proc_sent_list_1 = [0]*corpus_size
    topic_v_tensor_2 = torch.empty(corpus_size, n_basis, emb_size, device = device)
    topic_w_tensor_2 = torch.empty(corpus_size, n_basis, device = device)
    sent_emb_tensor_2 = torch.empty(corpus_size, encoder_emsize, device = device)
    avg_word_emb_tensor_2 = torch.empty(corpus_size, encoder_emsize, device = device)
    w_imp_list_2 = [0]*corpus_size
    proc_sent_list_2 = [0]*corpus_size

    for i_pairs, fields in enumerate(testing_list):
        sent_1 = fields[0]
        sent_2 = fields[1]
        store_topics(sent_1, sent_d2_topics, topic_v_tensor_1, topic_w_tensor_1, sent_emb_tensor_1, avg_word_emb_tensor_1, w_imp_list_1, proc_sent_list_1, i_pairs, device)
        store_topics(sent_2, sent_d2_topics, topic_v_tensor_2, topic_w_tensor_2, sent_emb_tensor_2, avg_word_emb_tensor_2, w_imp_list_2, proc_sent_list_2, i_pairs, device)
    #print(w_imp_list_1)
    #dataset = Set2SetDataset(topic_v_tensor_1, topic_w_tensor_1, sent_emb_tensor_1, avg_word_emb_tensor_1, w_imp_list_1, proc_sent_list_1, topic_v_tensor_2, topic_w_tensor_2, sent_emb_tensor_2, avg_word_emb_tensor_2, w_imp_list_2, proc_sent_list_2)
    dataset = Set2SetDataset(topic_v_tensor_1, topic_w_tensor_1, sent_emb_tensor_1, avg_word_emb_tensor_1, topic_v_tensor_2, topic_w_tensor_2, sent_emb_tensor_2, avg_word_emb_tensor_2)
    use_cuda = False
    if device == 'cude':
        use_cuda = True
    testing_pair_loader = torch.utils.data.DataLoader(dataset, batch_size = bsz, shuffle = False, pin_memory=use_cuda, drop_last=False)
    return testing_pair_loader, (w_imp_list_1, proc_sent_list_1, w_imp_list_2, proc_sent_list_2)

def compute_freq_prob(word_d2_idx_freq):
    all_idx, all_freq= list( zip(*word_d2_idx_freq.values()) )
    freq_sum = float(sum(all_freq))
    for w in word_d2_idx_freq:
        idx, freq = word_d2_idx_freq[w]
        word_d2_idx_freq[w].append(freq/freq_sum)

def safe_cosine_sim(emb_1, emb_2):
    dist = distance.cosine(emb_1, emb_2)
    if math.isnan(dist):
        return 0
    else:
        return 1 - dist

def compute_AP_best_F1_acc(score_list, gt_list, correct_label = 1):
    sorted_idx = np.argsort(score_list)
    sorted_idx = sorted_idx.tolist()[::-1]
    #print(sorted_idx)
    #total_correct = sum(gt_list)
    total_correct = sum([1 if x == correct_label else 0 for x in  gt_list ])
    correct_count = 0
    total_count = 0
    precision_list = []
    F1_list = []
    acc_list = []
    false_neg_num = total_correct
    for idx in sorted_idx:
        total_count += 1
        if gt_list[idx] == correct_label:
            correct_count += 1
            precision = correct_count/float(total_count)
            precision_list.append( precision )
            recall = correct_count / float(total_correct)
            F1_list.append(  2*(precision*recall)/(recall+precision) )
            false_neg_num = total_correct - correct_count
        rest_num = len(sorted_idx) - total_count
        true_neg_num = rest_num - false_neg_num
        acc_list.append( (correct_count + true_neg_num) / float(len(sorted_idx)) )
    return np.mean(precision_list), np.max(F1_list), np.max(acc_list)

def safe_normalization(weight):
    weight_sum = torch.sum(weight, dim = 1, keepdim=True)
    #assert weight_sum>0
    return weight / (weight_sum + 0.000000000001)

def compute_cosine_sim(source, target):
    #assume that two matrices have been normalized
    C = target.permute(0,2,1)
    cos_sim_st = torch.bmm(source, C)
    cos_sim_ts = cos_sim_st.permute(0,2,1)
    return cos_sim_st, cos_sim_ts 

def weighted_average(cosine_sim, weight):
    #assume that weight are normalized
    return (cosine_sim * weight).mean(dim = 1)

def max_cosine_given_sim(cos_sim_st, target_w = None):
    cosine_sim_s_to_t, max_i = torch.max(cos_sim_st, dim = 1)
    #cosine_sim should have dimension (n_batch,n_set)
    sim_avg_st = cosine_sim_s_to_t.mean(dim = 1)
    if target_w is None:
        return sim_avg_st
    sim_w_avg_st = weighted_average(cosine_sim_s_to_t, target_w)
    return sim_avg_st, sim_w_avg_st

def lc_pred_dist(target, source, target_w, L1_losss_B, device):
    #coeff_mat_s_to_t = nsd_loss.estimate_coeff_mat_batch(target, source, L1_losss_B, device, max_iter = 1000)
    with torch.enable_grad():
        coeff_mat_s_to_t = nsd_loss.estimate_coeff_mat_batch_opt(target, source, L1_losss_B, device, coeff_opt = 'rmsprop', lr = 0.05, max_iter = 1000)
    pred_embeddings = torch.bmm(coeff_mat_s_to_t, source)
    dist_st = torch.pow( torch.norm( pred_embeddings - target, dim = 2 ), 2)
    #dist_st = torch.norm( pred_embeddings - target, dim = 2 )
    dist_avg_st = dist_st.mean(dim = 1)
    if target_w is None:
        return dist_avg_st
    dist_w_avg_st = weighted_average(dist_st, target_w)
    return dist_avg_st, dist_w_avg_st

#def get_w_emb(proc_sent, word2emb, emb_size, word_d2_idx_freq):
#    w_embs = []
#    w_prob = []
#    for w in proc_sent:
#        if w in word2emb:
#            w_embs.append(word2emb[w].reshape(1,-1))
#            if word_d2_idx_freq is not None:
#                w_prob.append(word_d2_idx_freq[w][2])
#            else:
#                w_prob.append(1)
#        else:
#            #assert w == '<eos>' or w == '<unk>', w
#            w_embs.append(np.zeros( (1,emb_size) ))
#            w_prob.append(0)
#    w_embs = np.concatenate( w_embs, axis = 0 )
#    w_prob = np.array(w_prob)
#    if word_d2_idx_freq is None:
#        w_prob_sum = w_prob.sum()
#        if w_prob_sum > 0:
#            w_prob = w_prob / w_prob_sum
#    return w_embs, w_prob

def get_w_emb(proc_sent, word2emb, emb_size):
    w_embs = []
    for w in proc_sent:
        if w in word2emb:
            w_embs.append(word2emb[w].reshape(1,-1))
        else:
            w_embs.append(np.zeros( (1,emb_size) ))
    w_embs = np.concatenate( w_embs, axis = 0 )
    return w_embs

def get_w_prob(proc_sent, word2emb, word_d2_idx_freq):
    w_prob = []
    for i, w in enumerate(proc_sent):
        if w in word2emb:
            #if word_d2_idx_freq is not None:
            if isinstance(word_d2_idx_freq,dict):
                w_prob.append(word_d2_idx_freq[w][2])
            elif isinstance(word_d2_idx_freq,list):
                w_prob.append(word_d2_idx_freq[i])
            else:
                w_prob.append(1)
        else:
            #assert w == '<eos>' or w == '<unk>', w
            w_prob.append(0)
    w_prob = np.array(w_prob)
    if word_d2_idx_freq is None or isinstance(word_d2_idx_freq,list):
        w_prob_sum = w_prob.sum()
        if w_prob_sum > 0:
            w_prob = w_prob / w_prob_sum
    return w_prob

def weighted_sum_emb_list(w_embs, w_imp=None):
    if w_imp is None:
        return np.sum(w_embs, axis = 0)
    else:
        return np.matmul(w_imp, w_embs)

def check_OOV(s_sent_emb, t_sent_emb):
    OOV_first = 0
    OOV_second = 0
    if np.sum(np.abs(s_sent_emb)) == 0:
        OOV_first = 1
    if np.sum(np.abs(t_sent_emb)) == 0:
        OOV_second = 1
    OOV_all_sent = [OOV_first, OOV_second]
    return OOV_all_sent

def reconstruct_sent_emb(source, t_w_emb_tensor, L1_losss_B, device, weights = None):
    cos_sim_st, cos_sim_ts = compute_cosine_sim(source, t_w_emb_tensor)
    if weights is None:
        sim_avg_st = max_cosine_given_sim(cos_sim_st)
        dist_avg_st = lc_pred_dist(t_w_emb_tensor, source, None, L1_losss_B, device)
        return sim_avg_st, dist_avg_st
    else:
        sim_avg_st, sim_w_avg_st = max_cosine_given_sim(cos_sim_st, weights)
        dist_avg_st, dist_w_avg_st = lc_pred_dist(t_w_emb_tensor, source, weights, L1_losss_B, device)
        return sim_w_avg_st, dist_w_avg_st

def store_w_embs_prob_to_tensors(w_uni_tensor, w_embs_tensor, i, w_embs, w_prob, device):
    w_num = w_embs.shape[0]
    w_embs_tensor[i,-w_num:,:] = torch.tensor(w_embs, device = device)
    w_uni_tensor[i,-w_num:] = torch.tensor(w_prob, device = device)

def convert_prob_to_weights(proc_sent, word_d2_idx_freq, alpha):
    w_weights_raw = []
    for w in proc_sent:
        if w in word_d2_idx_freq:
            prob = word_d2_idx_freq[w][2]
            w_weights_raw.append( alpha / (alpha + prob) )
        else:
            w_weights_raw.append(0)
    return w_weights_raw

def predict_entail_scores(testing_pair_loader, L1_losss_B, device, word2emb, word_d2_idx_freq, other_info, compute_WMD = True, OOV_sim_zero = True):

    corpus_size = len(testing_pair_loader.dataset)
    pred_scores = [0] * corpus_size
    w_imp_list_1, proc_sent_list_1, w_imp_list_2, proc_sent_list_2 = other_info
    max_phrase_len = max( [len(x) for x in proc_sent_list_1 + proc_sent_list_2 ])
    
    emb_size = list(word2emb.values())[0].size
    sent_embs_SIF = [0] * corpus_size
    OOV_first_list = [0] * corpus_size
    OOV_second_list = [0] * corpus_size
    OOV_value = -100000000000
    for source, source_w, source_sent_emb, source_avg_word_emb, target, target_w, target_sent_emb, target_avg_word_emb, idx_batch in testing_pair_loader:
        #normalize w
        sys.stdout.write( str(idx_batch[0].item()) + ' ' )
        sys.stdout.flush()
        source_w = safe_normalization(source_w)
        target_w = safe_normalization(target_w)
        
        #Kmeans loss
        cos_sim_st, cos_sim_ts = compute_cosine_sim(source, target)
        
        uniform_s = torch.ones_like(source_w) / source_w.size(1)
        uniform_t = torch.ones_like(target_w) / target_w.size(1)
        cos_dist_st = 1 - cos_sim_st
        #Wasserstein_trans = Sinkhorn.batch_sinkhorn_loss_weighted( cos_dist_st, uniform_s, uniform_t, epsilon=1, niter=100)
        #Wasserstein_dist = torch.sum( Wasserstein_trans * cos_dist_st, dim = 2).sum(dim = 1)
        #Wasserstein_trans_w = Sinkhorn.batch_sinkhorn_loss_weighted( cos_dist_st, source_w, target_w, epsilon=1, niter=100)
        #Wasserstein_dist_w = torch.sum( Wasserstein_trans_w * cos_dist_st, dim = 2).sum(dim = 1)

        sim_avg_st, sim_w_avg_st = max_cosine_given_sim(cos_sim_st, target_w)
        sim_avg_ts, sim_w_avg_ts = max_cosine_given_sim(cos_sim_ts, source_w)
        #sim_avg = (sim_avg_st + sim_avg_ts)/2
        #sim_w_avg = (sim_w_avg_st + sim_w_avg_ts)/2
        sim_avg = sim_avg_st
        sim_w_avg = sim_w_avg_st

        sim_avg_max = torch.max( sim_avg_st, sim_avg_ts )
        sim_avg_min = torch.min( sim_avg_st, sim_avg_ts )
        sim_avg_diff = sim_avg_ts - sim_avg_st
        sim_avg_diff_inv = sim_avg_st - sim_avg_ts
        sim_w_avg_diff = sim_w_avg_ts - sim_w_avg_st
        sim_w_avg_diff_inv = sim_w_avg_st - sim_w_avg_ts

        dist_avg_st, dist_w_avg_st = lc_pred_dist(target, source, target_w, L1_losss_B, device)
        dist_avg_ts, dist_w_avg_ts = lc_pred_dist(source, target, source_w, L1_losss_B, device)
        #dist_avg = (dist_avg_st + dist_avg_ts)/2
        #dist_w_avg = (dist_w_avg_st + dist_w_avg_ts)/2
        dist_avg = dist_avg_st
        dist_w_avg = dist_w_avg_st
        dist_avg_diff = dist_avg_ts - dist_avg_st
        dist_avg_diff_inv = dist_avg_st - dist_avg_ts
        dist_w_avg_diff = dist_w_avg_ts - dist_w_avg_st
        dist_w_avg_diff_inv = dist_w_avg_st - dist_w_avg_ts

        cosine_sim = F.cosine_similarity(source_sent_emb, target_sent_emb, dim = 1)
        cosine_sim_word = F.cosine_similarity(source_avg_word_emb, target_avg_word_emb, dim = 1)

        #s_w_emb_avg_tensor = torch.empty(sim_avg.size(0), 1, emb_size, device = device)
        #t_w_emb_avg_tensor = torch.empty(sim_avg.size(0), 1, emb_size, device = device)
        s_w_uni_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, device = device)
        t_w_uni_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, device = device)
        s_w_freq4_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, device = device)
        t_w_freq4_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, device = device)
        s_w_freq4_sim_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, device = device)
        t_w_freq4_sim_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, device = device)
        s_w_sim_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, device = device)
        t_w_sim_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, device = device)
        s_w_embs_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, emb_size, device = device)
        t_w_embs_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, emb_size, device = device)
        baseline_score_list = [0]* sim_avg.size(0)
        for i in range(sim_avg.size(0)):
        #coeff_mat should have dimension (n_batch,n_set,n_basis)
            idx = idx_batch[i]
            source_proc_sent = proc_sent_list_1[idx]
            target_proc_sent = proc_sent_list_2[idx]
            source_w_imp_list = w_imp_list_1[idx]
            target_w_imp_list = w_imp_list_2[idx]
            w_embs_source = get_w_emb(source_proc_sent, word2emb, emb_size)
            w_embs_target = get_w_emb(target_proc_sent, word2emb, emb_size)
            w_uni_prob_source = get_w_prob(source_proc_sent, word2emb, word_d2_idx_freq = None)
            w_uni_prob_target = get_w_prob(target_proc_sent, word2emb, word_d2_idx_freq = None)
            
            alpha = 0.0001
            #w_prob_source = get_w_prob(source_proc_sent, word2emb, word_d2_idx_freq)
            #w_prob_target = get_w_prob(target_proc_sent, word2emb, word_d2_idx_freq)
            #w_weights_source = alpha / (w_prob_source+alpha)
            #w_weights_target = alpha / (w_prob_target+alpha)

            w_weights_raw_source = convert_prob_to_weights(source_proc_sent, word_d2_idx_freq, alpha)
            w_weights_raw_target = convert_prob_to_weights(target_proc_sent, word_d2_idx_freq, alpha)
            w_weights_source = get_w_prob(source_proc_sent, word2emb, word_d2_idx_freq = w_weights_raw_source)
            w_weights_target = get_w_prob(target_proc_sent, word2emb, word_d2_idx_freq = w_weights_raw_target)
            
            w_weights_sim_raw_source = (source_w_imp_list[0] * w_weights_raw_source).tolist()
            w_weights_sim_raw_target = (target_w_imp_list[0] * w_weights_raw_target).tolist()
            w_weights_sim_source = get_w_prob(source_proc_sent, word2emb, word_d2_idx_freq = w_weights_sim_raw_source)
            w_weights_sim_target = get_w_prob(target_proc_sent, word2emb, word_d2_idx_freq = w_weights_sim_raw_target)
            
            w_sim_raw_source = source_w_imp_list[0].tolist()
            w_sim_raw_target = target_w_imp_list[0].tolist()
            w_sim_source = get_w_prob(source_proc_sent, word2emb, word_d2_idx_freq = w_sim_raw_source)
            w_sim_target = get_w_prob(target_proc_sent, word2emb, word_d2_idx_freq = w_sim_raw_target)

            if compute_WMD:
                #w_num = len(source_proc_sent)
                #s_w_weights_tensor[i,-w_num:] = torch.tensor(w_weights_source, device = device)
                #w_num = len(target_proc_sent)
                #t_w_weights_tensor[i,-w_num:] = torch.tensor(w_weights_target, device = device)
                
                w_embs_source_norm = w_embs_source / (0.000000000001 + np.linalg.norm(w_embs_source, axis = 1, keepdims=True))
                w_embs_target_norm = w_embs_target / (0.000000000001 + np.linalg.norm(w_embs_target, axis = 1, keepdims=True))
                M = 1 - np.matmul( w_embs_source_norm,  np.transpose(w_embs_target_norm) )
                Wd = ot.emd2(w_uni_prob_source, w_uni_prob_target, M)
                #Wd_reg = ot.sinkhorn2(w_uni_source, w_uni_target, M, 1)
                Wd_w = ot.emd2(w_weights_source, w_weights_target, M)
            else:
                Wd = np.random.rand(1)[0]
                #Wd_reg = np.random.rand(1)[0]
                Wd_w = np.random.rand(1)[0]
            
            w_num = w_embs_source.shape[0]
            s_w_freq4_tensor[i,-w_num:] = torch.tensor(w_weights_source, device = device)
            s_w_freq4_sim_tensor[i,-w_num:] = torch.tensor(w_weights_sim_source, device = device)
            s_w_sim_tensor[i,-w_num:] = torch.tensor(w_sim_source, device = device)
            w_num = w_embs_target.shape[0]
            t_w_freq4_tensor[i,-w_num:] = torch.tensor(w_weights_target, device = device)
            t_w_freq4_sim_tensor[i,-w_num:] = torch.tensor(w_weights_sim_target, device = device)
            t_w_sim_tensor[i,-w_num:] = torch.tensor(w_sim_target, device = device)

            store_w_embs_prob_to_tensors(s_w_uni_tensor, s_w_embs_tensor, i, w_embs_source, w_uni_prob_source, device)
            store_w_embs_prob_to_tensors(t_w_uni_tensor, t_w_embs_tensor, i, w_embs_target, w_uni_prob_target, device)

            s_sent_emb = weighted_sum_emb_list(w_embs_source)
            t_sent_emb = weighted_sum_emb_list(w_embs_target)
            score_baseline = safe_cosine_sim(s_sent_emb, t_sent_emb)
            
            s_sent_emb_freq4 = weighted_sum_emb_list(w_embs_source, w_weights_source)
            t_sent_emb_freq4 = weighted_sum_emb_list(w_embs_target, w_weights_target)
            score_baseline_freq4 = safe_cosine_sim(s_sent_emb_freq4, t_sent_emb_freq4)
            
            s_sent_emb_freq4_sim = weighted_sum_emb_list(w_embs_source, w_weights_sim_source)
            t_sent_emb_freq4_sim = weighted_sum_emb_list(w_embs_target, w_weights_sim_target)
            score_baseline_freq4_sim = safe_cosine_sim(s_sent_emb_freq4_sim, t_sent_emb_freq4_sim)
            
            s_sent_emb_sim = weighted_sum_emb_list(w_embs_source, w_sim_source)
            t_sent_emb_sim = weighted_sum_emb_list(w_embs_target, w_sim_target)
            score_baseline_sim = safe_cosine_sim(s_sent_emb_sim, t_sent_emb_sim)
            #s_w_emb_avg_tensor[i,0,:] = torch.tensor(s_sent_emb, device = device)
            #t_w_emb_avg_tensor[i,0,:] = torch.tensor(t_sent_emb, device = device)

            OOV_first, OOV_second = check_OOV(s_sent_emb, t_sent_emb)

            OOV_first_list[idx] = OOV_first
            OOV_second_list[idx] = OOV_second
            
            baseline_score_list[i] = score_baseline
            if OOV_sim_zero and (OOV_first == 1 or OOV_second == 1):
                pred_scores[idx] = [OOV_value] * 26
            else:
                pred_scores[idx] = [-Wd, -Wd_w, -dist_avg[i].item(), -dist_w_avg[i].item(), sim_avg[i].item(), sim_w_avg[i].item(), cosine_sim[i].item(), cosine_sim_word[i].item(), score_baseline, score_baseline_sim, score_baseline_freq4, score_baseline_freq4_sim, sim_avg_max[i].item(), sim_avg_min[i].item(), sim_avg_diff[i].item(), sim_avg_diff[i].item() * sim_avg[i].item(), sim_avg_diff_inv[i].item(), sim_w_avg_diff[i].item(), sim_w_avg_diff_inv[i].item(), sim_avg_diff_inv[i].item() * sim_avg[i].item(), sim_w_avg_diff[i].item() * sim_avg[i].item(), dist_avg_diff[i].item(), dist_avg_diff_inv[i].item(), dist_w_avg_diff[i].item(), dist_w_avg_diff_inv[i].item(), dist_avg_diff_inv[i].item() + dist_w_avg_diff[i].item()]
        
        s_w_embs_tensor = s_w_embs_tensor /(0.000000000001 + s_w_embs_tensor.norm(dim = 2, keepdim=True) )
        t_w_embs_tensor = t_w_embs_tensor /(0.000000000001 + t_w_embs_tensor.norm(dim = 2, keepdim=True) )
        #s_w_emb_avg_tensor = s_w_emb_avg_tensor / (0.000000000001 + s_w_emb_avg_tensor.norm(dim = 2, keepdim=True) )
        #t_w_emb_avg_tensor = t_w_emb_avg_tensor / (0.000000000001 + t_w_emb_avg_tensor.norm(dim = 2, keepdim=True) )
        #sim_avg_st, dist_avg_st = reconstruct_sent_emb(source, t_w_emb_avg_tensor, L1_losss_B, device)
        #sim_avg_ts, dist_avg_ts = reconstruct_sent_emb(target, s_w_emb_avg_tensor, L1_losss_B, device)
        sim_w_emb_avg_st, dist_w_emb_avg_st = reconstruct_sent_emb(s_w_embs_tensor, t_w_embs_tensor, L1_losss_B, device, t_w_uni_tensor)
        #sim_w_emb_avg_ts, dist_w_emb_avg_ts = reconstruct_sent_emb(t_w_embs_tensor, s_w_embs_tensor, L1_losss_B, device, s_w_uni_tensor)
        sim_w_emb_freq4_avg_st, dist_w_emb_freq4_avg_st = reconstruct_sent_emb(s_w_embs_tensor, t_w_embs_tensor, L1_losss_B, device, t_w_freq4_tensor)
        sim_w_emb_freq4_avg_ts, dist_w_emb_freq4_avg_ts = reconstruct_sent_emb(t_w_embs_tensor, s_w_embs_tensor, L1_losss_B, device, s_w_freq4_tensor)
        
        sim_w_emb_topic_st, dist_w_emb_topic_st = reconstruct_sent_emb(s_w_embs_tensor, target, L1_losss_B, device, None)
        #sim_w_emb_freq4_topic_st, dist_w_emb_freq4_topic_st = reconstruct_sent_emb(s_w_embs_tensor, target, L1_losss_B, device, target_w)
        sim_w_emb_sim_avg_st, dist_w_emb_sim_avg_st = reconstruct_sent_emb(s_w_embs_tensor, t_w_embs_tensor, L1_losss_B, device, t_w_sim_tensor)
        sim_w_emb_freq4_sim_avg_st, dist_w_emb_freq4_sim_avg_st = reconstruct_sent_emb(s_w_embs_tensor, t_w_embs_tensor, L1_losss_B, device, t_w_freq4_sim_tensor)
        #sim_w_emb_freq4_sim_topic_st, dist_w_emb_freq4_sim_topic_st = reconstruct_sent_emb(s_w_embs_tensor, target, L1_losss_B, device, )
        
        sim_w_avg_st, dist_w_avg_st = reconstruct_sent_emb(source, t_w_embs_tensor, L1_losss_B, device, t_w_uni_tensor)
        sim_w_avg_ts, dist_w_avg_ts = reconstruct_sent_emb(target, s_w_embs_tensor, L1_losss_B, device, s_w_uni_tensor)

        sim_w_freq4_avg_st, dist_w_freq4_avg_st = reconstruct_sent_emb(source, t_w_embs_tensor, L1_losss_B, device, t_w_freq4_tensor)
        sim_w_freq4_avg_ts, dist_w_freq4_avg_ts = reconstruct_sent_emb(target, s_w_embs_tensor, L1_losss_B, device, s_w_freq4_tensor)

        for i, idx in enumerate(idx_batch):
            if OOV_sim_zero and (OOV_first_list[idx] == 1 or OOV_second_list[idx] == 1):
                pred_scores[idx] = [OOV_value] * 21  + pred_scores[idx]
            else:
                #pred_scores[idx] = [sim_avg_st[i].item() - sim_avg_ts[i].item(), sim_avg_ts[i].item() - sim_avg_st[i].item(), dist_avg_st[i].item() - dist_avg_ts[i].item(), dist_avg_ts[i].item() - dist_avg_st[i].item()] + pred_scores[idx]
                best_diff = dist_w_avg_ts[i].item() - dist_w_avg_st[i].item()
                pred_scores[idx] = [sim_w_freq4_avg_st[i].item(), sim_w_emb_freq4_sim_avg_st[i].item(), -dist_w_freq4_avg_st[i].item(), -dist_w_emb_freq4_sim_avg_st[i].item(), sim_w_avg_st[i].item(), sim_w_emb_sim_avg_st[i].item(), -dist_w_avg_st[i].item(), -dist_w_emb_sim_avg_st[i].item(), sim_w_emb_freq4_avg_st[i].item(), -dist_w_emb_freq4_avg_st[i].item(), sim_w_emb_avg_st[i].item(), sim_w_emb_topic_st[i].item(), -dist_w_emb_avg_st[i].item(), -dist_w_emb_topic_st[i].item(), sim_w_emb_freq4_avg_st[i].item() - sim_w_emb_freq4_avg_ts[i].item(), sim_w_freq4_avg_st[i].item() - sim_w_freq4_avg_ts[i].item(), dist_w_freq4_avg_ts[i].item() - dist_w_freq4_avg_st[i].item(), sim_w_avg_st[i].item() - sim_w_avg_ts[i].item(), dist_w_avg_ts[i].item() - dist_w_avg_st[i].item(), best_diff * sim_avg[i].item(),  best_diff * cosine_sim_word[i].item(), best_diff * baseline_score_list[i]  ] + pred_scores[idx]

    print("Total OOV first: ", np.sum(OOV_first_list), ", Total OOV second:", np.sum(OOV_second_list))
    #method_names = ["kmeans_sim_st_ts_diff", "kmeans_sim_ts_st_diff", "SC_dist_st_ts_diff", "SC_dist_ts_st_diff", "Wasserstein", "Wasserstein_w", "kmeans", "kmeans_w", "en_sent_emb", "avg_en_word_emb", "baseline", "kmeans_max", "kmeans_min", "kmeans_diff", "kmeans_diff_avg", "kmeans_diff_inv", "kmeans_diff_w", "kmeans_diff_w_inv", "kmeans_diff_inv_avg", "kmeans_diff_w_avg", "SC_diff", "SC_diff_inv", "SC_diff_w", "SC_diff_w_inv", "SC_diff_inv+SC_diff_w"] 
    method_names = ["kmeans_word_sim_st_freq4", "kmeans_word_emb_sim_st_freq4_sim", "SC_word_dist_st_freq4", "SC_word_emb_dist_st_freq4_sim", "kmeans_word_sim_st", "kmeans_word_emb_sim_st_sim", "SC_word_dist_st", "SC_word_emb_dist_st_sim", "kmeans_word_emb_sim_st_freq4", "SC_word_emb_dist_st_freq4", "kmeans_word_emb_sim_st", "kmeans_word_emb_sim_topic_st", "SC_word_emb_dist_st", "SC_word_emb_dist_topic_st", "kmeans_word_emb_sim_st_ts_diff_freq4", "kmeans_word_sim_st_ts_diff_freq4", "SC_word_dist_ts_st_diff_freq4",  "kmeans_word_sim_st_ts_diff", "SC_word_dist_ts_st_diff", "best_diff_kmeans", "best_diff_avg_en_word_emb", "best_diff_baseline", "WMD exact or rnd", "WMD weighted exact or rnd", "SC", "SC_w", "kmeans", "kmeans_w", "en_sent_emb", "avg_en_word_emb", "baseline", "baseline_sim", "baseline_freq4", "baseline_freq4_sim", "kmeans_max", "kmeans_min", "kmeans_diff", "kmeans_diff_avg", "kmeans_diff_inv", "kmeans_diff_w", "kmeans_diff_w_inv", "kmeans_diff_inv_avg", "kmeans_diff_w_avg", "SC_diff", "SC_diff_inv", "SC_diff_w", "SC_diff_w_inv", "SC_diff_inv+SC_diff_w"] 

    return pred_scores, method_names, OOV_value

def predict_hyper_scores(testing_pair_loader, L1_losss_B, device, word2emb, other_info, OOV_sim_zero = True):

    corpus_size = len(testing_pair_loader.dataset)
    pred_scores = [0] * corpus_size
    w_imp_list_1, proc_sent_list_1, w_imp_list_2, proc_sent_list_2 = other_info
    max_phrase_len = max( [len(x) for x in proc_sent_list_1 + proc_sent_list_2 ])
    
    emb_size = list(word2emb.values())[0].size
    sent_embs_SIF = [0] * corpus_size
    OOV_first_list = [0] * corpus_size
    OOV_second_list = [0] * corpus_size
    OOV_value = -100000000000
    for source, source_w, source_sent_emb, source_avg_word_emb, target, target_w, target_sent_emb, target_avg_word_emb, idx_batch in testing_pair_loader:
        #normalize w
        sys.stdout.write( str(idx_batch[0].item()) + ' ' )
        sys.stdout.flush()
        source_w = safe_normalization(source_w)
        target_w = safe_normalization(target_w)
        
        #Kmeans loss
        cos_sim_st, cos_sim_ts = compute_cosine_sim(source, target)
        
        uniform_s = torch.ones_like(source_w) / source_w.size(1)
        uniform_t = torch.ones_like(target_w) / target_w.size(1)
        cos_dist_st = 1 - cos_sim_st
        Wasserstein_trans = Sinkhorn.batch_sinkhorn_loss_weighted( cos_dist_st, uniform_s, uniform_t, epsilon=1, niter=100)
        Wasserstein_dist = torch.sum( Wasserstein_trans * cos_dist_st, dim = 2).sum(dim = 1)
        Wasserstein_trans_w = Sinkhorn.batch_sinkhorn_loss_weighted( cos_dist_st, source_w, target_w, epsilon=1, niter=100)
        Wasserstein_dist_w = torch.sum( Wasserstein_trans_w * cos_dist_st, dim = 2).sum(dim = 1)

        sim_avg_st, sim_w_avg_st = max_cosine_given_sim(cos_sim_st, target_w)
        sim_avg_ts, sim_w_avg_ts = max_cosine_given_sim(cos_sim_ts, source_w)
        sim_avg = (sim_avg_st + sim_avg_ts)/2
        sim_w_avg = (sim_w_avg_st + sim_w_avg_ts)/2

        sim_avg_max = torch.max( sim_avg_st, sim_avg_ts )
        sim_avg_min = torch.min( sim_avg_st, sim_avg_ts )
        sim_avg_diff = sim_avg_ts - sim_avg_st
        sim_avg_diff_inv = sim_avg_st - sim_avg_ts
        sim_w_avg_diff = sim_w_avg_ts - sim_w_avg_st
        sim_w_avg_diff_inv = sim_w_avg_st - sim_w_avg_ts

        dist_avg_st, dist_w_avg_st = lc_pred_dist(target, source, target_w, L1_losss_B, device)
        dist_avg_ts, dist_w_avg_ts = lc_pred_dist(source, target, source_w, L1_losss_B, device)
        dist_avg_diff = dist_avg_ts - dist_avg_st
        dist_avg_diff_inv = dist_avg_st - dist_avg_ts
        dist_w_avg_diff = dist_w_avg_ts - dist_w_avg_st
        dist_w_avg_diff_inv = dist_w_avg_st - dist_w_avg_ts

        cosine_sim = F.cosine_similarity(source_sent_emb, target_sent_emb, dim = 1)
        cosine_sim_word = F.cosine_similarity(source_avg_word_emb, target_avg_word_emb, dim = 1)

        s_w_emb_avg_tensor = torch.empty(sim_avg.size(0), 1, emb_size, device = device)
        t_w_emb_avg_tensor = torch.empty(sim_avg.size(0), 1, emb_size, device = device)
        s_w_uni_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, device = device)
        t_w_uni_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, device = device)
        s_w_embs_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, emb_size, device = device)
        t_w_embs_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, emb_size, device = device)
        baseline_score_list = [0]* sim_avg.size(0)
        for i in range(sim_avg.size(0)):
        #coeff_mat should have dimension (n_batch,n_set,n_basis)
            idx = idx_batch[i]
            source_proc_sent = proc_sent_list_1[idx]
            target_proc_sent = proc_sent_list_2[idx]
            source_w_imp_list = w_imp_list_1[idx]
            target_w_imp_list = w_imp_list_2[idx]
            w_embs_source = get_w_emb(source_proc_sent, word2emb, emb_size)
            w_embs_target = get_w_emb(target_proc_sent, word2emb, emb_size)
            w_prob_source = get_w_prob(source_proc_sent, word2emb, word_d2_idx_freq = None)
            w_prob_target = get_w_prob(target_proc_sent, word2emb, word_d2_idx_freq = None)
            
            store_w_embs_prob_to_tensors(s_w_uni_tensor, s_w_embs_tensor, i, w_embs_source, w_prob_source, device)
            store_w_embs_prob_to_tensors(t_w_uni_tensor, t_w_embs_tensor, i, w_embs_target, w_prob_target, device)

            s_sent_emb = weighted_sum_emb_list(w_embs_source)
            t_sent_emb = weighted_sum_emb_list(w_embs_target)
            score_baseline = safe_cosine_sim(s_sent_emb, t_sent_emb)
            
            s_w_emb_avg_tensor[i,0,:] = torch.tensor(s_sent_emb, device = device)
            t_w_emb_avg_tensor[i,0,:] = torch.tensor(t_sent_emb, device = device)

            OOV_first, OOV_second = check_OOV(s_sent_emb, t_sent_emb)

            OOV_first_list[idx] = OOV_first
            OOV_second_list[idx] = OOV_second
            
            baseline_score_list[i] = score_baseline
            if OOV_sim_zero and (OOV_first == 1 or OOV_second == 1):
                pred_scores[idx] = [OOV_value] * 21
            else:
                pred_scores[idx] = [-Wasserstein_dist[i].item(), -Wasserstein_dist_w[i].item(), sim_avg[i].item(), sim_w_avg[i].item(), cosine_sim[i].item(), cosine_sim_word[i].item(), score_baseline, sim_avg_max[i].item(), sim_avg_min[i].item(), sim_avg_diff[i].item(), sim_avg_diff[i].item() * sim_avg[i].item(), sim_avg_diff_inv[i].item(), sim_w_avg_diff[i].item(), sim_w_avg_diff_inv[i].item(), sim_avg_diff_inv[i].item()* sim_avg[i].item(), sim_w_avg_diff[i].item() * sim_avg[i].item(), dist_avg_diff[i].item(), dist_avg_diff_inv[i].item(), dist_w_avg_diff[i].item(), dist_w_avg_diff_inv[i].item(), dist_avg_diff_inv[i].item() + dist_w_avg_diff[i].item()]
        
        s_w_embs_tensor = s_w_embs_tensor /(0.000000000001 + s_w_embs_tensor.norm(dim = 2, keepdim=True) )
        t_w_embs_tensor = t_w_embs_tensor /(0.000000000001 + t_w_embs_tensor.norm(dim = 2, keepdim=True) )
        s_w_emb_avg_tensor = s_w_emb_avg_tensor / (0.000000000001 + s_w_emb_avg_tensor.norm(dim = 2, keepdim=True) )
        t_w_emb_avg_tensor = t_w_emb_avg_tensor / (0.000000000001 + t_w_emb_avg_tensor.norm(dim = 2, keepdim=True) )
        sim_avg_st, dist_avg_st = reconstruct_sent_emb(source, t_w_emb_avg_tensor, L1_losss_B, device)
        sim_avg_ts, dist_avg_ts = reconstruct_sent_emb(target, s_w_emb_avg_tensor, L1_losss_B, device)
        sim_w_avg_st, dist_w_avg_st = reconstruct_sent_emb(source, t_w_embs_tensor, L1_losss_B, device, t_w_uni_tensor)
        sim_w_avg_ts, dist_w_avg_ts = reconstruct_sent_emb(target, s_w_embs_tensor, L1_losss_B, device, s_w_uni_tensor)

        for i, idx in enumerate(idx_batch):
            if OOV_sim_zero and (OOV_first_list[idx] == 1 or OOV_second_list[idx] == 1):
                pred_scores[idx] = [OOV_value] * 9  + pred_scores[idx]
            else:
                #pred_scores[idx] = [sim_avg_st[i].item() - sim_avg_ts[i].item(), sim_avg_ts[i].item() - sim_avg_st[i].item(), dist_avg_st[i].item() - dist_avg_ts[i].item(), dist_avg_ts[i].item() - dist_avg_st[i].item()] + pred_scores[idx]
                best_diff = dist_w_avg_ts[i].item() - dist_w_avg_st[i].item()
                pred_scores[idx] = [sim_avg_st[i].item() - sim_avg_ts[i].item(), dist_avg_ts[i].item() - dist_avg_st[i].item(), sim_w_avg_st[i].item() - sim_w_avg_ts[i].item(), dist_w_avg_ts[i].item() - dist_w_avg_st[i].item(), best_diff / (0.001+ Wasserstein_dist[i].item()), best_diff / (0.001+ Wasserstein_dist_w[i].item()), best_diff * sim_avg[i].item(),  best_diff * cosine_sim_word[i].item(), best_diff * baseline_score_list[i]  ] + pred_scores[idx]

    print("Total OOV first: ", np.sum(OOV_first_list), ", Total OOV second:", np.sum(OOV_second_list))
    #method_names = ["kmeans_sim_st_ts_diff", "kmeans_sim_ts_st_diff", "SC_dist_st_ts_diff", "SC_dist_ts_st_diff", "Wasserstein", "Wasserstein_w", "kmeans", "kmeans_w", "en_sent_emb", "avg_en_word_emb", "baseline", "kmeans_max", "kmeans_min", "kmeans_diff", "kmeans_diff_avg", "kmeans_diff_inv", "kmeans_diff_w", "kmeans_diff_w_inv", "kmeans_diff_inv_avg", "kmeans_diff_w_avg", "SC_diff", "SC_diff_inv", "SC_diff_w", "SC_diff_w_inv", "SC_diff_inv+SC_diff_w"] 
    method_names = ["kmeans_sim_st_ts_diff", "SC_dist_ts_st_diff", "kmeans_word_sim_st_ts_diff", "SC_word_dist_ts_st_diff", "best_diff_Wasserstein", "best_diff_Wasserstein_w", "best_diff_kmeans", "best_diff_avg_en_word_emb", "best_diff_baseline", "Wasserstein", "Wasserstein_w", "kmeans", "kmeans_w", "en_sent_emb", "avg_en_word_emb", "baseline", "kmeans_max", "kmeans_min", "kmeans_diff", "kmeans_diff_avg", "kmeans_diff_inv", "kmeans_diff_w", "kmeans_diff_w_inv", "kmeans_diff_inv_avg", "kmeans_diff_w_avg", "SC_diff", "SC_diff_inv", "SC_diff_w", "SC_diff_w_inv", "SC_diff_inv+SC_diff_w"] 

    return pred_scores, method_names, OOV_value

#def predict_sim_scores(testing_pair_loader, L1_losss_B, device, word2emb, other_info, word_d2_idx_freq, OOV_sim_zero = True):
def predict_sim_scores(testing_pair_loader, L1_losss_B, device, word2emb, other_info, word_d2_idx_freq, OOV_sim_zero = True, compute_WMD = False, pc_mode = 'self', path_to_pc = ''):
    
#    def weighted_sum_emb_list_sq(w_embs, w_imp):
#        emb_size = w_embs[0].size
#        sent_emb = np.zeros(emb_size)
#        for i, emb in enumerate(w_embs):
#            if w_imp is None:
#                sent_emb += emb
#            else:
#                sent_emb += emb * w_imp[i] * w_imp[i]
#        return sent_emb
#
#    def weighted_sum_emb_list(w_embs, w_imp=None, alpha = -1):
#        emb_size = w_embs[0].size
#        sent_emb = np.zeros(emb_size)
#        for i, emb in enumerate(w_embs):
#            if w_imp is None:
#                sent_emb += emb
#            else:
#                #print(emb)
#                #print(w_imp)
#                if alpha == -1:
#                    sent_emb += emb * w_imp[i]
#                else:
#                    sent_emb += emb * (alpha/(w_imp[i]+alpha))
#        return sent_emb
#    
#    def weighted_sum_emb_list_2(w_embs, w_imp, w_prob, alpha):
#        emb_size = w_embs[0].size
#        sent_emb = np.zeros(emb_size)
#        for i, emb in enumerate(w_embs):
#            if w_imp is None:
#                sent_emb += emb
#            else:
#                sent_emb += emb * w_imp[i] * (alpha/(w_prob[i]+alpha))
#        return sent_emb
#
     

    def freq_weighting(w_embs, w_prob, w_imp, alpha):
        w_imp_freq = alpha/(w_prob+alpha)
        sent_emb_freq = weighted_sum_emb_list(w_embs, w_imp_freq)
        sent_emb_freq_w_sim = weighted_sum_emb_list(w_embs, w_imp_freq * w_imp)
        return sent_emb_freq, sent_emb_freq_w_sim

    def sentence_emb_from_sum(w_embs, w_imp_list, w_prob):
        sent_emb = weighted_sum_emb_list(w_embs)
        alpha = 0.001
        sent_emb_freq_3, sent_emb_freq_3_w_sim = freq_weighting(w_embs, w_prob, w_imp_list[0], alpha)
        #sent_emb_freq_3 = weighted_sum_emb_list(w_embs, w_prob, alpha)
        #sent_emb_freq_3_w_sim = weighted_sum_emb_list_2(w_embs, w_imp_list[0], w_prob, alpha)
        alpha = 0.0001
        sent_emb_freq_4, sent_emb_freq_4_w_sim = freq_weighting(w_embs, w_prob, w_imp_list[0], alpha)
        #sent_emb_freq_4 = weighted_sum_emb_list(w_embs, w_prob, alpha)
        #sent_emb_freq_4_w_sim = weighted_sum_emb_list_2(w_embs, w_imp_list[0], w_prob, alpha)
        alpha = 0.00001
        sent_emb_freq_5, sent_emb_freq_5_w_sim = freq_weighting(w_embs, w_prob, w_imp_list[0], alpha)
        #sent_emb_freq_5 = weighted_sum_emb_list(w_embs, w_prob, alpha)
        #sent_emb_freq_5_w_sim = weighted_sum_emb_list_2(w_embs, w_imp_list[0], w_prob, alpha)
        sent_emb_w_sim = weighted_sum_emb_list(w_embs, w_imp_list[0])
        #sent_emb_w_sim_sq = weighted_sum_emb_list_sq(w_embs, w_imp_list[0])
        sent_emb_w_sim_coeff = weighted_sum_emb_list(w_embs, w_imp_list[1])
        sent_emb_w_coeff = weighted_sum_emb_list(w_embs, w_imp_list[2])
        #return sent_emb, sent_emb_freq_3, sent_emb_freq_4, sent_emb_freq_5, sent_emb_freq_3_w_sim, sent_emb_freq_4_w_sim, sent_emb_freq_5_w_sim, sent_emb_w_sim, sent_emb_w_sim_sq, sent_emb_w_sim_coeff, sent_emb_w_coeff
        return sent_emb, sent_emb_freq_3, sent_emb_freq_4, sent_emb_freq_5, sent_emb_freq_3_w_sim, sent_emb_freq_4_w_sim, sent_emb_freq_5_w_sim, sent_emb_w_sim, sent_emb_w_sim_coeff, sent_emb_w_coeff

    
    def weighted_sent_avg(w_embs_source, source_w_imp_list, w_prob_source, w_embs_target, target_w_imp_list, w_prob_target):
        #s_sent_emb, s_sent_emb_freq_3, s_sent_emb_freq_4, s_sent_emb_freq_5, s_sent_emb_freq_3_w_sim, s_sent_emb_freq_4_w_sim, s_sent_emb_freq_5_w_sim, s_sent_emb_w_sim, s_sent_emb_w_sim_sq, s_sent_emb_w_sim_coeff, s_sent_emb_w_coeff = sentence_emb_from_sum(w_embs_source, source_w_imp_list, w_prob_source)
        #t_sent_emb, t_sent_emb_freq_3, t_sent_emb_freq_4, t_sent_emb_freq_5, t_sent_emb_freq_3_w_sim, t_sent_emb_freq_4_w_sim, t_sent_emb_freq_5_w_sim, t_sent_emb_w_sim, t_sent_emb_w_sim_sq, t_sent_emb_w_sim_coeff, t_sent_emb_w_coeff = sentence_emb_from_sum(w_embs_target, target_w_imp_list, w_prob_target)
        s_sent_emb, s_sent_emb_freq_3, s_sent_emb_freq_4, s_sent_emb_freq_5, s_sent_emb_freq_3_w_sim, s_sent_emb_freq_4_w_sim, s_sent_emb_freq_5_w_sim, s_sent_emb_w_sim, s_sent_emb_w_sim_coeff, s_sent_emb_w_coeff = sentence_emb_from_sum(w_embs_source, source_w_imp_list, w_prob_source)
        t_sent_emb, t_sent_emb_freq_3, t_sent_emb_freq_4, t_sent_emb_freq_5, t_sent_emb_freq_3_w_sim, t_sent_emb_freq_4_w_sim, t_sent_emb_freq_5_w_sim, t_sent_emb_w_sim, t_sent_emb_w_sim_coeff, t_sent_emb_w_coeff = sentence_emb_from_sum(w_embs_target, target_w_imp_list, w_prob_target)
        score_baseline = safe_cosine_sim(s_sent_emb, t_sent_emb)       
        score_baseline_freq_3 = safe_cosine_sim(s_sent_emb_freq_3, t_sent_emb_freq_3)
        score_baseline_freq_4 = safe_cosine_sim(s_sent_emb_freq_4, t_sent_emb_freq_4)
        score_baseline_freq_5 = safe_cosine_sim(s_sent_emb_freq_5, t_sent_emb_freq_5)
        score_baseline_freq_3_w_sim = safe_cosine_sim(s_sent_emb_freq_3_w_sim, t_sent_emb_freq_3_w_sim)
        score_baseline_freq_4_w_sim = safe_cosine_sim(s_sent_emb_freq_4_w_sim, t_sent_emb_freq_4_w_sim)
        score_baseline_freq_5_w_sim = safe_cosine_sim(s_sent_emb_freq_5_w_sim, t_sent_emb_freq_5_w_sim)
        score_w_sim = safe_cosine_sim(s_sent_emb_w_sim, t_sent_emb_w_sim)
        #score_w_sim_sq = safe_cosine_sim(s_sent_emb_w_sim_sq, t_sent_emb_w_sim_sq)
        score_w_sim_coeff = safe_cosine_sim(s_sent_emb_w_sim_coeff, t_sent_emb_w_sim_coeff)
        score_w_coeff = safe_cosine_sim(s_sent_emb_w_coeff, t_sent_emb_w_coeff)
        #return [score_w_sim, score_w_sim_sq, score_w_sim_coeff, score_w_coeff, score_baseline_freq_3_w_sim, score_baseline_freq_4_w_sim, score_baseline_freq_5_w_sim, score_baseline_freq_3, score_baseline_freq_4, score_baseline_freq_5, score_baseline]
        scores_w_emb = [score_w_sim, score_w_sim_coeff, score_w_coeff, score_baseline_freq_3_w_sim, score_baseline_freq_4_w_sim, score_baseline_freq_5_w_sim, score_baseline_freq_3, score_baseline_freq_4, score_baseline_freq_5, score_baseline]
        sent_emb_list = [ [s_sent_emb_freq_4, t_sent_emb_freq_4], [s_sent_emb_freq_4_w_sim, t_sent_emb_freq_4_w_sim] ]
        OOV_all_sent = check_OOV(s_sent_emb, t_sent_emb)
        return scores_w_emb, sent_emb_list, OOV_all_sent, t_sent_emb, t_sent_emb_freq_4_w_sim, t_sent_emb_w_sim
    
    def safe_2_norm(emb):
        emb_mag = np.linalg.norm(emb)
        if emb_mag > 0:
            emb_norm = emb / emb_mag
        else:
            emb_norm = emb
        return emb_norm

    def combine_topic_by_word(topics_i, emb_norm):
        cosine_sim = np.matmul(topics_i, emb_norm)
        cosine_sim_pos = np.maximum(0, cosine_sim)
        #cosine_sim_pos = cosine_sim_pos * cosine_sim_pos
        cosine_sim_pos_sum = sum(cosine_sim_pos)
        if cosine_sim_pos_sum > 0:
            cosine_sim_pos /= cosine_sim_pos_sum
        local_emb_i = np.matmul( cosine_sim_pos, topics_i)
        local_emb_i = safe_2_norm(local_emb_i)
        return local_emb_i

    def compute_local_embs(w_embs, topics_i, source_w):
        local_w_embs = []
        local_w_embs_wt = []
        topics_i_wt = source_w.reshape(-1,1) * topics_i
        local_weight = 0.5
        for emb in w_embs:
            #print(emb.shape)
            #print(topics_i.shape)
            emb_norm = safe_2_norm(emb)
            local_emb_i = combine_topic_by_word(topics_i, emb_norm)
            local_emb_i_wt = combine_topic_by_word(topics_i_wt, emb_norm)
            local_w_embs.append(local_emb_i * local_weight + emb * (1-local_weight))
            local_w_embs_wt.append(local_emb_i_wt * local_weight + emb * (1-local_weight))

        return local_w_embs, local_w_embs_wt

    #def max_cosine_sim(target, source, target_w):
    #    cosine_sim_s_to_t, nn_source_idx = nsd_loss.estimate_coeff_mat_batch_max_cos(target, source)
    #    #cosine_sim should have dimension (n_batch,n_set)
    #    sim_avg_st = cosine_sim_s_to_t.mean(dim = 1)
    #    sim_w_avg_st = weighted_average(cosine_sim_s_to_t, target_w)
    #    return sim_avg_st, sim_w_avg_st

    corpus_size = len(testing_pair_loader.dataset)
    pred_scores = [0] * corpus_size
    w_imp_list_1, proc_sent_list_1, w_imp_list_2, proc_sent_list_2 = other_info
    
    emb_size = list(word2emb.values())[0].size
    sent_embs_SIF = [0] * corpus_size
    OOV_first_list = [0] * corpus_size
    OOV_second_list = [0] * corpus_size
    #OOV_value = -100000000000 #TODO current Turney and SemEval results in paper are based on this line
    OOV_value = 0 #TODO BiRD evaluation is based on this option

    max_phrase_len = max( [len(x) for x in proc_sent_list_1 + proc_sent_list_2 ])
    #output_idx_list = []
    #for source, source_w, source_sent_emb, source_avg_word_emb, source_w_imp_list, source_proc_sent, target, target_w, target_sent_emb, target_avg_word_emb, target_w_imp_list, target_proc_sent in testing_pair_loader:
    for source, source_w, source_sent_emb, source_avg_word_emb, target, target_w, target_sent_emb, target_avg_word_emb, idx_batch in testing_pair_loader:
        #source_w_imp_list = source_w_imp_list.tolist()
        #target_w_imp_list= target_w_imp_list.tolist()
        #print(source_w_imp_list)
        #normalize w
        torch.cuda.synchronize()
        sys.stdout.write( str(idx_batch[0].item()) + ' ' )
        sys.stdout.flush()
        source_w = safe_normalization(source_w)
        target_w = safe_normalization(target_w)
        
        #Kmeans loss
        cos_sim_st, cos_sim_ts = compute_cosine_sim(source, target)
        
        uniform_s = torch.ones_like(source_w) / source_w.size(1)
        uniform_t = torch.ones_like(target_w) / target_w.size(1)
        cos_dist_st = 1 - cos_sim_st
        #Wasserstein_trans = Sinkhorn.batch_sinkhorn_loss_weighted( cos_dist_st, uniform_s, uniform_t, epsilon=1, niter=100)
        #Wasserstein_dist = torch.sum( Wasserstein_trans * cos_dist_st, dim = 2).sum(dim = 1)
        #Wasserstein_trans = Sinkhorn.batch_sinkhorn_loss_weighted( cos_dist_st, uniform_s, uniform_t, epsilon=2, niter=100)
        #Wasserstein_dist_e2 = torch.sum( Wasserstein_trans * cos_dist_st, dim = 2).sum(dim = 1)
        #Wasserstein_trans = Sinkhorn.batch_sinkhorn_loss_weighted( cos_dist_st, uniform_s, uniform_t, epsilon=0.5, niter=100)
        #Wasserstein_dist_e05 = torch.sum( Wasserstein_trans * cos_dist_st, dim = 2).sum(dim = 1)
        
        #Wasserstein_trans_w = Sinkhorn.batch_sinkhorn_loss_weighted( cos_dist_st, source_w, target_w, epsilon=1, niter=100)
        #Wasserstein_dist_w = torch.sum( Wasserstein_trans_w * cos_dist_st, dim = 2).sum(dim = 1)

        sim_avg_st, sim_w_avg_st = max_cosine_given_sim(cos_sim_st, target_w)
        sim_avg_ts, sim_w_avg_ts = max_cosine_given_sim(cos_sim_ts, source_w)
        #sim_avg_st, sim_w_avg_st = max_cosine_sim(target, source, target_w)
        #sim_avg_ts, sim_w_avg_ts = max_cosine_sim(source, target, source_w)
        sim_avg = (sim_avg_st + sim_avg_ts)/2
        sim_w_avg = (sim_w_avg_st + sim_w_avg_ts)/2

        dist_avg_st, dist_w_avg_st = lc_pred_dist(target, source, target_w, L1_losss_B, device)
        dist_avg_ts, dist_w_avg_ts = lc_pred_dist(source, target, source_w, L1_losss_B, device)
        dist_avg = (dist_avg_st + dist_avg_ts)/2
        dist_w_avg = (dist_w_avg_st + dist_w_avg_ts)/2

        cosine_sim = F.cosine_similarity(source_sent_emb, target_sent_emb, dim = 1)
        cosine_sim_word = F.cosine_similarity(source_avg_word_emb, target_avg_word_emb, dim = 1)

        t_w_emb_avg_tensor = torch.empty(sim_avg.size(0), 1, emb_size, device = device)
        t_w_emb_freq_4_w_sim_tensor = torch.empty(sim_avg.size(0), 1, emb_size, device = device)
        t_w_emb_w_sim_tensor = torch.empty(sim_avg.size(0), 1, emb_size, device = device)

        #if compute_WMD:
        #    s_w_uni_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, device = device)
        #    t_w_uni_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, device = device)
        #    s_w_weights_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, device = device)
        #    t_w_weights_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, device = device)
        #    s_w_embs_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, emb_size, device = device)
        #    t_w_embs_tensor = torch.zeros(sim_avg.size(0), max_phrase_len, emb_size, device = device)
        uniform_s_np = uniform_s.cpu().numpy()
        uniform_t_np = uniform_t.cpu().numpy()
        source_w_np = source_w.cpu().numpy()
        target_w_np = target_w.cpu().numpy()
        cos_dist_st_np = cos_dist_st.cpu().numpy()
        for i in range(sim_avg.size(0)):
        #coeff_mat should have dimension (n_batch,n_set,n_basis)
            idx = idx_batch[i]
            
            uniform_s_i = uniform_s_np[i,:]
            uniform_t_i = uniform_t_np[i,:]
            source_w_i = source_w_np[i,:]
            target_w_i = target_w_np[i,:]
            cos_dist_st_i = cos_dist_st_np[i,:,:]
            OT_uni = ot.emd2(uniform_s_i, uniform_t_i, cos_dist_st_i)
            #source_w_i = source_w_i / ( np.sum(source_w_i) + 0.000000000001)
            #target_w_i = target_w_i / ( np.sum(target_w_i) + 0.000000000001)
            OT_w = ot.emd2(source_w_i, target_w_i, cos_dist_st_i)
            
            source_proc_sent = proc_sent_list_1[idx]
            target_proc_sent = proc_sent_list_2[idx]
            source_w_imp_list = w_imp_list_1[idx]
            target_w_imp_list = w_imp_list_2[idx]
            #print(source_proc_sent)
            #print(source_w_imp_list)
            w_embs_source = get_w_emb(source_proc_sent, word2emb, emb_size)
            w_embs_target = get_w_emb(target_proc_sent, word2emb, emb_size)
            w_prob_source = get_w_prob(source_proc_sent, word2emb, word_d2_idx_freq)
            w_prob_target = get_w_prob(target_proc_sent, word2emb, word_d2_idx_freq)
            if compute_WMD:
                w_uni_source = get_w_prob(source_proc_sent, word2emb, word_d2_idx_freq = None)
                w_uni_target = get_w_prob(target_proc_sent, word2emb, word_d2_idx_freq = None)
                #store_w_embs_prob_to_tensors(s_w_uni_tensor, s_w_embs_tensor, i, w_embs_source, w_uni_source, device)
                #store_w_embs_prob_to_tensors(t_w_uni_tensor, t_w_embs_tensor, i, w_embs_target, w_uni_target, device)
                
                alpha = 0.0001
                w_weights_raw_source = convert_prob_to_weights(source_proc_sent, word_d2_idx_freq, alpha)
                w_weights_raw_target = convert_prob_to_weights(target_proc_sent, word_d2_idx_freq, alpha)
                w_weights_source = get_w_prob(source_proc_sent, word2emb, word_d2_idx_freq = w_weights_raw_source)
                w_weights_target = get_w_prob(target_proc_sent, word2emb, word_d2_idx_freq = w_weights_raw_target)

                w_weights_sim_raw_source = (source_w_imp_list[0] * w_weights_raw_source).tolist()
                w_weights_sim_raw_target = (target_w_imp_list[0] * w_weights_raw_target).tolist()
                w_weights_sim_source = get_w_prob(source_proc_sent, word2emb, word_d2_idx_freq = w_weights_sim_raw_source)
                w_weights_sim_target = get_w_prob(target_proc_sent, word2emb, word_d2_idx_freq = w_weights_sim_raw_target)

                w_sim_raw_source = source_w_imp_list[0].tolist()
                w_sim_raw_target = target_w_imp_list[0].tolist()
                w_sim_source = get_w_prob(source_proc_sent, word2emb, word_d2_idx_freq = w_sim_raw_source)
                w_sim_target = get_w_prob(target_proc_sent, word2emb, word_d2_idx_freq = w_sim_raw_target)
                #w_num = len(source_proc_sent)
                #s_w_weights_tensor[i,-w_num:] = torch.tensor(w_weights_source, device = device)
                #w_num = len(target_proc_sent)
                #t_w_weights_tensor[i,-w_num:] = torch.tensor(w_weights_target, device = device)
                
                w_embs_source_norm = w_embs_source / (0.000000000001 + np.linalg.norm(w_embs_source, axis = 1, keepdims=True))
                w_embs_target_norm = w_embs_target / (0.000000000001 + np.linalg.norm(w_embs_target, axis = 1, keepdims=True))
                M = 1 - np.matmul( w_embs_source_norm,  np.transpose(w_embs_target_norm) )
                Wd = ot.emd2(w_uni_source, w_uni_target, M)
                #Wd_reg = ot.sinkhorn2(w_uni_source, w_uni_target, M, 1)
                Wd_w = ot.emd2(w_weights_source, w_weights_target, M)
                Wd_w_sim = ot.emd2(w_weights_sim_source, w_weights_sim_target, M)
                Wd_sim = ot.emd2(w_sim_source, w_sim_target, M)
            else:
                Wd = np.random.rand(1)[0]
                #Wd_reg = np.random.rand(1)[0]
                Wd_w = np.random.rand(1)[0]
                Wd_w_sim = np.random.rand(1)[0]
                Wd_sim = np.random.rand(1)[0]

            #print(w_embs_source)
            scores_w_emb_org, sent_emb_list_org, OOV_all_sent, t_sent_emb, t_sent_emb_freq_4_w_sim, t_sent_emb_w_sim = weighted_sent_avg(w_embs_source, source_w_imp_list, w_prob_source, w_embs_target, target_w_imp_list, w_prob_target)
            t_w_emb_avg_tensor[i,0,:] = torch.tensor(t_sent_emb, device = device)
            t_w_emb_freq_4_w_sim_tensor[i,0,:] = torch.tensor(t_sent_emb_freq_4_w_sim, device = device)
            t_w_emb_w_sim_tensor[i,0,:] = torch.tensor(t_sent_emb_w_sim, device = device)
            #scores_str = ["w_imp_sim", "w_imp_sim_sq", "w_imp_sim_coeff", "w_imp_coeff",  "w_imp_sim_freq_3", "w_imp_sim_freq_4", "w_imp_sim_freq_5", "baseline_freq3", "baseline_freq4", "baseline_freq5", "baseline"]
            scores_str = ["w_imp_sim", "w_imp_sim_coeff", "w_imp_coeff",  "w_imp_sim_freq_3", "w_imp_sim_freq_4", "w_imp_sim_freq_5", "baseline_freq3", "baseline_freq4", "baseline_freq5", "baseline"]
            local_emb_source, local_emb_wt_source = compute_local_embs(w_embs_source, source[i,:,:].cpu().numpy(), source_w[i,:].cpu().numpy())
            local_emb_target, local_emb_wt_target = compute_local_embs(w_embs_target, target[i,:,:].cpu().numpy(), target_w[i,:].cpu().numpy())
            #scores_w_emb_local = weighted_sent_avg(local_emb_source, source_w_imp_list, w_prob_source, local_emb_target, target_w_imp_list, w_prob_target)
            out_fields = weighted_sent_avg(local_emb_wt_source, source_w_imp_list, w_prob_source, local_emb_wt_target, target_w_imp_list, w_prob_target)
            scores_w_emb_local_wt, sent_emb_list_local_wt = out_fields[:2]
            #scores_str_local = [x + '_local' for x in scores_str]            
            scores_str_local_wt = [x + '_local_wt' for x in scores_str]            
            sent_embs_SIF[idx] = sent_emb_list_org + sent_emb_list_local_wt + [  [source_sent_emb[i,:].cpu().numpy(), target_sent_emb[i,:].cpu().numpy()] ] + [ [source_avg_word_emb[i,:].cpu().numpy(), target_avg_word_emb[i,:].cpu().numpy()] ]
            OOV_first_list[idx] = OOV_all_sent[0]
            OOV_second_list[idx] = OOV_all_sent[1]

            #pred_scores.append([sim_avg[i].item(), sim_w_avg[i].item(), dist_avg[i].item(), dist_w_avg[i].item(), cosine_sim[i].item(), cosine_sim_word[i].item()] + scores_w_emb_org + scores_w_emb_local + scores_w_emb_local_wt)
            if OOV_sim_zero and (OOV_all_sent[0] == 1 or OOV_all_sent[1] == 1):
                pred_scores[idx] = [OOV_value] * (12 + len(scores_str)*2)
            else:
                #pred_scores[idx] = [-Wd, -Wd_w, -Wasserstein_dist[i].item(), -Wasserstein_dist_w[i].item(), sim_avg[i].item(), sim_w_avg[i].item(), -dist_avg[i].item(), -dist_w_avg[i].item(), cosine_sim[i].item(), cosine_sim_word[i].item()] + scores_w_emb_org + scores_w_emb_local_wt
                pred_scores[idx] = [-Wd, -Wd_w, -Wd_w_sim, -Wd_sim, -OT_uni, -OT_w, sim_avg[i].item(), sim_w_avg[i].item(), -dist_avg[i].item(), -dist_w_avg[i].item(), cosine_sim[i].item(), cosine_sim_word[i].item()] + scores_w_emb_org + scores_w_emb_local_wt
        

        #if compute_WMD:
        #    s_w_embs_tensor = s_w_embs_tensor / (0.000000000001 + s_w_embs_tensor.norm(dim = 2, keepdim=True) )
        #    t_w_embs_tensor = t_w_embs_tensor / (0.000000000001 + t_w_embs_tensor.norm(dim = 2, keepdim=True) )
        #    cos_w_sim_st, cos_w_sim_ts = compute_cosine_sim(s_w_embs_tensor, t_w_embs_tensor)
        #    cos_w_dist_st = 1 - cos_w_sim_st
        #    WMD_trans = Sinkhorn.batch_sinkhorn_loss_weighted( cos_w_dist_st, s_w_uni_tensor, t_w_uni_tensor, epsilon=1, niter=300)
        #    WMD_dist = torch.sum( WMD_trans * cos_w_dist_st, dim = 2).sum(dim = 1)
        #    WMD_w_trans = Sinkhorn.batch_sinkhorn_loss_weighted( cos_w_dist_st, s_w_weights_tensor, t_w_weights_tensor, epsilon=1, niter=300)
        #    WMD_w_dist = torch.sum( WMD_w_trans * cos_w_dist_st, dim = 2).sum(dim = 1)
        #else:
        #    WMD_dist = torch.randn(sim_avg.size(0))
        #    WMD_w_dist = torch.randn(sim_avg.size(0))
            
        #t_w_emb_avg_tensor = t_w_emb_avg_tensor / (0.000000000001 + t_w_emb_avg_tensor.norm(dim = 2, keepdim=True) )
        #t_w_emb_freq_4_w_sim_tensor = t_w_emb_freq_4_w_sim_tensor / (0.000000000001 + t_w_emb_freq_4_w_sim_tensor.norm(dim = 2, keepdim=True) )
        #t_w_emb_w_sim_tensor = t_w_emb_w_sim_tensor / (0.000000000001 + t_w_emb_w_sim_tensor.norm(dim = 2, keepdim=True) )
        #sim_avg_st, dist_avg_st = reconstruct_sent_emb(source, t_w_emb_avg_tensor, L1_losss_B, device)
        #sim_freq_4_w_sim_st, dist_freq_4_w_sim_st = reconstruct_sent_emb(source, t_w_emb_freq_4_w_sim_tensor, L1_losss_B, device)
        #sim_w_sim_st, dist_w_sim_st = reconstruct_sent_emb(source, t_w_emb_w_sim_tensor, L1_losss_B, device)
        #for i, idx in enumerate(idx_batch):
        #    if OOV_sim_zero and (OOV_first_list[idx] == 1 or OOV_second_list[idx] == 1):
        #        pred_scores[idx] = [OOV_value] * 6  + pred_scores[idx]
        #    else:
        #        #pred_scores[idx] = [ -WMD_dist[i].item(), -WMD_w_dist[i].item(), sim_avg_st[i].item(), -dist_avg_st[i].item(), sim_w_sim_st[i].item(), -dist_w_sim_st[i].item(), sim_freq_4_w_sim_st[i].item(), -dist_freq_4_w_sim_st[i].item()] + pred_scores[idx]
        #        pred_scores[idx] = [ sim_avg_st[i].item(), -dist_avg_st[i].item(), sim_w_sim_st[i].item(), -dist_w_sim_st[i].item(), sim_freq_4_w_sim_st[i].item(), -dist_freq_4_w_sim_st[i].item()] + pred_scores[idx]
                
            #output_idx_list.append(idx.item())
    #method_names = ["kmeans", "kmeans_w", "SC_rmsprop", "SC_rmsprop_w", "en_sent_emb", "avg_en_word_emb"] + scores_str + scores_str_local + scores_str_local_wt
    print("Total OOV first: ", np.sum(OOV_first_list), ", Total OOV second:", np.sum(OOV_second_list))
    #method_names = ["WMD or rnd", "WMD weighted or rnd", "sim_avg_st", "dist_avg_s", "sim_w_sim_st", "dist_w_sim_s", "sim_freq_4_w_sim_st", "dist_freq_4_w_sim_s", "WMD exact or rnd", "WMD weighted exact or rnd", "WMD reg or rnd", "Wasserstein", "Wasserstein_e2", "Wasserstein_e05", "Wasserstein_w", "kmeans", "kmeans_w", "SC_rmsprop", "SC_rmsprop_w", "en_sent_emb", "avg_en_word_emb"] + scores_str + scores_str_local_wt
    #method_names = ["WMD or rnd", "WMD weighted or rnd", "sim_avg_st", "dist_avg_s", "sim_w_sim_st", "dist_w_sim_s", "sim_freq_4_w_sim_st", "dist_freq_4_w_sim_s", "WMD exact or rnd", "WMD weighted exact or rnd", "WMD reg or rnd", "Wasserstein", "Wasserstein_e2", "Wasserstein_e05", "Wasserstein_w", "kmeans", "kmeans_w", "SC_rmsprop", "SC_rmsprop_w", "en_sent_emb", "avg_en_word_emb"] + scores_str + scores_str_local_wt
    method_names = ["WMD exact or rnd", "WMD weighted freq4 exact or rnd", "WMD weighted freq4 sim exact or rnd", "WMD weighted sim exact or rnd",  "Wasserstein", "Wasserstein_w", "kmeans", "kmeans_w", "SC_rmsprop", "SC_rmsprop_w", "en_sent_emb", "avg_en_word_emb"] + scores_str + scores_str_local_wt
    if pc_mode == 'none':
        return pred_scores, method_names

    #sent_embs_freq, sent_embs_freq_w_sim, sent_embs_freq_local_wt, sent_embs_freq_w_sim_local_wt = 
    pred_scores_transpose = list(zip(*pred_scores))
    #print(len(pred_scores_transpose))
    pc_method_names = ["baseline_freq_4_pc1", "w_imp_sim_freq_4_pc1", "baseline_freq_4_local_wt_pc1", "w_imp_sim_freq_4_local_wt_pc1", "en_sent_emb_pc1", "avg_en_word_emb_pc1"]
    method_names += pc_method_names

    for i, sent_embs in enumerate(zip(*sent_embs_SIF)):
        flatten_sent_embs = np.concatenate([emb.reshape(1,-1) for sublist in sent_embs for emb in sublist], axis = 0)
        #print(flatten_sent_embs.shape)
        sent_embs_pc = SIF.remove_pc(flatten_sent_embs, 1, pc_mode, path_to_pc+'_'+pc_method_names[i])
        #print(sent_embs_pc.shape)
        score = []
        for i in range( int(len(sent_embs_pc)/2) ):
            sent_emb_1 = sent_embs_pc[2*i]
            sent_emb_2 = sent_embs_pc[2*i+1]
            sim = safe_cosine_sim(sent_emb_1, sent_emb_2)
            score.append( sim )
        #score_sorted = [score[i] for i in output_idx_list]
        #print(len(score_sorted))
        #pred_scores_transpose += [score_sorted]
        pred_scores_transpose += [score]
    #print(len(pred_scores_transpose))
    pred_scores = [list(x) for x in zip(*pred_scores_transpose)]
    #print(len(pred_scores))
    return pred_scores, method_names
