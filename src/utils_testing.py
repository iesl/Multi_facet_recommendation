import torch
import nsd_loss
import numpy as np
import gc


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
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout applied to the output layer (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.4,
                        help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropout to remove words from embedding layer (0 = no dropout)')

    ###decoder
    parser.add_argument('--de_model', type=str, default='LSTM',
                        help='type of decoder model (LSTM)')
    parser.add_argument('--nhidlast2', type=int, default=-1,
                        help='hidden embedding size of the second LSTM')
    parser.add_argument('--n_basis', type=int, default=10,
                        help='number of basis we want to predict')

def predict_batch(feature, parallel_encoder, parallel_decoder, word_norm_emb, n_basis, top_k):
    output_emb, hidden, output_emb_last = parallel_encoder(feature.t())
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

    return basis_norm_pred, coeff_order, coeff_sum, top_value, top_index

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

def dump_prediction_to_json(feature, basis_norm_pred, idx2word_freq, coeff_order, coeff_sum, top_value, top_index, basis_json, org_sent_list):
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
        output_dict['proc_sent'] = ' '.join(proc_sent)
        output_dict['topics'] = topic_list
        basis_json.append(output_dict)
        #basis_json.append([current_idx, org_sent, ' '.join(proc_sent)])

def output_sent_basis(dataloader, org_sent_list, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, n_basis):
    basis_json = []
    top_k = 5
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            feature, target = sample_batched

            basis_norm_pred, coeff_order, coeff_sum, top_value, top_index = predict_batch(feature, parallel_encoder, parallel_decoder, word_norm_emb, n_basis, top_k)
            dump_prediction_to_json(feature, basis_norm_pred, idx2word_freq, coeff_order, coeff_sum, top_value, top_index, basis_json, org_sent_list)
    return basis_json

def visualize_topics_val(dataloader, parallel_encoder, parallel_decoder, word_norm_emb, idx2word_freq, outf, n_basis, max_batch_num):
    #topics_num = 0
    top_k = 5
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            feature, target = sample_batched

            basis_norm_pred, coeff_order, coeff_sum, top_value, top_index = predict_batch(feature, parallel_encoder, parallel_decoder, word_norm_emb, n_basis, top_k)
            print_basis_text(feature, idx2word_freq, coeff_order, coeff_sum, top_value, top_index, i_batch, outf)

            if i_batch >= max_batch_num:
                break
