import argparse
import os, sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
#import torch.utils.data

import gc

import random
import model as model_code
import nsd_loss
#import coherency_eval

from utils import create_exp_dir, save_checkpoint, load_idx2word_freq, load_emb_file, load_corpus

parser = argparse.ArgumentParser(description='PyTorch Neural Set Decoder for Sentnece Embedding')

###path
parser.add_argument('--data', type=str, default='./data/processed/wackypedia/',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./models/',
                    help='model checkpoint to use')
parser.add_argument('--emb_file', type=str, default='target_emb.pt',
                    help='path to the file of a word embedding file')
parser.add_argument('--outf', type=str, default='gen_log/generated.txt',
                    help='output file for generated text')

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

###system
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', default=True, action='store_false',
                    help='use CUDA')
parser.add_argument('--single_gpu', default=False, action='store_true',
                    help='use single GPU')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
parser.add_argument('--max_batch_num', type=int, default=100, 
                    help='number of batches for evaluation')


args = parser.parse_args()

if args.emb_file == "target_emb.pt":
    args.emb_file =  os.path.join(args.checkpoint,"target_emb.pt")

if args.nhidlast2 < 0:
    args.nhidlast2 = args.emsize


# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

########################
print("Loading data")
########################

device = torch.device("cuda" if args.cuda else "cpu")

idx2word_freq, dataloader_train, dataloader_val, dataloader_val_shuffled = load_corpus(args.data, args.batch_size, args.batch_size, device)



########################
print("Loading Models")
########################

if len(args.emb_file) > 0:
    if args.emb_file[-3:] == '.pt':
        word_emb = torch.load( args.emb_file )
        output_emb_size = word_emb.size(1)
    else:
        word_emb, output_emb_size = load_emb_file(args.emb_file,device,idx2word_freq)
else:
    output_emb_size = args.emsize

ntokens = len(idx2word_freq)
if args.en_model == "LSTM":
    external_emb = torch.tensor([0.])
    encoder = model_code.RNNModel_simple(args.en_model, ntokens, args.emsize, args.nhid, args.nlayers,
                   args.dropout, args.dropouti, args.dropoute, external_emb)

    decoder = model_code.RNNModel_decoder(args.de_model, args.nhid * 2, args.nhidlast2, output_emb_size, 1, args.n_basis, linear_mapping_dim = 0, dropoutp= 0.5)

encoder.load_state_dict(torch.load(os.path.join(args.checkpoint, 'encoder.pt')))
decoder.load_state_dict(torch.load(os.path.join(args.checkpoint, 'decoder.pt')))

if len(args.emb_file) == 0:
    word_emb = encoder.encoder.weight.detach()

word_norm_emb = word_emb / (0.000000000001 + word_emb.norm(dim = 1, keepdim=True) )
word_norm_emb[0,:] = 0

if args.cuda:
    if args.single_gpu:
        parallel_encoder = encoder.cuda()
        parallel_decoder = decoder.cuda()
    else:
        parallel_encoder = nn.DataParallel(encoder, dim=1).cuda()
        parallel_decoder = decoder.cuda()
else:
    parallel_encoder = encoder
    parallel_decoder = decoder

encoder.eval()
decoder.eval()



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

def visualize_topics_val(dataloader, outf):
    topics_num = 0
    top_k = 5
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            feature, target = sample_batched
            
            output_emb, hidden, output_emb_last = parallel_encoder(feature.t())
            basis_pred, coeff_pred = nsd_loss.predict_basis(decoder, args.n_basis, output_emb_last, predict_coeff_sum = True )

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

            feature_text = convert_feature_to_text(feature, idx2word_freq)
            for i_sent in range(len(feature_text)):
                outf.write('{}th sent: '.format(i_sent)+' '.join(feature_text[i_sent])+'\n')

                for j in range(args.n_basis):
                    org_ind = coeff_order[i_sent, j]
                    outf.write(str(j)+', org '+str(org_ind)+', '+str( coeff_sum[i_sent,org_ind,0] )+' - '+str( coeff_sum[i_sent,org_ind,1] )+': ')

                    for k in range(top_k):
                        word_nn = idx2word_freq[top_index[i_sent,k,org_ind].item()][0]
                        outf.write( word_nn+' {:5.3f}'.format(top_value[i_sent,k,org_ind].item())+' ' )
                    outf.write('\n')
                outf.write('\n')

            if i_batch >= args.max_batch_num:
                break

with open(args.outf, 'w') as outf:
    outf.write('Shuffled Validation Topics:\n\n')
    visualize_topics_val(dataloader_val_shuffled, outf)
    outf.write('Validation Topics:\n\n')
    visualize_topics_val(dataloader_val, outf)
    outf.write('Training Topics:\n\n')
    visualize_topics_val(dataloader_train, outf)

#test_batch_size = 1
#test_data = batchify(corpus.test, test_batch_size, args)
