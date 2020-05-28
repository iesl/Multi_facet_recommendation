import argparse
import os, sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
#from torch.utils.data import Dataset, DataLoader
#import torch.utils.data
import torch.nn.functional as F
import torch.cuda as cutorch
import gc
import random


import model as model_code
import nsd_loss
from utils import seed_all_randomness, create_exp_dir, save_checkpoint, load_idx2word_freq, load_emb_file_to_dict, load_emb_file_to_tensor, load_corpus, output_parallel_models, str2bool
from utils_testing import compute_freq_prob_idx2word
from transformers import get_linear_schedule_with_warmup

from scibert.modeling_bert import BertModel
from scibert.configuration_bert import BertConfig

parser = argparse.ArgumentParser(description='PyTorch Neural Set Decoder for Sentnece Embedding')

###path
parser.add_argument('--data', type=str, default='./data/processed/citeulike-a_lower/',
                    help='location of the data corpus')
parser.add_argument('--tensor_folder', type=str, default='tensors_cold_0',
                    help='location of the data corpus')
parser.add_argument('--training_file', type=str, default='train.pt',
                    help='location of training file')
parser.add_argument('--save', type=str,  default='./models/citeulike-a',
                    help='path to save the final model')
parser.add_argument('--target_embedding_suffix', type=str,  default='',
                    help='append this name to user_emb and tag_emb files before .pt')
parser.add_argument('--log_file_name', type=str,  default='log.txt',
                    help='the log file will be at --save / --log_file_name')

#parser.add_argument('--emb_file', type=str, default='./resources/Google-vec-neg300_filtered_wac_bookp1.txt',
#parser.add_argument('--emb_file', type=str, default='./resources/glove.840B.300d_filtered_wac_bookp1.txt',
#parser.add_argument('--stop_word_file', type=str, default='./resources/stop_word_list',
#                    help='path to the file of a stop word list')

# embeddings
# source embedding (pattern/relation word embedding)
parser.add_argument('--source_emsize', type=int, default=0,
                    help='size of word embeddings')
#parser.add_argument('--update_source_emb', default=False, action='store_true',
#                    help='Whether to update source embedding')
parser.add_argument('--source_emb_file', type=str, default='',
                    help='path to the file of a word embedding file')
parser.add_argument('--source_emb_source', type=str, default='ext',
                    help='Could be ext (external), rand or ewe (encode word embedding)')

#target embedding
parser.add_argument('--target_emsize', type=int, default=200,
                    help='size of entity pair embeddings')
parser.add_argument('--update_target_emb', type=str2bool, nargs='?', default=True,
                    help='Whether to update target embedding')
#parser.add_argument('--target_emb_source', type=str, default='ext',
#parser.add_argument('--target_emb_source', type=str, default='rand',
#                    help='Could be ext (external), rand or ewe (encode word embedding)')
parser.add_argument('--user_emb_file', type=str, default='',
                    help='Location of the user embedding file')
parser.add_argument('--tag_emb_file', type=str, default='',
                    help='Location of the tag embedding file')

###encoder
#both
parser.add_argument('--en_model', type=str, default='TRANS',
                    help='type of encoder model (LSTM, LSTM+TRANS, TRANS+LSTM, TRANS)')

parser.add_argument('--dropouti', type=float, default=0.3,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='dropout applied to the output layer (0 = no dropout) in case of LSTM, transformer dropouts in case of TRANS')
#LSTM only
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer in LSTM')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
#TRANS only
parser.add_argument('--encode_trans_layers', type=int, default=3,
                    help='How many layers we have in transformer. Do not have effect if de_model is LSTM')
parser.add_argument('--trans_nhid', type=int, default=-1,
                    help='number of hidden units per layer in transformer')
#parser.add_argument('--nhidlast', type=int, default=-1,
#                    help='number of hidden units for the last rnn layer')
#parser.add_argument('--dropouth', type=float, default=0.3,
#                    help='dropout for rnn layers (0 = no dropout)')
#parser.add_argument('--dropoutl', type=float, default=-0.2,
#                    help='dropout applied to layers (0 = no dropout)')
#parser.add_argument('--wdrop', type=float, default=0.5,
#                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')

###decoder
#both
parser.add_argument('--de_model', type=str, default='TRANS',
                    help='type of decoder model (LSTM, LSTM+TRANS, TRANS+LSTM, TRANS)')
parser.add_argument('--de_coeff_model', type=str, default='TRANS',
                    help='type of decoder model to predict coefficients (LSTM, TRANS)')
parser.add_argument('--de_output_layers', type=str, default='no_dynamic',
                    help='could be no_dynamic, single dynamic, double dynamic')
parser.add_argument('--n_basis', type=int, default=5,
                    help='number of basis we want to predict')
#parser.add_argument('--linear_mapping_dim', type=int, default=0,
#                    help='map the input embedding by linear transformation')
parser.add_argument('--positional_option', type=str, default='linear',
                    help='options of encode positional embedding into models (linear, cat, add)')
parser.add_argument('--dropoutp', type=float, default=0.3,
                    help='dropout of positional embedding or input embedding after linear transformation (when linear_mapping_dim != 0)')

#LSTM only
parser.add_argument('--nhidlast2', type=int, default=-1,
                    help='hidden embedding size of the second LSTM')
parser.add_argument('--dropout_prob_lstm', type=float, default=0,
                    help='LSTM decoder dropout')
#TRANS only
parser.add_argument('--trans_layers', type=int, default=3,
                    help='How many layers we have in transformer. Do not have effect if de_model is LSTM')
parser.add_argument('--de_en_connection', type=str2bool, nargs='?', default=True, 
                    help='If True, using Transformer decoder in our decoder. Otherwise, using Transformer encoder')
parser.add_argument('--dropout_prob_trans', type=float, default=0.3,
                    help='hidden_dropout_prob and attention_probs_dropout_prob in Transformer')

#coeff
parser.add_argument('--user_w', type=float, default=1,
                    help='Weights for user loss')
parser.add_argument('--tag_w', type=float, default=0,
                    help='Weights for tag loss')
parser.add_argument('--switch_user_tag_roles', type=str2bool, nargs='?', default=False,
                    help='If true and use TRANS_two_heads as de_coeff_model, switch the magnitude of two models')
parser.add_argument('--auto_w', type=float, default=0,
                    help='Weights for autoencoder loss')
parser.add_argument('--auto_avg', type=str2bool, nargs='?', default=False,
                    help='Average bases for autoencoder loss')
parser.add_argument('--neg_sample_w', type=float, default=1,
                    help='Negative sampling weights')
parser.add_argument('--rand_neg_method', type=str, default='shuffle',
                    help='Negative sampling method. Could be paper_uniform, uniform, shuffle, and rotate')
parser.add_argument('--w_loss_coeff', type=float, default=0.1,
                    help='weights for coefficient prediction loss')
parser.add_argument('--L1_losss_B', type=float, default=0.2,
                    help='L1 loss for the coefficient matrix')
#parser.add_argument('--coeff_opt', type=str, default='lc',
parser.add_argument('--coeff_opt', type=str, default='prod',
                    help='Could be max, lc, maxlc, prod')
parser.add_argument('--loss_type', type=str, default='sim',
                    help='Could be sim or dist')
parser.add_argument('--target_norm', type=str2bool, nargs='?', default=True,
                    help='Whether target embedding is normalized')
parser.add_argument('--target_l2', type=float, default=0,
                    help='L2 norm on target embeddings')
parser.add_argument('--inv_freq_w', type=str2bool, nargs='?', default=False,
                    help='Whether emphasize the rare users')
parser.add_argument('--coeff_opt_algo', type=str, default='rmsprop',
#parser.add_argument('--coeff_opt_algo', type=str, default='sgd_bmm',
                    help='Could be sgd_bmm, sgd, asgd, adagrad, rmsprop, and adam')

###training
parser.add_argument('--optimizer', type=str, default="Adam",
                    help='optimization algorithm. Could be SGD, Adam or AdamW')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--lr2_divide', type=float, default=1.0,
                    help='drop this ratio for the learning rate of the second LSTM')
parser.add_argument('--lr_target', type=float, default=-1,
                    help='learning rate of target embedding')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=200, metavar='N',
                    help='batch size')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--wdecay', type=float, default=1e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--nonmono', type=int, default=10,
                    help='decay learning rate after seeing how many validation performance drop')
parser.add_argument('--warmup_proportion', type=float, default=0,
                    help='fraction of warmup steps in case of AdamW with linear warmup')
parser.add_argument('--training_split_num', type=int, default=1,
                    help='We want to split training corpus into how many subsets. Splitting training dataset seems to make pytorch run much faster and we can store and eval the model more frequently')
parser.add_argument('--valid_per_epoch', type=int, default=1,
                    help='Number of times we want to run through validation data and save model within an epoch')
parser.add_argument('--copy_training', type=str2bool, nargs='?', default=True, 
                    help='turn off this option to save some cpu memory when loading training data')
#parser.add_argument('--continue_train', action='store_true',
#                    help='continue train from a checkpoint')

###system
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=str2bool, nargs='?', default=True, 
                    help='use CUDA')
parser.add_argument('--single_gpu', default=False, action='store_true',
                    help='use single GPU')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--loading_target_embedding', type=str2bool, nargs='?', default=True, 
                    help='If continue_train is true, whether we want to load user and tag embeddings')
parser.add_argument('--freeze_encoder_decoder', type=str2bool, nargs='?', default=False, 
                    help='If True, only update target embeddings')
parser.add_argument('--norm_basis_when_freezing', type=str2bool, nargs='?', default=False, 
                    help='If True, only update target embeddings')
parser.add_argument('--always_save_model', type=str2bool, nargs='?', default=False, 
                    help='If True, ignore the validation loss and always save model')
parser.add_argument('--start_training_split', type=int, default=0,
                    help='We want to split training corpus into how many subsets. Splitting training dataset seems to make pytorch run much faster and we can store and eval the model more frequently')


args = parser.parse_args()

########################
print("Set up environment")
########################
assert args.training_split_num >= args.valid_per_epoch

if args.switch_user_tag_roles:
    assert args.de_coeff_model == 'TRANS_two_heads'

if args.coeff_opt == 'maxlc':
    current_coeff_opt = 'max'
else:
    current_coeff_opt = args.coeff_opt

#if args.dropoutl < 0:
#    args.dropoutl = args.dropouth
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

if args.lr_target < 0:
    args.lr_target = args.lr

assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

if not args.continue_train:
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=['./src/main.py', './src/model.py', './src/nsd_loss.py'])

def logging(s, print_=True, log_=True):
    if print_:
        print(s)
        sys.stdout.flush()
    if log_:
        with open(os.path.join(args.save, args.log_file_name), 'a+') as f_log:
            f_log.write(s + '\n')

# Set the random seed manually for reproducibility.
seed_all_randomness(args.seed,args.cuda)

logging('Args: {}'.format(args))

########################
print("Loading data")
########################

device = torch.device("cuda" if args.cuda else "cpu")

#idx2word_freq, target_idx2word_freq, dataloader_train_arr, dataloader_val, dataloader_val_shuffled, max_sent_len = \
#    load_corpus(args.data, args.batch_size, args.batch_size, device, args.tensor_folder, args.training_file, args.training_split_num, args.copy_training)
idx2word_freq, user_idx2word_freq, tag_idx2word_freq, dataloader_train_arr, dataloader_val,  max_sent_len = \
    load_corpus(args.data, args.batch_size, args.batch_size, device, args.tensor_folder, args.training_file, args.training_split_num, args.copy_training)


def counter_to_tensor(idx2word_freq,device, uniform=True, smooth_alpha = 0):
    total = len(idx2word_freq)
    w_freq = torch.zeros(total, dtype=torch.float, device = device, requires_grad = False)
    for i in range(total):
        if uniform:
            w_freq[i] = 1
        else:
            if smooth_alpha == 0:
                w_freq[i] = idx2word_freq[i][1]
            else:
                w_freq[i] = (smooth_alpha + idx2word_freq[i][2]) / smooth_alpha
        #w_freq[i] = math.sqrt(idx2word_freq[x][1])
    w_freq[0] = -1
    return w_freq

# Initialize or load source embeddings
source_emb = torch.tensor([0.])
extra_init_idx = []
if len(args.source_emb_file) > 0:
    #with torch.no_grad():
    source_emb, source_emb_size, extra_init_idx = load_emb_file_to_tensor(args.source_emb_file, device, idx2word_freq)
    source_emb = source_emb / (0.000000000001 + source_emb.norm(dim = 1, keepdim=True))
    source_emb.requires_grad = False
    #source_emb.requires_grad = args.update_target_emb
    print("loading ", args.source_emb_file)
#else:
    #if args.source_emb_source == 'ewe':
    #    source_emb_size = args.source_emsize
    #    print("Using word embedding from encoder")
elif args.source_emb_source == 'rand': #and args.update_source_emb == True:
    source_emb_size = args.source_emsize
    source_emb = torch.randn(len(idx2word_freq), source_emb_size, device = device, requires_grad = False)
    source_emb = source_emb / (0.000000000001 + source_emb.norm(dim = 1, keepdim=True))
    source_emb.requires_grad = True
    print("Initialize source embedding randomly")
elif args.source_emb_source == 'scibert':
    print("do not need to initialize word embedding")
else:
    #print("We don't support such source_emb_source " + args.source_emb_source + ", update_source_emb ", args.update_source_emb, ", and source_emb_file "+ args.source_emb_file)
    print("We don't support such source_emb_source " + args.source_emb_source + ", and source_emb_file "+ args.source_emb_file)
    sys.exit(1)

# Load target embeddings (for now target embeddings are assumed to be always loaded)
# TODO: Add pretrained target embedding scenario?
#if args.target_emb_source == 'ext' and len(args.user_emb_file) > 0 and len(args.tag_emb_file) > 0:
#    target_emb_dict, target_emb_sz = load_emb_file_to_dict(args.target_emb_file)
#    num_entpairs = len(target_emb_dict)
#    target_emb = torch.empty(num_entpairs, target_emb_sz, device=device, requires_grad=False)
#    for entpair, emb in target_emb_dict.items():
#        index = int(entpair[2:])        
#        val = torch.tensor(emb, device = device, requires_grad = False)        
#        target_emb[index,:] = val
#    target_emb.requires_grad = args.update_target_emb
#elif args.target_emb_source == 'rand' and args.update_target_emb:
#    target_emb_sz = args.target_emsize
#    user_emb = torch.randn(len(user_idx2word_freq), target_emb_sz, device = device, requires_grad = False)
#    user_emb = user_emb / (0.000000000001 + user_emb.norm(dim = 1, keepdim=True))
#    user_emb.requires_grad = True
#    tag_emb = torch.randn(len(tag_idx2word_freq), target_emb_sz, device = device, requires_grad = False)
#    tag_emb = tag_emb / (0.000000000001 + tag_emb.norm(dim = 1, keepdim=True))
#    tag_emb.requires_grad = True
#    print("Initialize target embedding randomly")
#else:
#    print("We don't support such target_emb_source " + args.target_emb_source + ", update_target_emb ", args.update_target_emb, ", and user_emb_file " + args.user_emb_file)
#    sys.exit(1)

num_special_token = 3

def load_ext_emb(emb_file, target_emb_sz, idx2word_freq):
    num_w = len(idx2word_freq)
    if len(emb_file) > 0:
        if emb_file[-3:] == '.pt':
            target_emb = torch.load(emb_file).to(device = device)
            target_emb.requires_grad = False
            target_emb_sz = target_emb.size(1)
        else:
            word2emb, emb_size = load_emb_file_to_dict(emb_file, convert_np = False)
            target_emb_sz = emb_size
            target_emb = torch.randn(num_w, target_emb_sz, device = device, requires_grad = False)
            OOV_freq = 0
            total_freq = 0
            OOV_type = 0
            for i in range(num_special_token, num_w):
                w = idx2word_freq[i][0]
                total_freq += idx2word_freq[i][1]
                if w in word2emb:
                    val = torch.tensor(word2emb[w], device = device, requires_grad = False)
                    #val = np.array(word2emb[w])
                    target_emb[i,:] = val
                else:
                    OOV_type += 1 
                    OOV_freq += idx2word_freq[i][1]
            print("OOV word type percentage: {}%".format( OOV_type/float(num_w)*100 ))
            print("OOV token percentage: {}%".format( OOV_freq/float(total_freq)*100 ))
    else:
        target_emb = torch.randn(num_w, target_emb_sz, device = device, requires_grad = False)
    #if args.coeff_opt != 'prod':
    target_emb = target_emb / (0.000000000001 + target_emb.norm(dim = 1, keepdim=True))
    target_emb.requires_grad = True
    return target_emb, target_emb_sz

user_emb, target_emb_sz = load_ext_emb(args.user_emb_file, args.target_emsize, user_idx2word_freq)
if args.tag_w > 0:
    tag_emb, target_emb_sz_tag = load_ext_emb(args.tag_emb_file, args.target_emsize, tag_idx2word_freq)
else:
    tag_emb = torch.zeros(0)
    target_emb_sz_tag = target_emb_sz

assert target_emb_sz == target_emb_sz_tag

if args.trans_nhid < 0:
    if args.target_emsize > 0:
        args.trans_nhid = args.target_emsize
    else:
        args.trans_nhid = target_emb_sz


#w_freq = counter_to_tensor(idx2word_freq,device)
if not args.inv_freq_w:
    user_uniform = counter_to_tensor(user_idx2word_freq, device, uniform=True)
    tag_uniform = counter_to_tensor(tag_idx2word_freq, device, uniform=True)
else:
    user_uniform = counter_to_tensor(user_idx2word_freq, device, uniform=False)
    tag_uniform = counter_to_tensor(tag_idx2word_freq, device, uniform=False)
user_freq = counter_to_tensor(user_idx2word_freq, device, uniform=False)
tag_freq = counter_to_tensor(tag_idx2word_freq, device, uniform=False)
user_freq[:num_special_token] = 0 #When do the categorical sampling, do not include <null>, <eos> and <unk> (just gives 0 probability)
tag_freq[:num_special_token] = 0

if args.auto_w > 0:
    compute_freq_prob_idx2word(idx2word_freq)
    feature_uniform = counter_to_tensor(idx2word_freq, device, uniform=False, smooth_alpha=1e-4)
    feature_freq = counter_to_tensor(idx2word_freq, device, uniform=False)
    feature_linear_layer = torch.randn(source_emb_size, target_emb_sz, device = device, requires_grad = True)
else:
    feature_linear_layer = torch.zeros(0)


########################
print("Building models")
########################

#if args.en_model == "LSTM":
#encoder = model_code.RNNModel(args.en_model, ntokens, args.source_emsize, args.nhid, args.nhidlast, args.nlayers,
#               args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop,
#               args.tied, args.dropoutl, args.n_experts)


if args.en_model == 'scibert':
    model_name = 'scibert-scivocab-uncased'
    bert_config = BertConfig.from_pretrained(model_name)
    bert_config.hidden_dropout_prob = args.dropout
    bert_config.attention_probs_dropout_prob = args.dropout
    encoder = BertModel.from_pretrained(model_name, config = bert_config)
    encoder.output_dim = bert_config.hidden_size
else:
    ntokens = len(idx2word_freq)
    #encoder = model_code.RNNModel_simple(args.en_model, ntokens, args.source_emsize, args.nhid, args.nlayers,
    encoder = model_code.SEQ2EMB(args.en_model.split('+'), ntokens, args.source_emsize, args.nhid, args.nlayers,
                                 args.dropout, args.dropouti, args.dropoute, max_sent_len, source_emb, extra_init_idx, args.encode_trans_layers, args.trans_nhid)

if args.auto_w == 0:
    del source_emb

if args.nhidlast2 < 0:
    #args.nhidlast2 = source_emb_size
    args.nhidlast2 = encoder.output_dim
#if args.linear_mapping_dim < 0:
#    args.linear_mapping_dim = encoder.output_dim

#decoder = model_code.EMB2SEQ(args.de_model.split('+'), args.de_coeff_model, encoder.output_dim, args.nhidlast2, source_emb_size, target_emb_sz, 1, args.n_basis, positional_option = args.positional_option, dropoutp= args.dropoutp, trans_layers = args.trans_layers, using_memory =  args.de_en_connection, dropout_prob_trans = args.dropout_prob_trans, dropout_prob_lstm=args.dropout_prob_lstm)
decoder = model_code.EMB2SEQ(args.de_model.split('+'), args.de_coeff_model, encoder.output_dim, args.nhidlast2, target_emb_sz, 1, args.n_basis, positional_option = args.positional_option, dropoutp= args.dropoutp, trans_layers = args.trans_layers, using_memory =  args.de_en_connection, dropout_prob_trans = args.dropout_prob_trans, dropout_prob_lstm=args.dropout_prob_lstm, de_output_layers = args.de_output_layers)
#decoder = model_code.EMB2SEQ(args.de_model.split('+'), encoder.output_dim, args.nhidlast2, source_emb_size, 1, args.n_basis, linear_mapping_dim = args.linear_mapping_dim, dropoutp= args.dropoutp, trans_layers = args.trans_layers, using_memory =  args.de_en_connection)
#decoder = model_code.RNNModel_decoder(args.de_model, args.nhid * 2, args.nhidlast2, source_emb_size, 1, args.n_basis, linear_mapping_dim = args.linear_mapping_dim, dropoutp= 0.5)
#decoder = model_code.RNNModel_decoder(args.de_model, args.nhid * 2, args.nhidlast2, source_emb_size, 1, args.n_basis, linear_mapping_dim = args.nhid, dropoutp= 0.5)

if args.de_en_connection and decoder.trans_dim is not None and encoder.output_dim != decoder.trans_dim:
    print("dimension mismatch. The encoder output dimension is ", encoder.output_dim, " and the transformer dimension in decoder is ", decoder.trans_dim)
    sys.exit(1)

import torch.nn.init as weight_init
def initialize_weights(net, normal_std):
    for name, param in net.named_parameters(): 
        if 'bias' in name or 'rnn' not in name:
            #print("skip "+name)
            continue
        print("normal init "+name+" with std"+str(normal_std) )
        weight_init.normal_(param, std = normal_std)

#initialize_weights(encoder, 0.01)
#initialize_weights(decoder, 0.01)

#if args.continue_train:
#    encoder.load_state_dict(torch.load(os.path.join(args.save, 'encoder.pt')))
#    decoder.load_state_dict(torch.load(os.path.join(args.save, 'decoder.pt')))
#load optimizers

if args.continue_train:
    encoder.load_state_dict(torch.load(os.path.join(args.save, 'encoder.pt')))
    decoder.load_state_dict(torch.load(os.path.join(args.save, 'decoder.pt')))
    if args.loading_target_embedding:
        user_emb_load = torch.load(os.path.join(args.save, 'user_emb.pt'))
        user_emb = user_emb.new_tensor(user_emb_load)
        if args.tag_w > 0:
            tag_emb_load = torch.load(os.path.join(args.save, 'tag_emb.pt'))
            tag_emb = tag_emb.new_tensor(tag_emb_load)
        if args.auto_w > 0:
            auto_emb_load = torch.load(os.path.join(args.save, 'auto_emb.pt'))
            auto_emb = auto_emb.new_tensor(auto_emb_load)
            
            

parallel_encoder, parallel_decoder = output_parallel_models(args.cuda, args.single_gpu, encoder, decoder)

total_params = sum(x.data.nelement() for x in encoder.parameters())
logging('Encoder total parameters: {}'.format(total_params))
total_params = sum(x.data.nelement() for x in decoder.parameters())
logging('Decoder total parameters: {}'.format(total_params))

########################
print("Training")
########################


def evaluate(dataloader, current_coeff_opt):
    # Turn on evaluation mode which disables dropout.
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_loss_set_user = 0
    total_loss_set_neg_user = 0
    total_loss_set_tag = 0
    total_loss_set_neg_tag = 0
    total_loss_set_auto = 0
    total_loss_set_neg_auto = 0
    total_loss_set_reg = 0
    total_loss_set_div = 0
    total_loss_set_div_target_user = 0.
    total_loss_set_div_target_tag = 0.
    #total_loss_coeff_pred = 0.
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            feature, feature_type, user, tag, repeat_num, user_len, tag_len, sample_idx = sample_batched
            
            if args.freeze_encoder_decoder and epoch > 1:
                #load cache
                sample_idx_np = sample_idx.numpy()
                basis_pred = torch.tensor(basis_pred_test_cache[sample_idx_np,:,:], dtype=torch.float ,device = device)
                basis_pred_tag = torch.tensor(basis_pred_tag_test_cache[sample_idx_np,:,:], dtype=torch.float, device = device)
            else:
                if args.en_model == 'scibert':
                    output_emb, output_emb_last = parallel_encoder(feature)
                else:
                    output_emb_last, output_emb = parallel_encoder(feature, feature_type)
                #basis_pred, coeff_pred = parallel_decoder(output_emb_last, output_emb, predict_coeff_sum = True)
                basis_pred, basis_pred_tag, basis_pred_auto = parallel_decoder(output_emb_last, output_emb, predict_coeff_sum = False)
                if args.freeze_encoder_decoder:
                    #store cache
                    if args.norm_basis_when_freezing:
                        basis_pred = basis_pred / (0.000000000001 + basis_pred.norm(dim = 2, keepdim=True) )
                        basis_pred_tag = basis_pred_tag / (0.000000000001 + basis_pred_tag.norm(dim = 2, keepdim=True) )
                    sample_idx_np = sample_idx.numpy()
                    basis_pred_test_cache[sample_idx_np,:,:] = basis_pred.cpu().numpy()
                    basis_pred_tag_test_cache[sample_idx_np,:,:] = basis_pred_tag.cpu().numpy()
            
            #output_emb, hidden, output_emb_last = parallel_encoder(feature.t())
            #if args.en_model == 'scibert':
            #    output_emb, output_emb_last = parallel_encoder(feature)
            #else:
            #    output_emb_last, output_emb = parallel_encoder(feature, feature_type)
            #basis_pred, coeff_pred =  parallel_decoder(output_emb_last, output_emb, predict_coeff_sum = True)
            #basis_pred, basis_pred_tag, basis_pred_auto =  parallel_decoder(output_emb_last, output_emb, predict_coeff_sum = False)
            #if len(args.target_emb_file) > 0 or args.target_emb_source == 'rand':
            #    input_emb = target_emb
            #elif args.target_emb_source == 'ewe':
            #    input_emb = encoder.encoder.weight.detach()

            compute_target_grad = False
            #loss_set, loss_set_reg, loss_set_div, loss_set_neg, loss_coeff_pred = nsd_loss.compute_loss_set(output_emb_last, parallel_decoder, input_emb, target, args.n_basis, args.L1_losss_B, device, w_freq, current_coeff_opt, compute_target_grad)
            # Changed input emb to target
            #loss_set, loss_set_reg, loss_set_div, loss_set_neg, loss_coeff_pred = nsd_loss.compute_loss_set(output_emb_last, basis_pred, coeff_pred, input_emb, target, args.L1_losss_B, device, target_freq, current_coeff_opt, compute_target_grad, args.coeff_opt_algo)
            if args.user_w > 0:
                #loss_set_user, loss_set_neg_user, loss_set_div, loss_set_reg, loss_set_div_target_user = nsd_loss.compute_loss_set(output_emb_last, basis_pred, None, user_emb, user, args.L1_losss_B, device, user_uniform, user_freq, repeat_num, user_len, current_coeff_opt, args.loss_type, compute_target_grad, args.coeff_opt_algo, args.rand_neg_method, args.target_norm)
                if args.switch_user_tag_roles:
                    input_basis = basis_pred_tag
                else:
                    input_basis = basis_pred
                loss_set_user, loss_set_neg_user, loss_set_div, loss_set_reg, loss_set_div_target_user = nsd_loss.compute_loss_set(input_basis, user_emb, user, args.L1_losss_B, device, user_uniform, user_freq, repeat_num, user_len, current_coeff_opt, args.loss_type, compute_target_grad, args.coeff_opt_algo, args.rand_neg_method, args.target_norm)
            else:
                loss_set_user = torch.tensor(0)
                loss_set_neg_user = torch.tensor(0)
                loss_set_div_target_user = torch.tensor(0)
                
            if args.tag_w > 0:
                #loss_set_tag, loss_set_neg_tag, loss_set_div, loss_set_reg, loss_set_div_target_tag = nsd_loss.compute_loss_set(output_emb_last, basis_pred, None, tag_emb, tag, args.L1_losss_B, device, tag_uniform, tag_freq, repeat_num, tag_len, current_coeff_opt, args.loss_type, compute_target_grad, args.coeff_opt_algo, args.rand_neg_method, args.target_norm ) #, compute_div_reg = False)
                if args.switch_user_tag_roles:
                    input_basis = basis_pred
                else:
                    input_basis = basis_pred_tag
                loss_set_tag, loss_set_neg_tag, loss_set_div, loss_set_reg, loss_set_div_target_tag = nsd_loss.compute_loss_set(input_basis, tag_emb, tag, args.L1_losss_B, device, tag_uniform, tag_freq, repeat_num, tag_len, current_coeff_opt, args.loss_type, compute_target_grad, args.coeff_opt_algo, args.rand_neg_method, args.target_norm ) #, compute_div_reg = False)
            else:
                loss_set_tag = torch.tensor(0)
                loss_set_neg_tag = torch.tensor(0)
                loss_set_div_target_tag = torch.tensor(0)
            
            if args.auto_w > 0:
                feature_len = None 
                if args.auto_avg:
                    basis_pred_auto_compressed = basis_pred_auto.mean(dim=1).unsqueeze(dim=1)
                else:
                    basis_pred_auto_compressed = basis_pred_auto
                #rand_neg_method = args.rand_neg_method
                rand_neg_method = 'rotate'
                loss_set_auto, loss_set_neg_auto = nsd_loss.compute_loss_set(basis_pred_auto_compressed, source_emb, feature, args.L1_losss_B, device, feature_uniform, feature_freq, repeat_num, feature_len, current_coeff_opt, args.loss_type, compute_target_grad, args.coeff_opt_algo, rand_neg_method, args.target_norm, compute_div_reg = False, target_linear_layer = feature_linear_layer, pre_avg = True)
            else:
                loss_set_auto = torch.tensor(0, device = device)
                loss_set_neg_auto = torch.tensor(0, device = device)
            
            #loss = loss_set + loss_set_neg + args.w_loss_coeff* loss_coeff_pred
            loss = loss_set_user + args.neg_sample_w * loss_set_neg_user + args.tag_w * ( loss_set_tag + args.neg_sample_w * loss_set_neg_tag ) + args.auto_w * ( loss_set_auto + args.neg_sample_w * loss_set_neg_auto )
            batch_size = feature.size(0)
            total_loss += loss * batch_size
            total_loss_set_user += loss_set_user * batch_size
            total_loss_set_neg_user += loss_set_neg_user * batch_size
            total_loss_set_tag += loss_set_tag * batch_size
            total_loss_set_neg_tag += loss_set_neg_tag * batch_size
            total_loss_set_auto += loss_set_auto * batch_size
            total_loss_set_neg_auto += loss_set_neg_auto * batch_size
            total_loss_set_reg += loss_set_reg * batch_size
            total_loss_set_div += loss_set_div * batch_size
            total_loss_set_div_target_user += loss_set_div_target_user * batch_size
            total_loss_set_div_target_tag += loss_set_div_target_tag * batch_size
            #total_loss_coeff_pred += loss_coeff_pred * batch_size

    return total_loss.item() / len(dataloader.dataset), total_loss_set_user.item() / len(dataloader.dataset), total_loss_set_neg_user.item() / len(dataloader.dataset), total_loss_set_tag.item() / len(dataloader.dataset), total_loss_set_neg_tag.item() / len(dataloader.dataset), total_loss_set_auto.item() / len(dataloader.dataset), total_loss_set_neg_auto.item() / len(dataloader.dataset), total_loss_set_reg.item() / len(dataloader.dataset), total_loss_set_div.item() / len(dataloader.dataset), total_loss_set_div_target_user.item() / len(dataloader.dataset), total_loss_set_div_target_tag.item() / len(dataloader.dataset)


def train_one_epoch(dataloader_train, lr, current_coeff_opt, split_i):
    start_time = time.time()
    total_loss = 0.
    total_loss_set_user = 0.
    total_loss_set_neg_user = 0.
    total_loss_set_tag = 0.
    total_loss_set_neg_tag = 0.
    total_loss_set_auto = 0.
    total_loss_set_neg_auto = 0.
    total_loss_set_reg = 0.
    total_loss_set_div = 0.
    total_loss_set_div_target_user = 0.
    total_loss_set_div_target_tag = 0.
    #total_loss_coeff_pred = 0.

    
    if args.freeze_encoder_decoder:
        encoder.eval()
        decoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False
        for p in decoder.parameters():
            p.requires_grad = False
    else:
        encoder.train()
        decoder.train()
    for i_batch, sample_batched in enumerate(dataloader_train):
        feature, feature_type, user, tag, repeat_num, user_len, tag_len, sample_idx = sample_batched
        #print(target)
        #print(feature.size())
        #print(target.size())
        optimizer_e.zero_grad()
        optimizer_d.zero_grad()
        optimizer_t.zero_grad()
        optimizer_auto.zero_grad()
        #encoder.zero_grad()
        #decoder.zero_grad()
        #output_emb, hidden, output_emb_last = parallel_encoder(feature.t())
        #output_emb, hidden, output_emb_last = parallel_encoder(feature)
        if args.freeze_encoder_decoder and epoch > 1:
            #load cache
            sample_idx_np = sample_idx.numpy()
            basis_pred = torch.tensor(basis_pred_train_cache[sample_idx_np,:,:], dtype=torch.float ,device = device)
            basis_pred_tag = torch.tensor(basis_pred_tag_train_cache[sample_idx_np,:,:], dtype=torch.float ,device = device)
        else:
            if args.en_model == 'scibert':
                output_emb, output_emb_last = parallel_encoder(feature)
            else:
                output_emb_last, output_emb = parallel_encoder(feature, feature_type)
            #basis_pred, coeff_pred = parallel_decoder(output_emb_last, output_emb, predict_coeff_sum = True)
            basis_pred, basis_pred_tag, basis_pred_auto = parallel_decoder(output_emb_last, output_emb, predict_coeff_sum = False)
            if args.freeze_encoder_decoder:
                #store cache
                if args.norm_basis_when_freezing:
                    basis_pred = basis_pred / (0.000000000001 + basis_pred.norm(dim = 2, keepdim=True) )
                    basis_pred_tag = basis_pred_tag / (0.000000000001 + basis_pred_tag.norm(dim = 2, keepdim=True) )
                sample_idx_np = sample_idx.numpy()
                basis_pred_train_cache[sample_idx_np,:,:] = basis_pred.cpu().numpy()
                basis_pred_tag_train_cache[sample_idx_np,:,:] = basis_pred_tag.cpu().numpy()
        #if len(args.target_emb_file) > 0  or args.target_emb_source == 'rand':
        #    input_emb = target_emb
        #elif args.target_emb_source == 'ewe':
        #    input_emb = encoder.encoder.weight.detach()
            # compute_target_grad = False
        compute_target_grad = args.update_target_emb
        #print(compute_target_grad)
        #print(input_emb.requires_grad)
        #loss_set, loss_set_reg, loss_set_div, loss_set_neg, loss_coeff_pred = nsd_loss.compute_loss_set(output_emb_last, parallel_decoder, input_emb, target, args.n_basis, args.L1_losss_B, device, w_freq, current_coeff_opt, compute_target_grad)
        if args.user_w > 0:
            #loss_set_user, loss_set_neg_user, loss_set_div, loss_set_reg, loss_set_div_target_user = nsd_loss.compute_loss_set(output_emb_last, basis_pred, None, user_emb, user, args.L1_losss_B, device, user_uniform, user_freq, repeat_num, user_len, current_coeff_opt, args.loss_type, compute_target_grad, args.coeff_opt_algo, args.rand_neg_method, args.target_norm)
            if args.switch_user_tag_roles:
                input_basis = basis_pred_tag
            else:
                input_basis = basis_pred
            loss_set_user, loss_set_neg_user, loss_set_div, loss_set_reg, loss_set_div_target_user = nsd_loss.compute_loss_set(input_basis, user_emb, user, args.L1_losss_B, device, user_uniform, user_freq, repeat_num, user_len, current_coeff_opt, args.loss_type, compute_target_grad, args.coeff_opt_algo, args.rand_neg_method, args.target_norm)
            if torch.isnan(loss_set_user):
                sys.stdout.write('user nan, ')
                continue
        else:
            loss_set_user = torch.tensor(0, device = device)
            loss_set_neg_user = torch.tensor(0, device = device)
            loss_set_div_target_user = torch.tensor(0, device = device)
            
        if args.tag_w > 0:
            if args.switch_user_tag_roles:
                input_basis = basis_pred
            else:
                input_basis = basis_pred_tag
            loss_set_tag, loss_set_neg_tag, loss_set_div, loss_set_reg, loss_set_div_target_tag = nsd_loss.compute_loss_set(input_basis, tag_emb, tag, args.L1_losss_B, device, tag_uniform, tag_freq, repeat_num, tag_len, current_coeff_opt, args.loss_type, compute_target_grad, args.coeff_opt_algo, args.rand_neg_method, args.target_norm)#, compute_div_reg = False)
            #loss_set_tag, loss_set_neg_tag, loss_set_div, loss_set_reg, loss_set_div_target_tag = nsd_loss.compute_loss_set(output_emb_last, basis_pred, None, tag_emb, tag, args.L1_losss_B, device, tag_uniform, tag_freq, repeat_num, tag_len, current_coeff_opt, args.loss_type, compute_target_grad, args.coeff_opt_algo, args.rand_neg_method, args.target_norm)#, compute_div_reg = False)
            if torch.isnan(loss_set_tag):
                sys.stdout.write('tag nan, ')
                continue
        else:
            loss_set_tag = torch.tensor(0, device = device)
            loss_set_neg_tag = torch.tensor(0, device = device)
            loss_set_div_target_tag = torch.tensor(0, device = device)
        
        if args.auto_w > 0:
            feature_len = None 
            if args.auto_avg:
                basis_pred_auto_compressed = basis_pred_auto.mean(dim=1).unsqueeze(dim=1)
            else:
                basis_pred_auto_compressed = basis_pred_auto
            #rand_neg_method = args.rand_neg_method
            rand_neg_method = 'rotate'
            loss_set_auto, loss_set_neg_auto = nsd_loss.compute_loss_set(basis_pred_auto_compressed, source_emb, feature, args.L1_losss_B, device, feature_uniform, feature_freq, repeat_num, feature_len, current_coeff_opt, args.loss_type, compute_target_grad, args.coeff_opt_algo, rand_neg_method, args.target_norm, compute_div_reg = False, target_linear_layer = feature_linear_layer, pre_avg = True)
            if torch.isnan(loss_set_auto):
                sys.stdout.write('auto nan, ')
                continue
        else:
            loss_set_auto = torch.tensor(0, device = device)
            loss_set_neg_auto = torch.tensor(0, device = device)
        
        total_loss_set_user += loss_set_user.item() * args.small_batch_size / args.batch_size
        total_loss_set_neg_user += loss_set_neg_user.item() * args.small_batch_size / args.batch_size
        total_loss_set_tag += loss_set_tag.item() * args.small_batch_size / args.batch_size
        total_loss_set_neg_tag += loss_set_neg_tag.item() * args.small_batch_size / args.batch_size
        total_loss_set_auto += loss_set_auto.item() * args.small_batch_size / args.batch_size
        total_loss_set_neg_auto += loss_set_neg_auto.item() * args.small_batch_size / args.batch_size

        total_loss_set_reg += loss_set_reg.item() * args.small_batch_size / args.batch_size
        total_loss_set_div += loss_set_div.item() * args.small_batch_size / args.batch_size
        total_loss_set_div_target_tag += loss_set_div_target_tag.item() * args.small_batch_size / args.batch_size
        total_loss_set_div_target_user += loss_set_div_target_user.item() * args.small_batch_size / args.batch_size
        #total_loss_coeff_pred += loss_coeff_pred.item() * args.small_batch_size / args.batch_size
        
        #BT_nonneg = torch.max( torch.tensor([0.0], device=device), BT )
        #loss = loss_set + loss_set_neg + args.w_loss_coeff* loss_coeff_pred
        #loss = 9 * torch.max( torch.tensor([0.7], device=device), loss_set) +  loss_set + loss_set_neg + args.w_loss_coeff* loss_coeff_pred + 0.01 * loss_set_div
        #loss = 4 * torch.max( torch.tensor([0.7], device=device), loss_set) + 4 * torch.max( torch.tensor([0.7], device=device), -loss_set_neg) +  loss_set + loss_set_neg + args.w_loss_coeff* loss_coeff_pred
        #loss = loss_set + 0.9 * loss_set_neg + args.w_loss_coeff* loss_coeff_pred
        #loss = loss_set + args.w_loss_coeff* loss_coeff_pred
        #loss = loss_set + args.w_loss_coeff* loss_coeff_pred
        loss = args.user_w * loss_set_user 
        loss += args.tag_w * loss_set_tag 
        loss += args.auto_w * loss_set_auto
        if args.loss_type == 'sim':
            loss += args.user_w * args.neg_sample_w * loss_set_neg_user
            loss += args.tag_w * args.neg_sample_w * loss_set_neg_tag
            loss += args.auto_w * args.neg_sample_w * loss_set_neg_auto
        else:
            if -loss_set_neg_user > 1:
                loss -= args.user_w * args.neg_sample_w * loss_set_neg_user
            else:
                loss += args.user_w * args.neg_sample_w * loss_set_neg_user
            if -loss_set_neg_tag > 1:
                loss -= args.tag_w * args.neg_sample_w * loss_set_neg_tag
            else:
                loss += args.tag_w * args.neg_sample_w * loss_set_neg_tag
            if -loss_set_neg_auto > 1:
                loss -= args.auto_w * args.neg_sample_w * loss_set_neg_auto
            else:
                loss += args.auto_w * args.neg_sample_w * loss_set_neg_auto
        
        #loss += 0.01 * (loss_set_div_target_user + loss_set_div_target_tag)

        loss *= args.small_batch_size / args.batch_size
        total_loss += loss.item()

        loss.backward()

        gc.collect()
        
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)
        optimizer_e.step()
        #if len(args.target_emb_file) == 0 and args.target_emb_source == 'ewe':
        #    encoder.encoder.weight.data[0,:] = 0
            
        optimizer_d.step()

        if args.update_target_emb:
            optimizer_t.step()
            optimizer_auto.step()
            if args.user_w > 0:
                user_emb.data[0,:] = 0 
            
            if args.tag_w > 0:
                tag_emb.data[0,:] = 0
            #if args.coeff_opt != 'prod':
            #    user_emb.data = user_emb.data / (0.000000000001 + user_emb.data.norm(dim = 1, keepdim=True))
            #    tag_emb.data = tag_emb.data / (0.000000000001 + tag_emb.data.norm(dim = 1, keepdim=True))
            #print(external_emb.requires_grad)
            #print(external_emb.grad)
            #if args.optimizer == 'SGD':
            #    target_emb.data -= lr/args.lr2_divide/10.0 * target_emb.grad.data
            #else: #Adam or AdamW
            #    #target_emb.data -= 0.1/args.lr2_divide/10.0 * target_emb.grad.data
            #    target_emb.data -= 10/args.lr2_divide/10.0 * target_emb.grad.data
            #if args.target_emb_source != 'ewe':
            #    target_emb.data[0,:] = 0
            #user_emb.grad.data.zero_()
        
        if args.optimizer == 'AdamW':
            scheduler_e.step()
            scheduler_d.step()
            scheduler_t.step()

        if i_batch % args.log_interval == 0 and i_batch > 0:
            cur_loss = total_loss / args.log_interval
            cur_loss_set_user = total_loss_set_user / args.log_interval
            cur_loss_set_neg_user = total_loss_set_neg_user / args.log_interval
            cur_loss_set_tag = total_loss_set_tag / args.log_interval
            cur_loss_set_neg_tag = total_loss_set_neg_tag / args.log_interval
            cur_loss_set_auto = total_loss_set_auto / args.log_interval
            cur_loss_set_neg_auto = total_loss_set_neg_auto / args.log_interval
            cur_loss_set_reg = total_loss_set_reg / args.log_interval
            cur_loss_set_div = total_loss_set_div / args.log_interval
            cur_loss_set_div_target_user = total_loss_set_div_target_user / args.log_interval
            cur_loss_set_div_target_tag = total_loss_set_div_target_tag / args.log_interval
            #cur_loss_coeff_pred = total_loss_coeff_pred / args.log_interval
            elapsed = time.time() - start_time
            logging('| e {:3d} {:3d} | {:5d}/{:5d} b | lr-enc {:.6f} | ms/batch {:5.2f} | '
                    'l {:5.2f} | l_f_u {:5.5f} + {:2.2f}*{:5.5f} = {:5.5f} | div_u {:5.2f} | l_f_t {:5.4f} + {:2.2f}*{:5.4f} = {:5.4f} | div_t {:5.2f} | l_f_a {:5.4f} + {:2.2f}*{:5.4f} = {:5.4f} | reg {:5.2f} | div {:5.2f} '.format(
                epoch, split_i, i_batch, len(dataloader_train.dataset) // args.batch_size, optimizer_e.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, cur_loss_set_user, args.neg_sample_w, cur_loss_set_neg_user, cur_loss_set_user + args.neg_sample_w * cur_loss_set_neg_user, cur_loss_set_div_target_user, cur_loss_set_tag, args.neg_sample_w, cur_loss_set_neg_tag, cur_loss_set_tag + args.neg_sample_w * cur_loss_set_neg_tag, cur_loss_set_div_target_tag, cur_loss_set_auto, args.neg_sample_w, cur_loss_set_neg_auto, cur_loss_set_auto + args.neg_sample_w * cur_loss_set_neg_auto, cur_loss_set_reg, cur_loss_set_div))
            #if args.coeff_opt == 'maxlc' and current_coeff_opt == 'max' and cur_loss_set + cur_loss_set_neg < -0.02:
            if args.coeff_opt == 'maxlc' and current_coeff_opt == 'max' and cur_loss_set_user + cur_loss_set_neg_user < -0.02:
                current_coeff_opt = 'lc'
                print("switch to lc")
            total_loss = 0.
            total_loss_set_user = 0.
            total_loss_set_neg_user = 0.
            total_loss_set_tag = 0.
            total_loss_set_neg_tag = 0.
            total_loss_set_auto = 0.
            total_loss_set_neg_auto = 0.
            total_loss_set_reg = 0.
            total_loss_set_div = 0.
            total_loss_set_div_target_tag = 0.
            total_loss_set_div_target_user = 0.
            #total_loss_coeff_pred = 0.
            start_time = time.time()
    return current_coeff_opt

if args.optimizer == 'SGD':
    optimizer_e = torch.optim.SGD(encoder.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer_d = torch.optim.SGD(decoder.parameters(), lr=args.lr/args.lr2_divide, weight_decay=args.wdecay)
    #optimizer_t = torch.optim.SGD([user_emb, tag_emb], lr=args.lr, weight_decay=args.wdecay)
    optimizer_t = torch.optim.SGD([user_emb, tag_emb, feature_linear_layer], lr=args.lr_target)
elif args.optimizer == 'Adam':
    optimizer_e = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer_d = torch.optim.Adam(decoder.parameters(), lr=args.lr/args.lr2_divide, weight_decay=args.wdecay)
    #optimizer_t = torch.optim.Adam([user_emb, tag_emb], lr=args.lr, weight_decay=args.wdecay)
    optimizer_t = torch.optim.Adam([user_emb, tag_emb], lr=args.lr_target, weight_decay=args.target_l2)# , weight_decay=0.00000001)
    optimizer_auto = torch.optim.SGD([feature_linear_layer], lr=args.lr_target)#, weight_decay=args.target_l2)# , weight_decay=0.00000001)
    #optimizer_t = torch.optim.Adam([user_emb, tag_emb], lr=args.lr/5)
else:
    optimizer_e = torch.optim.AdamW(encoder.parameters(), lr=args.lr)
    optimizer_d = torch.optim.AdamW(decoder.parameters(), lr=args.lr/args.lr2_divide)
    optimizer_t = torch.optim.AdamW([user_emb, tag_emb, feature_linear_layer], lr=args.lr_target)
    num_training_steps = sum([len(train_split) for train_split in dataloader_train_arr]) * args.epochs
    num_warmup_steps = args.warmup_proportion * num_training_steps
    print("Warmup steps:{}, Total steps:{}".format(num_warmup_steps, num_training_steps))
    scheduler_e = get_linear_schedule_with_warmup(optimizer_e, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps) 
    scheduler_d = get_linear_schedule_with_warmup(optimizer_d, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps) 
    scheduler_t = get_linear_schedule_with_warmup(optimizer_t, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps) 


lr = args.lr
best_val_loss = None
nonmono_count = 0
saving_freq = int(math.floor(args.training_split_num / args.valid_per_epoch))


if args.freeze_encoder_decoder:
    assert len(dataloader_train_arr) == 1
    assert args.auto_w == 0
    num_sample_train = dataloader_train_arr[0].dataset.feature.size(0)
    basis_pred_train_cache = np.empty( (num_sample_train, args.n_basis, target_emb_sz) )
    basis_pred_tag_train_cache = np.empty( (num_sample_train, args.n_basis, target_emb_sz) )
    num_sample_test = dataloader_val.dataset.feature.size(0)
    basis_pred_test_cache = np.empty( (num_sample_test, args.n_basis, target_emb_sz) )
    basis_pred_tag_test_cache = np.empty( (num_sample_test, args.n_basis, target_emb_sz) )

steps = 0
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    for i in range(len(dataloader_train_arr)):
        if epoch == 1 and i < args.start_training_split:
            print("Skipping epoch "+str(epoch) + ' split '+str(i) )
            continue
        current_coeff_opt = train_one_epoch(dataloader_train_arr[i], lr, current_coeff_opt, i)
        steps += len(dataloader_train_arr[i])
        if i != args.training_split_num - 1 and (i + 1) % saving_freq != 0:
            continue

        if dataloader_val is not None:
            val_loss_all, val_loss_set_user, val_loss_set_neg_user, val_loss_set_tag, val_loss_set_neg_tag, val_loss_set_auto, val_loss_set_neg_auto, val_loss_set_reg, val_loss_set_div, val_loss_set_div_target_user, val_loss_set_div_target_tag = evaluate(dataloader_val, current_coeff_opt)
            logging('-' * 89)
            logging('| end of epoch {:3d} split {:3d} | time: {:5.2f}s | lr {:.6f} | valid loss {:5.2f} | l_f_u {:5.5f} + {:2.2f}*{:5.5f} = {:5.5f} | div_u {:5.2f} | l_f_t {:5.4f} + {:2.2f}*{:5.4f} = {:5.4f} | div_t {:5.2f} | l_f_a {:5.4f} + {:2.2f}*{:5.4f} = {:5.4f} | reg {:5.2f} | div {:5.2f} | '
                    .format(epoch, i, (time.time() - epoch_start_time), lr, val_loss_all, val_loss_set_user, args.neg_sample_w, val_loss_set_neg_user, val_loss_set_user + args.neg_sample_w * val_loss_set_neg_user, val_loss_set_div_target_user, val_loss_set_tag, args.neg_sample_w, val_loss_set_neg_tag, val_loss_set_tag + args.neg_sample_w*val_loss_set_neg_tag, val_loss_set_div_target_tag, val_loss_set_auto, args.neg_sample_w, val_loss_set_neg_auto, val_loss_set_auto + args.neg_sample_w*val_loss_set_neg_auto, val_loss_set_reg, val_loss_set_div))
            logging('-' * 89)
        if args.user_w >= args.tag_w:
            val_loss_important = val_loss_set_user + val_loss_set_neg_user
        else:
            val_loss_important = val_loss_set_tag + val_loss_set_neg_tag
            
        if args.always_save_model or not best_val_loss or val_loss_important < best_val_loss:
            #save_checkpoint(encoder, decoder, optimizer_e, optimizer_d, source_emb, target_emb, args.save)
            if args.freeze_encoder_decoder:
                save_model = False
            else:
                save_model = True
            target_embedding_suffix = args.target_embedding_suffix
            if args.always_save_model:
                target_embedding_suffix += '_always'

            save_checkpoint(encoder, decoder, optimizer_e, optimizer_d, optimizer_t, user_emb, tag_emb, feature_linear_layer, args.save, save_model = save_model, target_embedding_suffix = target_embedding_suffix)
            best_val_loss = val_loss_important
            logging('Models Saved')
        else:
            nonmono_count += 1
        
        # Do not anneal lr when in warmup phase
        #if args.optimizer == 'AdamW' and steps < num_warmup_steps:
        if args.optimizer == 'AdamW':
            continue

        if nonmono_count >= args.nonmono:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            nonmono_count = 0
            lr /= 4.0
            for param_group in optimizer_e.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_d.param_groups:
                param_group['lr'] = lr/args.lr2_divide
            for param_group in optimizer_t.param_groups:
                #param_group['lr'] = lr/5.0
                param_group['lr'] = lr / args.lr * args.lr_target
            for param_group in optimizer_auto.param_groups:
                #param_group['lr'] = lr/5.0
                param_group['lr'] = lr / args.lr * args.lr_target
