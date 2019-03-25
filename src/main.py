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

from utils import create_exp_dir, save_checkpoint, load_idx2word_freq, load_emb_file, load_corpus

parser = argparse.ArgumentParser(description='PyTorch Neural Set Decoder for Sentnece Embedding')

###path
parser.add_argument('--data', type=str, default='./data/processed/wackypedia/',
                    help='location of the data corpus')
parser.add_argument('--save', type=str,  default='./models/Wacky',
                    help='path to save the final model')
#parser.add_argument('--emb_file', type=str, default='./resources/Google-vec-neg300_filtered_wac_bookp1.txt',
parser.add_argument('--emb_file', type=str, default='./resources/glove.840B.300d_filtered_wac_bookp1.txt',
                    help='path to the file of a word embedding file')
#parser.add_argument('--stop_word_file', type=str, default='./resources/stop_word_list',
#                    help='path to the file of a stop word list')

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
#parser.add_argument('--nhidlast', type=int, default=-1,
#                    help='number of hidden units for the last rnn layer')
#parser.add_argument('--dropouth', type=float, default=0.3,
#                    help='dropout for rnn layers (0 = no dropout)')
#parser.add_argument('--dropoutl', type=float, default=-0.2,
#                    help='dropout applied to layers (0 = no dropout)')
#parser.add_argument('--wdrop', type=float, default=0.5,
#                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')

###decoder
parser.add_argument('--de_model', type=str, default='LSTM',
                    help='type of decoder model (LSTM)')
parser.add_argument('--nhidlast2', type=int, default=-1,
                    help='hidden embedding size of the second LSTM')
parser.add_argument('--n_basis', type=int, default=10,
                    help='number of basis we want to predict')
parser.add_argument('--w_loss_coeff', type=float, default=0.1,
                    help='weights for coefficient prediction loss')
parser.add_argument('--L1_losss_B', type=float, default=0.2,
                    help='L1 loss for the coefficient matrix')
parser.add_argument('--update_target_emb', default=False, action='store_true',
                    help='Whether to update target embedding')
#parser.add_argument('--coeff_opt', type=str, default='max',
parser.add_argument('--coeff_opt', type=str, default='lc',
                    help='Could be max or lc')

###training
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate')
parser.add_argument('--lr2_divide', type=float, default=1.0,
                    help='drop this ratio for the learning rate of the second LSTM')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='batch size')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--nonmono', type=int, default=1,
                    help='decay learning rate after seeing how many validation performance drop')
#parser.add_argument('--continue_train', action='store_true',
#                    help='continue train from a checkpoint')

###system
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', default=True, action='store_false',
                    help='use CUDA')
parser.add_argument('--single_gpu', default=False, action='store_true',
                    help='use single GPU')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')



args = parser.parse_args()

########################
print("Set up environment")
########################

if args.nhidlast2 < 0:
    args.nhidlast2 = args.emsize
#if args.dropoutl < 0:
#    args.dropoutl = args.dropouth
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

if not args.continue_train:
    args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(args.save, scripts_to_save=['./src/main.py', './src/model.py', './src/nsd_loss.py'])

def logging(s, print_=True, log_=True):
    if print_:
        print(s)
        sys.stdout.flush()
    if log_:
        with open(os.path.join(args.save, 'log.txt'), 'a+') as f_log:
            f_log.write(s + '\n')

# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed_all(args.seed)

########################
print("Loading data")
########################

device = torch.device("cuda" if args.cuda else "cpu")

idx2word_freq, dataloader_train, dataloader_val, dataloader_val_shuffled = load_corpus(args.data, args.batch_size, args.batch_size, device)


def counter_to_tensor(idx2word_freq,device):
    total = len(idx2word_freq)
    w_freq = torch.zeros(total, dtype=torch.float, device = device)
    for i in range(total):
        w_freq[i] = 1
        #w_freq[i] = math.sqrt(idx2word_freq[x][1])
        #w_freq[i] = idx2word_freq[x][1]
    w_freq[0] = -1
    return w_freq


external_emb = torch.tensor([0.])
if len(args.emb_file) > 0:
    #with torch.no_grad():
    external_emb, output_emb_size = load_emb_file(args.emb_file,device,idx2word_freq)
    external_emb = external_emb / (0.000000000001 + external_emb.norm(dim = 1, keepdim=True))
    external_emb.requires_grad = args.update_target_emb
else:
    output_emb_size = args.emsize

w_freq = counter_to_tensor(idx2word_freq,device)

########################
print("Building models")
########################

ntokens = len(idx2word_freq)
if args.en_model == "LSTM":
    #encoder = model_code.RNNModel(args.en_model, ntokens, args.emsize, args.nhid, args.nhidlast, args.nlayers,
    #               args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop,
    #               args.tied, args.dropoutl, args.n_experts)
    
    encoder = model_code.RNNModel_simple(args.en_model, ntokens, args.emsize, args.nhid, args.nlayers,
                   args.dropout, args.dropouti, args.dropoute, external_emb)

    decoder = model_code.RNNModel_decoder(args.de_model, args.nhid * 2, args.nhidlast2, output_emb_size, 1, args.n_basis, linear_mapping_dim = 0, dropoutp= 0.5)

#if args.continue_train:
#    encoder.load_state_dict(torch.load(os.path.join(args.save, 'encoder.pt')))
#    decoder.load_state_dict(torch.load(os.path.join(args.save, 'decoder.pt')))
#load optimizers

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

logging('Args: {}'.format(args))
total_params = sum(x.data.nelement() for x in encoder.parameters())
logging('Encoder total parameters: {}'.format(total_params))
total_params = sum(x.data.nelement() for x in decoder.parameters())
logging('Decoder total parameters: {}'.format(total_params))

########################
print("Training")
########################


def evaluate(dataloader, external_emb):
    # Turn on evaluation mode which disables dropout.
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_loss_set = 0
    total_loss_set_reg = 0
    total_loss_set_div = 0
    total_loss_set_neg = 0
    total_loss_coeff_pred = 0.
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            feature, target = sample_batched
            
            output_emb, hidden, output_emb_last = parallel_encoder(feature.t())
            if len(args.emb_file) > 0:
                input_emb = external_emb
            else:
                input_emb = encoder.encoder.weight.detach()
            compute_target_grad = False
            loss_set, loss_set_reg, loss_set_div, loss_set_neg, loss_coeff_pred = nsd_loss.compute_loss_set(output_emb_last, parallel_decoder, input_emb, target, args.n_basis, args.L1_losss_B, device, w_freq, args.coeff_opt, compute_target_grad)
            loss = loss_set + loss_set_neg + args.w_loss_coeff* loss_coeff_pred
            batch_size = feature.size(0)
            total_loss += loss * batch_size
            total_loss_set += loss_set * batch_size
            total_loss_set_reg += loss_set_reg * batch_size
            total_loss_set_div += loss_set_div * batch_size
            total_loss_set_neg += loss_set_neg * batch_size
            total_loss_coeff_pred += loss_coeff_pred * batch_size

    return total_loss.item() / len(dataloader.dataset), total_loss_set.item() / len(dataloader.dataset), \
           total_loss_set_neg.item() / len(dataloader.dataset), total_loss_coeff_pred.item() / len(dataloader.dataset), \
           total_loss_set_reg.item() / len(dataloader.dataset), total_loss_set_div.item() / len(dataloader.dataset)


def train_one_epoch(dataloader_train, external_emb, lr):
    start_time = time.time()
    total_loss = 0.
    total_loss_set = 0.
    total_loss_set_reg = 0.
    total_loss_set_div = 0.
    total_loss_set_neg = 0.
    total_loss_coeff_pred = 0.

    
    encoder.train()
    decoder.train()
    for i_batch, sample_batched in enumerate(dataloader_train):
        feature, target = sample_batched
        #print(feature.size())
        #print(target.size())
        optimizer_e.zero_grad()
        optimizer_d.zero_grad()
        #encoder.zero_grad()
        #decoder.zero_grad()
        output_emb, hidden, output_emb_last = parallel_encoder(feature.t())
        if len(args.emb_file) > 0:
            input_emb = external_emb
            compute_target_grad = args.update_target_emb
        else:
            input_emb = encoder.encoder.weight.detach()
            compute_target_grad = False
        loss_set, loss_set_reg, loss_set_div, loss_set_neg, loss_coeff_pred = nsd_loss.compute_loss_set(output_emb_last, parallel_decoder, input_emb, target, args.n_basis, args.L1_losss_B, device, w_freq, args.coeff_opt, compute_target_grad)
        total_loss_set += loss_set.item() * args.small_batch_size / args.batch_size
        total_loss_set_reg += loss_set_reg.item() * args.small_batch_size / args.batch_size
        total_loss_set_div += loss_set_div.item() * args.small_batch_size / args.batch_size
        total_loss_set_neg += loss_set_neg.item() * args.small_batch_size / args.batch_size
        total_loss_coeff_pred += loss_coeff_pred.item() * args.small_batch_size / args.batch_size
        
        loss = loss_set + loss_set_neg + args.w_loss_coeff* loss_coeff_pred
        #loss = loss_set + 0.5 * loss_set_neg + args.w_loss_coeff* loss_coeff_pred
        #loss = loss_set + args.w_loss_coeff* loss_coeff_pred
        
        loss *= args.small_batch_size / args.batch_size
        total_loss += loss.item()

        loss.backward()

        gc.collect()
        
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip)
        optimizer_e.step()
        encoder.encoder.weight.data[0,:] = 0
        optimizer_d.step()

        if args.update_target_emb:
            external_emb.data -= lr/args.lr2_divide/10.0 * external_emb.grad.data
            external_emb.grad.data.zero_()

        if i_batch % args.log_interval == 0 and i_batch > 0:
            cur_loss = total_loss / args.log_interval
            cur_loss_set = total_loss_set / args.log_interval
            cur_loss_set_reg = total_loss_set_reg / args.log_interval
            cur_loss_set_div = total_loss_set_div / args.log_interval
            cur_loss_set_neg = total_loss_set_neg / args.log_interval
            cur_loss_coeff_pred = total_loss_coeff_pred / args.log_interval
            elapsed = time.time() - start_time
            logging('| e {:3d} | {:5d}/{:5d} b | lr {:02.2f} | ms/batch {:5.2f} | '
                    'l {:5.2f} | l_f {:5.4f} + {:5.4f} | l_coeff {:5.3f} | reg {:5.2f} | div {:5.2f} '.format(
                epoch, i_batch, len(dataloader_train.dataset) // args.batch_size, optimizer_e.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, cur_loss_set, cur_loss_set_neg, cur_loss_coeff_pred, cur_loss_set_reg, cur_loss_set_div))
            total_loss = 0.
            total_loss_set = 0.
            total_loss_set_reg = 0.
            total_loss_set_div = 0.
            total_loss_set_neg = 0.
            total_loss_coeff_pred = 0.
            start_time = time.time()


optimizer_e = torch.optim.SGD(encoder.parameters(), lr=args.lr, weight_decay=args.wdecay)
optimizer_d = torch.optim.SGD(decoder.parameters(), lr=args.lr/args.lr2_divide, weight_decay=args.wdecay)

lr = args.lr
best_val_loss = None
nonmono_count = 0

for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train_one_epoch(dataloader_train, external_emb, lr)

    val_loss_all, val_loss_set, val_loss_set_neg, val_loss_ceoff_pred, val_loss_set_reg, val_loss_set_div = evaluate(dataloader_val, external_emb)
    logging('-' * 89)
    logging('| end of epoch {:3d} | time: {:5.2f}s | lr {:5.2f} | valid loss {:5.2f} | l_f {:5.4f} + {:5.4f} | l_coeff {:5.2f} | reg {:5.2f} | div {:5.2f} | '
            .format(epoch, (time.time() - epoch_start_time), lr,
                                       val_loss_all, val_loss_set, val_loss_set_neg, val_loss_ceoff_pred, val_loss_set_reg, val_loss_set_div))
    logging('-' * 89)
    #dataloader_val_shuffled?
    val_loss_all, val_loss_set, val_loss_set_neg, val_loss_ceoff_pred, val_loss_set_reg, val_loss_set_div = evaluate(dataloader_val_shuffled, external_emb)
    logging('-' * 89)
    logging('| Shuffled | time: {:5.2f}s | lr {:5.2f} | valid loss {:5.2f} | l_f {:5.4f} + {:5.4f} | l_coeff {:5.2f} | reg {:5.2f} | div {:5.2f} | '
            .format((time.time() - epoch_start_time), lr,
                                       val_loss_all, val_loss_set, val_loss_set_neg, val_loss_ceoff_pred, val_loss_set_reg, val_loss_set_div))
    logging('-' * 89)
    
    if not best_val_loss or val_loss_all < best_val_loss:
        save_checkpoint(encoder, decoder, optimizer_e, optimizer_d, external_emb, args.save)
        best_val_loss = val_loss_all
        logging('Models Saved')
    else:
        nonmono_count += 1
    
    if nonmono_count >= args.nonmono:
        # Anneal the learning rate if no improvement has been seen in the validation dataset.
        nonmono_count = 0
        lr /= 4.0
        for param_group in optimizer_e.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer_d.param_groups:
            param_group['lr'] = lr/args.lr2_divide