import argparse
import os
import numpy as np
import random
import torch
import sys
import json
sys.path.insert(0, sys.path[0]+'/../..')

#import torch.nn as nn
#import torch.utils.data
#import coherency_eval


from utils import seed_all_randomness, load_word_dict, load_idx2word_freq, loading_all_models, load_testing_article_summ, str2bool, Logger
import utils_testing
from pythonrouge.pythonrouge import Pythonrouge

parser = argparse.ArgumentParser(description='PyTorch Neural Set Decoder for Sentnece Embedding')

###path
parser.add_argument('--input', type=str, default='/iesl/canvas/hschang/language_modeling/cnn-dailymail/data/finished_files_subset',
                    help='location of the data corpus')
parser.add_argument('--dict', type=str, default='./data/processed/wiki2016_min100/dictionary_index',
                    help='location of the dictionary corpus')
parser.add_argument('--checkpoint', type=str, default='./models/',
                    help='model checkpoint to use')
parser.add_argument('--emb_file', type=str, default='target_emb.pt',
                    help='path to the file of a word embedding file')
parser.add_argument('--outf', type=str, default='eval_log/summ_scores.txt',
                    help='output file for generated text')
parser.add_argument('--outf_vis', type=str, default='eval_log/summ_vis/summ_',
                    help='output file for generated text')

###system
parser.add_argument('--baseline_only', default=False, action='store_true',
                    help='only compute the baselines')
parser.add_argument('--top_k_max', type=int, default=10,
                    help='How many sentences we want to select at most')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=str2bool, nargs='?', default=True,
                    help='use CUDA')
parser.add_argument('--single_gpu', default=False, action='store_true',
                    help='use single GPU')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size')
#parser.add_argument('--max_batch_num', type=int, default=100,
#                    help='number of batches for evaluation')

###data_format
parser.add_argument('--max_sent_len', type=int, default=50,
                    help='max sentence length for input features')

utils_testing.add_model_arguments(parser)

args = parser.parse_args()

args.outf_vis = '{}-{}.json'.format(args.outf_vis, time.strftime("%Y%m%d-%H%M%S"))

logger = Logger(args.outf)

logger.logging(args)
logger.logging(args.outf_vis)

if args.emb_file == "target_emb.pt":
    args.emb_file =  os.path.join(args.checkpoint,"target_emb.pt")

seed_all_randomness(args.seed, args.cuda)


########################
print("Loading Models")
########################
with open(args.dict) as f_in:
    idx2word_freq = load_idx2word_freq(f_in)

if not args. baseline_only:
    parallel_encoder, parallel_decoder, encoder, decoder, word_norm_emb = loading_all_models(args, idx2word_freq, device, args.max_sent_len)

    encoder.eval()
    decoder.eval()


with open(args.dict) as f_in:
    word_d2_idx_freq, max_ind = load_word_dict(f_in)

########################
print("Processing data")
########################

def load_tokenized_story(f_in):
    article = []
    abstract = []

    next_is_highlight = False
    for line in f_in:
        line = line.rstrip()
        if line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            abstract.append(line)
        else:
            article.append([line])
    #@highlight
    return article, abstract

##convert the set of word embedding we want to reconstruct into tensor
def article_to_embs(article, word_norm_emb, word_d2_idx_freq, device):
    emb_size = word_norm_emb.size(1)
    num_sent = len(article)
    sent_embs_tensor = torch.zeros(num_sent, emb_size, device = device)
    w_emb_tensors_list = []
    sent_lens = torch.empty(num_sent, device = device)
    for i_sent, sent in enumerate(article):
        w_emb_list = []
        w_list = sent.split()
        for w in w_list:
            if w in word_d2_idx_freq:
                w_idx = word_d2_idx_freq[w][0]
                sent_embs_tensor[i_sent,:] += word_norm_emb[w_idx,:]
                w_emb_list.append(word_norm_emb[w_idx,:])
        w_emb_tensors_list.append( torch.cat(w_emb_list, dim = 0) )
        sent_lens[i_sent] = len(w_list)
    sent_embs_tensor = sent_embs_tensor / (0.000000000001 + sent_embs_tensor.norm(dim = 1, keepdim=True) )

    flatten_article = [ w for sent in article for w in sent.split() ]
    num_word = len(flatten_article)
    all_words_tensor = torch.zeros(num_word, emb_size, device = device)
    for i_w, w in enumerate(flatten_article):
        if w in word_d2_idx_freq:
            w_idx = word_d2_idx_freq[w][0]
            all_words_tensor[i_w,:] = word_norm_emb[w_idx,:]

    return sent_embs_tensor, all_words_tensor, w_emb_tensors_list, sent_lens

def greedy_selection(sent_words_sim, top_k_max, sent_lens = None):
    num_words = sent_words_sim.size(1)
    max_sim = -torch.ones( (1,num_words), device = device )
    max_sent_idx_list = []
    for i in range(top_k_max):
        sent_sim_improvement = sent_words_sim - max_sim
        sent_sim_improvement[sent_sim_improvement<0] = 0
        sim_improve_sum = sent_sim_improvement.sum(dim = 1)
        if sent_lens is not None:
            sim_improve_sum = sim_improve_sum / sent_lens
        selected_sent = torch.argmax(sim_improve_sum)

        max_sim = max_sim + sent_sim_improvement[selected_sent,:]
        max_sent_idx_list.append(selected_sent.item())
    
    return max_sent_idx_list

def select_by_avg_dist_boost( sent_embs_tensor, all_words_tensor, top_k_max, device):
    sent_words_sim = torch.mm( sent_embs_tensor, all_words_tensor.t() )
    max_sent_idx_list = greedy_selection(sent_words_sim, top_k_max)
    return max_sent_idx_list

def select_by_topics(basis_coeff_list, all_words_tensor, top_k_max, device, sent_lens = None):
    sent_num = len(basis_coeff_list)
    num_words = all_words_tensor.size(1)
    sent_word_sim = torch.empty( (sent_num, num_words), device = device )
    for i in range(sent_num):
        if sent_lens is None:
            basis, topic_w = basis_coeff_list[i]
        else:
            basis= basis_coeff_list[i]
        sent_word_sim[i,:] = torch.mm( basis, all_words_tensor.t() ).max(dim = 0)
    max_sent_idx_list = greedy_selection(sent_words_sim, top_k_max, sent_lens)
    return max_sent_idx_list



def rank_sents(basis_coeff_list, article, word_norm_emb, word_d2_idx_freq, top_k_max, device):
    assert len(basis_coeff_list) == len(article)
    sent_embs_tensor, all_words_tensor, w_emb_tensors_list, sent_lens = article_to_embs(article, word_norm_emb, word_d2_idx_freq, device)
    
    m_d2_sent_ranks = {} 
    m_d2_sent_ranks['sent_emb_dist_avg'] = select_by_avg_dist_boost( sent_embs_tensor, all_words_tensor, top_k_max, device )
    if sent_d2_basis is not None:
        m_d2_sent_ranks['ours'] = select_by_topics(basis_coeff_list, all_words_tensor, top_k_max, device)
    #m_d2_sent_ranks['sent_emb_freq_dist_avg'] =
    m_d2_sent_ranks['norm_w_in_sent'] = select_by_topics( w_emb_tensors_list, all_words_tensor, top_k_max, device, sent_lens)
    #m_d2_sent_ranks['sent_emb_cluster_dist'] = 
    
    return m_d2_sent_ranks

device = torch.device("cuda" if args.cuda else "cpu")

fname_d2_sent_rank = {}
fname_list = []
article_list = []
abstract_list = []

stories = os.listdir(args.input)
for file_name in stories:
    print("Processing "+ file_name)
    #base_name = os.path.basename(file_name)
    with open(stories_dir + '/' + file_name) as f_in:
        
        fname_list.append(file_name)
        article, abstract = load_tokenized_story(f_in)
        article_list.append(article)
        abstract_list.append(abstract)
        with torch.no_grad():
            if args.baseline_only:
                sent_d2_basis = None
                article_proc = article
            else:
                dataloader_test = load_testing_article_summ(word_d2_idx_freq, article, args.max_sent_len, args.batch_size, device)
                #sent_d2_basis, article_proc = utils_testing.output_sent_basis_summ(dataloader_test, org_sent_list, parallel_encoder, parallel_decoder, args.n_basis, idx2word_freq)
                basis_coeff_list, article_proc = utils_testing.output_sent_basis_summ(dataloader_test, parallel_encoder, parallel_decoder, args.n_basis, idx2word_freq)
            fname_d2_sent_rank[file_name] = rank_sents(basis_coeff_list, article_proc, word_norm_emb, word_d2_idx_freq, args.top_k_max, device)


with open(args.outf_vis, 'w') as f_out:
    json.dump(fname_d2_sent_rank, f_out, indent = 1)

########################
print("Scoring all methods")
########################

if args.baseline_only:
    all_method_list = ['sent_emb_dist_avg', 'norm_w_in_sent', 'first']
else:
    all_method_list = ['ours', 'sent_emb_dist_avg', 'norm_w_in_sent', 'first']

for top_k in range(args.top_k_max):
    logger.logging(str(top_k))
    for method in all_method_list:
        logger.logging(method)
        selected_sent_all = []
        for i in range(len(fname_list)):
            file_name = fname_list[i]
            article = article_list[i]
            if method == 'first':
                sent_rank = list(range(top_k))
            else:
                sent_rank = fname_d2_sent_rank[file_name][method]
            selected_sent = [article[s] for s in sent_rank[:top_k]]
            selected_sent_all.append(selected_sent)
        rouge = Pythonrouge(summary_file_exist=False,
                    summary=selected_sent_all, reference=abstract_list,
                    n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                    #recall_only=True, stemming=True, stopwords=True,
                    recall_only=False, stemming=True, stopwords=True,
                    word_level=True, length_limit=False, length=50,
                    use_cf=True, cf=95, scoring_formula='average',
                    resampling=True, samples=1000, favor=True, p=0.5)
        score = rouge.calc_score()
        logger.logging(str(score))
