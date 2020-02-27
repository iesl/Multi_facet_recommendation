import argparse
import torch
import sys
import random
import os
sys.path.insert(0, sys.path[0]+'/..')
#sys.path.append("..")
import utils
import numpy as np

#remove the duplicated sentences
#remove the sentences which are too long
#remove stop words in target
#handle the min target filtering (If more than 30 words in output, just do random sampling)
#padding and store them into tensors, (random shuffle? two sets), store train, val, and test

parser = argparse.ArgumentParser(description='Preprocessing step 2')
parser.add_argument('--data_feature', type=str, default='',
                    help='location of the data corpus')
parser.add_argument('--data_user', type=str, default='',
                    help='location of the data corpus')
parser.add_argument('--data_tag', type=str, default='',
                    help='location of the data corpus')
parser.add_argument('--save', type=str, default='',
                    help='path to save the output data')
parser.add_argument('--max_sent_len', type=int, default=512,
                    help='max sentence length for input features')
parser.add_argument('--max_target_num', type=int, default=10,
                    help='max word number for output prediction w/o stop words (including above and below sentences)')
parser.add_argument('--max_sent_num', type=int, default='100000000000',
                    help='load only this number of sentences from input corpus')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cv_fold_num', type=int, default=5,
                    help='how many cross validation patitions')
parser.add_argument('--val_size_ratio', type=float, default=0.1,
                    help='The ratio of validation size and all size (train + val + test) ')
parser.add_argument('--min_test_user', type=int, default=5,
                    help='If number of users is smaller than this number, the paper will belong the training data')


args = parser.parse_args()

print(args)

random.seed(args.seed)
np.random.seed(args.seed)


def load_w_ind(f_in, max_sent_num, max_sent_len = -1):
    w_ind_corpus = []
    num_too_long_sent = 0

    for line in f_in:
        current_sent = line.rstrip()
        fields = current_sent.split(' ')
        end_idx = len(fields)
        if max_sent_len > 0 and len(fields) > max_sent_len:
            num_too_long_sent += 1
            end_idx = max_sent_len

        w_ind_corpus.append([int(x) for x in fields[:end_idx]])
        if len(w_ind_corpus) % 100000 == 0:
            print(len(w_ind_corpus))
            sys.stdout.flush()
        if len(w_ind_corpus) > max_sent_num:
            break
    print( "Finish loading {} sentences. While truncating {} long sentences".format(len(w_ind_corpus), num_too_long_sent) )
    return w_ind_corpus

corpus_input_name = args.data_feature + "corpus_index"
#dictionary_input_name = args.data_feature + "dictionary_index"
corpus_user_name = args.data_user + "corpus_index"
#dictionary_user_name = args.data_user + "dictionary_index"
corpus_tag_name = args.data_tag + "corpus_index"
#dictionary_tag_name = args.data_tag + "dictionary_index"


#if max_ind >= 2147483648:
#    print("Will cause overflow")
#    sys.exit()


with open(corpus_input_name) as f_in:
    w_ind_corpus = load_w_ind(f_in, args.max_sent_num, args.max_sent_len)

with open(corpus_user_name) as f_in:
    user_corpus = load_w_ind(f_in, args.max_sent_num)

with open(corpus_tag_name) as f_in:
    tag_corpus = load_w_ind(f_in, args.max_sent_num)

assert len(w_ind_corpus) == len(user_corpus)
assert len(w_ind_corpus) == len(tag_corpus)

def random_cv_partition(user_corpus, cv_fold_num):
    corpus_size = len(user_corpus)

    cv_partition_idx_np = -np.ones(corpus_size, dtype=np.int32)
    shuffled_order = list(range(corpus_size) )
    random.shuffle(shuffled_order)
    partition_now = 0
    for j in shuffled_order:
        if len(user_corpus[j]) < args.min_test_user:
            continue
        cv_partition_idx_np[j] = partition_now
        partition_now += 1
        if partition_now == cv_fold_num:
            partition_now = 0

    return cv_partition_idx_np

def store_tensors(f_out,tensor1, tensor2, tensor3):
    torch.save([tensor1, tensor2, tensor3], f_out)
    
def squeeze_into_tensors(save_idx, w_ind_corpus, user_corpus, tag_corpus, output_save_file):
    def save_to_tesnor(w_ind_corpus_dup_j, tensor_feature, i):
        sent_len = len(w_ind_corpus_dup_j)
        tensor_feature[i,:sent_len] = torch.tensor(w_ind_corpus_dup_j, dtype = store_type)

    def duplicate_for_long_targets(save_idx, w_ind_corpus, user_corpus, tag_corpus, max_target_num):
        w_ind_corpus_dup = []
        user_corpus_dup = []
        tag_corpus_dup = []

        for j in save_idx:
            current_w_idx = w_ind_corpus[j]
            current_users = user_corpus[j]
            current_tags = tag_corpus[j]
            while( len(current_users) > 0 or len(current_tags) > 0):
                w_ind_corpus_dup.append(current_w_idx)
                user_len = min(max_target_num, len(current_users))
                tag_len = min(max_target_num, len(current_tags))
                user_corpus_dup.append(current_users[:user_len])
                tag_corpus_dup.append(current_tags[:tag_len])
                current_users = current_users[ user_len: ]
                current_tags = current_tags[ tag_len: ]
        
        return w_ind_corpus_dup, user_corpus_dup, tag_corpus_dup
    
    w_ind_corpus_dup, user_corpus_dup, tag_corpus_dup = duplicate_for_long_targets(save_idx, w_ind_corpus, user_corpus, tag_corpus, args.max_target_num)
    
    store_type = torch.int32
    corpus_size = len(w_ind_corpus_dup)
    
    tensor_feature = torch.zeros(corpus_size, args.max_sent_len, dtype = store_type)
    tensor_user = torch.zeros(corpus_size, args.max_target_num, dtype = store_type)
    tensor_tag = torch.zeros(corpus_size, args.max_target_num, dtype = store_type)
    
    shuffled_order = list(range(corpus_size) )
    random.shuffle(shuffled_order)
    for i, j in enumerate(shuffled_order):
        save_to_tesnor(w_ind_corpus_dup[j], tensor_feature, i)
        save_to_tesnor(user_corpus_dup[j], tensor_user, i)
        save_to_tesnor(tag_corpus_dup[j], tensor_tag, i)

    with open(output_save_file,'wb') as f_out:
        store_tensors(f_out, tensor_feature, tensor_user, tensor_tag)

def compose_dataset(output_dir, test_indicator, w_ind_corpus, user_corpus, tag_corpus):
    corpus_size = len(w_ind_corpus)
    val_size = int(corpus_size * args.val_size_ratio)
    
    test_idx = np.nonzero(test_indicator)
    train_val_idx = np.nonzero(1-test_indicator)
    test_idx = test_idx[0]
    train_val_idx = train_val_idx[0]
    #print(test_idx[0])
    #print(test_idx.size())
    #print(train_val_idx.size())
    np.random.shuffle(train_val_idx)
    
    val_idx = train_val_idx[:val_size]
    train_idx = train_val_idx[val_size:]
    #print(val_size)
    #print(train_idx[0])
    #print(val_idx[0])
    squeeze_into_tensors(train_idx, w_ind_corpus, user_corpus, tag_corpus, output_dir + "/train.pt")
    squeeze_into_tensors(val_idx, w_ind_corpus, user_corpus, tag_corpus, output_dir + "/val.pt")
    squeeze_into_tensors(test_idx, w_ind_corpus, user_corpus, tag_corpus, output_dir + "/test.pt")

cv_partition_idx_np = random_cv_partition(user_corpus, args.cv_fold_num)

for k in range(args.cv_fold_num):
    #test_indicator = (cv_partition_idx_np == k)
    test_indicator = np.equal(cv_partition_idx_np, k).astype(int)
    print(test_indicator[:10])
    #print(cv_partition_idx_np[:10])
    output_dir = args.save + '_' + str(k)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    compose_dataset(output_dir, test_indicator, w_ind_corpus, user_corpus, tag_corpus)

