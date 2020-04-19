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
parser.add_argument('--data_type', type=str, default='',
                    help='location of the type corpus')
parser.add_argument('--data_bid_score', type=str, default='',
                    help='location of the bid score corpus')
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
parser.add_argument('--only_first_fold', type=utils.str2bool, nargs='?', default=False,
                    help='Only store the first fold (when the dataset is large)')
parser.add_argument('--val_size_ratio', type=float, default=0.1,
                    help='The ratio of validation size and all size (train + val + test) ')
parser.add_argument('--min_test_user', type=int, default=5,
                    help='If number of users is smaller than this number, the paper will belong the training data')
parser.add_argument('--push_to_right', type=utils.str2bool, nargs='?', default=True,
                    help='Whether we want to push the index in features to the end')
parser.add_argument('--input_file_name', type=str, default='corpus_index',
                    help='The file name where we load the indices')

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
        if max_sent_len > 0 and end_idx > max_sent_len:
            num_too_long_sent += 1
            if args.push_to_right:
                end_idx = max_sent_len - 1
                w_ind_corpus.append([int(x) for x in fields[:end_idx] + [fields[-1]] if len(x) > 0]) #make sure to include eos as the last word index
            else:
                end_idx = max_sent_len
                w_ind_corpus.append([int(x) for x in fields[:end_idx] if len(x) > 0])
        else:
            w_ind_corpus.append([int(x) for x in fields[:end_idx] if len(x) > 0])
        if len(w_ind_corpus) % 100000 == 0:
            print(len(w_ind_corpus))
            sys.stdout.flush()
        if len(w_ind_corpus) > max_sent_num:
            break
    print( "Finish loading {} sentences. While truncating {} long sentences".format(len(w_ind_corpus), num_too_long_sent) )
    return w_ind_corpus


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

def store_tensors(f_out,tensor1, tensor2, tensor3, tensor4, tensor5, tensor6, tensor7, tensor8):
    torch.save([tensor1, tensor2, tensor3, tensor4, tensor5, tensor6, tensor7, tensor8], f_out)
    
def squeeze_into_tensors(save_idx, w_ind_corpus, type_corpus, user_corpus, tag_corpus, bid_score_corpus, output_save_file):
    def save_to_tesnor(w_ind_corpus_dup_j, tensor_feature, i, store_right = False):
        sent_len = len(w_ind_corpus_dup_j)
        if store_right:
            tensor_feature[i,-(sent_len):] = torch.tensor(w_ind_corpus_dup_j, dtype = store_type)
        else:
            tensor_feature[i,:sent_len] = torch.tensor(w_ind_corpus_dup_j, dtype = store_type)

    def duplicate_for_long_targets(save_idx, w_ind_corpus, type_corpus, user_corpus, tag_corpus, bid_score_corpus, max_target_num):
        w_ind_corpus_dup = []
        user_corpus_dup = []
        tag_corpus_dup = []
        bid_score_corpus_dup = []
        type_corpus_dup = []
        num_repeat_corpus_dup = []
        len_corpus_dup = []
        
        num_no_label_item = 0

        for j in save_idx:
            current_w_idx = w_ind_corpus[j]
            current_users = user_corpus[j]
            current_tags = tag_corpus[j]
            if len(bid_score_corpus) > 0:
                current_bid_score = bid_score_corpus[j]
            else:
                current_bid_score = []
            if sum(current_users) == 0 and sum(current_tags) == 0:
                num_no_label_item += 1
                continue
            num_repeat = 0
            while( len(current_users) > 0 or len(current_tags) > 0):
                w_ind_corpus_dup.append(current_w_idx)
                if len(type_corpus) > 0:
                    type_corpus_dup.append(type_corpus[j])
                user_len = min(max_target_num, len(current_users))
                tag_len = min(max_target_num, len(current_tags))
                user_corpus_dup.append(current_users[:user_len])
                tag_corpus_dup.append(current_tags[:tag_len])
                if len(bid_score_corpus) > 0:
                    bid_score_corpus_dup.append(current_bid_score[:user_len])
                    current_bid_score = current_bid_score[ user_len: ]
                current_users = current_users[ user_len: ]
                current_tags = current_tags[ tag_len: ]
                len_corpus_dup.append( [user_len, tag_len] )
                num_repeat += 1
            num_repeat_corpus_dup += [num_repeat] * num_repeat
        
        print("Remove {} empty items with no label".format(num_no_label_item))

        return w_ind_corpus_dup, user_corpus_dup, tag_corpus_dup, num_repeat_corpus_dup, len_corpus_dup, type_corpus_dup, bid_score_corpus_dup
    
    w_ind_corpus_dup, user_corpus_dup, tag_corpus_dup, num_repeat_corpus_dup, len_corpus_dup, type_corpus_dup, bid_score_corpus_dup = duplicate_for_long_targets(save_idx, w_ind_corpus, type_corpus, user_corpus, tag_corpus, bid_score_corpus, args.max_target_num)
    
    store_type = torch.int32
    corpus_size = len(w_ind_corpus_dup)
    tensor_feature = torch.zeros(corpus_size, args.max_sent_len, dtype = store_type)
    if len(type_corpus) > 0:
        tensor_type = torch.zeros(corpus_size, args.max_sent_len, dtype = store_type)
    else:
        tensor_type = torch.zeros(0, dtype = store_type)
    if len(bid_score_corpus_dup) > 0:
        tensor_bid_score = torch.zeros(corpus_size, args.max_sent_len, dtype = store_type)
    else:
        tensor_bid_score = torch.zeros(0, dtype = store_type)
    tensor_user = torch.zeros(corpus_size, args.max_target_num, dtype = store_type)
    tensor_tag = torch.zeros(corpus_size, args.max_target_num, dtype = store_type)
    tensor_repeat_num = torch.zeros(corpus_size, dtype = store_type)
    tensor_user_len = torch.zeros(corpus_size, dtype = store_type)
    tensor_tag_len = torch.zeros(corpus_size, dtype = store_type)
    
    shuffled_order = list(range(corpus_size) )
    random.shuffle(shuffled_order)
    for i, j in enumerate(shuffled_order):
        if args.push_to_right:
            store_right = True
        else:
            store_right = False
        save_to_tesnor(w_ind_corpus_dup[j], tensor_feature, i, store_right)
        if len(type_corpus) > 0:
            save_to_tesnor(type_corpus_dup[j], tensor_type, i)
        if len(bid_score_corpus_dup) > 0:
            save_to_tesnor(bid_score_corpus_dup[j], tensor_bid_score, i)
        save_to_tesnor(user_corpus_dup[j], tensor_user, i)
        save_to_tesnor(tag_corpus_dup[j], tensor_tag, i)
        tensor_repeat_num[i] = num_repeat_corpus_dup[j]
        tensor_user_len[i] = len_corpus_dup[j][0]
        tensor_tag_len[i] = len_corpus_dup[j][1]
        #save_to_tesnor(num_repeat_corpus_dup[j], tensor_repeat_num, i)

    with open(output_save_file,'wb') as f_out:
        store_tensors(f_out, tensor_feature, tensor_type, tensor_user, tensor_tag, tensor_repeat_num, tensor_user_len, tensor_tag_len, tensor_bid_score)

def compose_dataset(output_dir, test_indicator, w_ind_corpus, type_corpus, user_corpus, tag_corpus):
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
    bid_score_corpus = []
    squeeze_into_tensors(train_idx, w_ind_corpus, type_corpus, user_corpus, tag_corpus, bid_score_corpus, output_dir + "/train.pt")
    squeeze_into_tensors(val_idx, w_ind_corpus, type_corpus, user_corpus, tag_corpus, bid_score_corpus, output_dir + "/val.pt")
    squeeze_into_tensors(test_idx, w_ind_corpus, type_corpus, user_corpus, tag_corpus, bid_score_corpus, output_dir + "/test.pt")


corpus_input_name = args.data_feature + args.input_file_name
corpus_user_name = args.data_user + args.input_file_name
corpus_tag_name = args.data_tag + args.input_file_name

#if max_ind >= 2147483648:
#    print("Will cause overflow")
#    sys.exit()


with open(corpus_input_name) as f_in:
    w_ind_corpus = load_w_ind(f_in, args.max_sent_num, args.max_sent_len)

with open(corpus_user_name) as f_in:
    user_corpus = load_w_ind(f_in, args.max_sent_num)

with open(corpus_tag_name) as f_in:
    tag_corpus = load_w_ind(f_in, args.max_sent_num)

if len(args.data_bid_score) > 0:
    corpus_bid_score_name = args.data_bid_score + args.input_file_name
    with open(corpus_bid_score_name) as f_in:
        bid_score_corpus = load_w_ind(f_in, args.max_sent_num)
else:
    bid_score_corpus = []

if len(args.data_type) > 0:
    corpus_type_name = args.data_type + args.input_file_name
    with open(corpus_type_name) as f_in:
        type_corpus = load_w_ind(f_in, args.max_sent_num, args.max_sent_len)
    assert len(w_ind_corpus) == len(type_corpus)
else:
    type_corpus = []

assert len(w_ind_corpus) == len(user_corpus)
assert len(w_ind_corpus) == len(tag_corpus)

if args.cv_fold_num == 0:
    all_idx = list(range(len(w_ind_corpus)))
    np.random.shuffle(all_idx)
    output_dir = os.path.dirname(args.save)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    squeeze_into_tensors(all_idx, w_ind_corpus, type_corpus, user_corpus, tag_corpus, bid_score_corpus, args.save)
else:
    cv_partition_idx_np = random_cv_partition(user_corpus, args.cv_fold_num)

    for k in range(args.cv_fold_num):
        #test_indicator = (cv_partition_idx_np == k)
        test_indicator = np.equal(cv_partition_idx_np, k).astype(int)
        print(test_indicator[:10])
        #print(cv_partition_idx_np[:10])
        output_dir = args.save + '_' + str(k)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        compose_dataset(output_dir, test_indicator, w_ind_corpus, type_corpus, user_corpus, tag_corpus)
        if args.only_first_fold:
            break
