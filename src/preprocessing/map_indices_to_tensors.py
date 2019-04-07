import argparse
import torch
import sys
import random
import os
sys.path.insert(0, sys.path[0]+'/..')
#sys.path.append("..")
import utils

#remove the duplicated sentences
#remove the sentences which are too long
#remove stop words in target
#handle the min target filtering (If more than 30 words in output, just do random sampling)
#padding and store them into tensors, (random shuffle? two sets), store train, val, and test

parser = argparse.ArgumentParser(description='Preprocessing step 2')
parser.add_argument('--data', type=str, default='./data/processed/wackypedia/',
                    help='location of the data corpus')
parser.add_argument('--save', type=str, default='./data/processed/wackypedia/tensors/',
#parser.add_argument('--save', type=str, default='./data/processed/wackypedia/tensors_multi150/',
                    help='path to save the output data')
parser.add_argument('--max_sent_len', type=int, default=50,
#parser.add_argument('--max_sent_len', type=int, default=150,
                    help='max sentence length for input features')
#parser.add_argument('--multi_sent', default=False, action='store_true',
parser.add_argument('--multi_sent', default=False, 
                    help='Whether do we want to cram multiple sentences into one input feature')
parser.add_argument('--max_target_num', type=int, default=30,
                    help='max word number for output prediction w/o stop words (including above and below sentences)')
parser.add_argument('--max_sent_num', type=int, default='100000000000',
                    help='load only this number of sentences from input corpus')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

parser.add_argument('--stop_word_file', type=str, default='./resources/stop_word_list',
                    help='path to the file of a stop word list')

args = parser.parse_args()

print(args)

random.seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

def convert_stop_to_ind(f_in, w_d2_ind_freq):
    stop_word_set = set()
    for line in f_in:
        w = line.rstrip()
        if w in w_d2_ind_freq:
            stop_word_set.add(w_d2_ind_freq[w][0])
    return stop_word_set

def convert_stop_to_ind_lower(f_in, idx2word_freq):
    stop_word_org_set = set()
    for line in f_in:
        w = line.rstrip()
        stop_word_org_set.add(w)
    stop_word_set = set()
    for idx, (w, freq) in enumerate(idx2word_freq):
        if w.lower() in stop_word_org_set:
            stop_word_set.add(idx)
    return stop_word_set
        
def load_w_ind(f_in, max_sent_num, max_sent_len):
    w_ind_corpus = []
    last_sent = ''
    num_duplicated_sent = 0
    num_too_long_sent = 0

    for line in f_in:
        current_sent = line.rstrip()
        if current_sent == last_sent:
            num_duplicated_sent += 1
            continue
        last_sent = current_sent
        fields = current_sent.split(' ')
        if len(fields) > max_sent_len:
            num_too_long_sent += 1
            continue
        w_ind_corpus.append([int(x) for x in fields])
        if len(w_ind_corpus) % 1000000 == 0:
            print(len(w_ind_corpus))
            sys.stdout.flush()
        if len(w_ind_corpus) > max_sent_num:
            break
    print( "Finish loading {} sentences. While removing {} duplicated and {} long sentences".format(len(w_ind_corpus),num_duplicated_sent, num_too_long_sent) )
    return w_ind_corpus

corpus_input_name = args.data + "corpus_index"
dictionary_input_name = args.data + "dictionary_index"

#with open(dictionary_input_name) as f_in:
#    w_d2_ind_freq, max_ind = utils.load_word_dict(f_in)

with open(dictionary_input_name) as f_in:
    idx2word_freq = utils.load_idx2word_freq(f_in)

max_ind = len(idx2word_freq)

if max_ind >= 2147483648:
    print("Will cause overflow")
    sys.exit()

store_type = torch.int32

with open(args.stop_word_file) as f_in:
    #stop_ind_set = convert_stop_to_ind(f_in, w_d2_ind_freq)
    stop_ind_set = convert_stop_to_ind_lower(f_in, idx2word_freq)

with open(corpus_input_name) as f_in:
    w_ind_corpus = load_w_ind(f_in, args.max_sent_num, args.max_sent_len)

corpus_size = len(w_ind_corpus)-2
#args.max_target_num+args.max_sent_len
print("Allocating {} bytes".format( corpus_size*(args.max_target_num+args.max_sent_len)*4 ) )
all_features = torch.zeros(corpus_size,args.max_sent_len,dtype = store_type)
all_targets = torch.zeros(corpus_size,args.max_target_num,dtype = store_type)

random_selection_num = 0

for i in range(1,len(w_ind_corpus)-1):
    output_i = i - 1
    if args.multi_sent:
        current_len = 0
        feature_list = []
        for j in range(i,len(w_ind_corpus)-1):
            w_ind_list = w_ind_corpus[j][:-1] #excluding <eos>
            sent_len = len(w_ind_list)
            current_len_prev = current_len
            current_len += sent_len
            if current_len > args.max_sent_len - 1:
                break
            feature_list += w_ind_list
        if current_len > args.max_sent_len - 1:
            current_len = current_len_prev
        feature_list.append(w_ind_corpus[j-1][-1])
        next_sent_ind = j
    else:
        feature_list = w_ind_corpus[i]
        current_len = len(feature_list) - 1
        next_sent_ind = i + 1

    all_features[output_i,-(current_len+1):] = torch.tensor(feature_list,dtype = store_type)
    prev_w_ind_list = w_ind_corpus[i-1]
    next_w_ind_list = w_ind_corpus[next_sent_ind]
    target_w_list = []
    for w in prev_w_ind_list+next_w_ind_list:
        if w not in stop_ind_set:
            target_w_list.append(w)
    if len(target_w_list) <= args.max_target_num:
        all_targets[output_i, :len(target_w_list) ] = torch.tensor(target_w_list, dtype = store_type)
    else:
        all_targets[output_i,:] = torch.tensor( random.sample(target_w_list, args.max_target_num), dtype = store_type)
        random_selection_num += 1
    
#for i in range(1,len(w_ind_corpus)-1):
#    output_i = i - 1
#    w_ind_list = w_ind_corpus[i]
#    sent_len = len(w_ind_list)
#    all_features[output_i,-sent_len:] = torch.tensor(w_ind_list,dtype = store_type)
#    prev_w_ind_list = w_ind_corpus[i-1]
#    next_w_ind_list = w_ind_corpus[i+1]
#    target_w_list = []
#    for w in prev_w_ind_list+next_w_ind_list:
#        if w not in stop_ind_set:
#            target_w_list.append(w)
#    if len(target_w_list) <= args.max_target_num:
#        all_targets[output_i, :len(target_w_list) ] = torch.tensor(target_w_list, dtype = store_type)
#    else:
#        all_targets[output_i,:] = torch.tensor( random.sample(target_w_list, args.max_target_num), dtype = store_type)
#        random_selection_num += 1

del w_ind_corpus

print("{} / {} needs to randomly select targets".format(random_selection_num,corpus_size))

#print("Finish loading all files")

training_output_name = args.save + "train.pt"
val_org_output_name = args.save + "val_org.pt"
test_org_output_name = args.save + "test_org.pt"
val_shuffled_output_name = args.save + "val_shuffled.pt"
test_shuffled_output_name = args.save + "test_shuffled.pt"

testing_size_ratio = 0.05
testing_size = int(corpus_size * testing_size_ratio)
print("Testing size: {}".format(testing_size))

def store_tensors(f_out,tensor1,tensor2):
    torch.save([tensor1,tensor2],f_out)

with open(test_org_output_name,'wb') as f_out:
    store_tensors(f_out,all_features[-testing_size:,:].clone(),all_targets[-testing_size:,:].clone())

with open(val_org_output_name,'wb') as f_out:
    store_tensors(f_out,all_features[-2*testing_size:-testing_size,:].clone(),all_targets[-2*testing_size:-testing_size,:].clone())

rest_size = corpus_size - 2*testing_size
shuffle_ind = list(range(rest_size))
random.shuffle(shuffle_ind)

with open(test_shuffled_output_name,'wb') as f_out:
    store_ind = shuffle_ind[-testing_size:]
    store_tensors(f_out,all_features[store_ind,:],all_targets[store_ind,:])

with open(val_shuffled_output_name,'wb') as f_out:
    store_ind = shuffle_ind[-2*testing_size:-testing_size]
    store_tensors(f_out,all_features[store_ind,:],all_targets[store_ind,:])

with open(training_output_name,'wb') as f_out:
    store_ind = shuffle_ind[:rest_size-2*testing_size]
    print("Training size: {}".format(len(store_ind)))
    store_tensors(f_out,all_features[store_ind,:],all_targets[store_ind,:])
