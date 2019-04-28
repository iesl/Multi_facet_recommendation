import argparse
import torch
import sys
import random
import os
sys.path.insert(0, sys.path[0]+'/..')
#sys.path.append("..")
import utils

#handle the min target filtering (If more than 30 words in output, just do random sampling)
#padding and store them into tensors, (random shuffle? two sets), store train, val, and test

parser = argparse.ArgumentParser(description='Preprocessing step 2')
parser.add_argument('--data', type=str, default='./data/processed/wiki2016_nchunk_lower_min100/',
                    help='location of the data corpus')
parser.add_argument('--save', type=str, default='./data/processed/wiki2016_nchunk_lower_min100/tensors/',
                    help='path to save the output data')
parser.add_argument('--max_sent_len', type=int, default=6,
                    help='max sentence length for input features')
parser.add_argument('--window_size', type=int, default=5,
                    help='max word number for output prediction w/o stop words (including above and below sentences)')
parser.add_argument('--max_sent_num', type=int, default='100000000000000',
#parser.add_argument('--max_sent_num', type=int, default='10000',
                    help='load only this number of sentences from input corpus')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

#parser.add_argument('--stop_word_file', type=str, default='./resources/stop_word_list',
#                    help='path to the file of a stop word list')

args = parser.parse_args()

print(args)

random.seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

max_target_num = args.window_size * 2
        
def load_save_w_ind(f_in, max_sent_num, max_sent_len):
    #w_ind_feature = []
    #w_ind_target = []
    #last_sent = ''
    #num_duplicated_sent = 0
    num_too_long_sent = 0
    num_unk = 0
    is_feature = 1
    output_i = 0

    all_lines = f_in.readlines()
    assert len(all_lines) % 2 == 0
    corpus_size = len(all_lines)/2

    print("Allocating {} bytes".format( corpus_size*(max_target_num+max_sent_len)*4 ) )
    store_type = torch.int32
    all_features = torch.zeros(corpus_size,max_sent_len,dtype = store_type)
    all_targets = torch.zeros(corpus_size,max_target_num,dtype = store_type)
    

    for line in all_lines:
        current_sent = line.rstrip()
        #if current_sent == last_sent:
        #    num_duplicated_sent += 1
        #    continue
        #last_sent = current_sent
        fields = current_sent.split(' ')
        if is_feature:
            if len(fields) > max_sent_len:
                num_too_long_sent += 1
                #continue #I guess truncation will perform better than discard
            feature_list = [int(x) for x in fields]
            current_len = min(args.max_sent_len,len(feature_list))
            all_features[output_i,-current_len:] = torch.tensor(feature_list[-current_len:],dtype = store_type)
            #w_ind_feature.append([int(x) for x in fields])
            is_feature = 0
            if len(w_ind_feature) % 1000000 == 0:
                print(len(w_ind_feature))
                sys.stdout.flush()
        else:
            assert len(fields) <= max_target_num + 1
            target_list = [int(x) for x in fields[:-1]] #we should skip <eos>
            all_targets[output_i, :len(target_list) ] = torch.tensor(target_list, dtype = store_type)
            #w_ind_target.append([int(x) for x in fields])
            if len(w_ind_target) > max_sent_num:
                break
            is_feature = 1
            output_i += 1
    print( "Finish loading {} phrases. While having {} long phrases".format(len(w_ind_feature),num_too_long_sent) )
    return all_features, all_targets

corpus_input_name = args.data + "corpus_index"
#dictionary_input_name = args.data + "dictionary_index"

#with open(dictionary_input_name) as f_in:
#    w_d2_ind_freq, max_ind = utils.load_word_dict(f_in)

#with open(dictionary_input_name) as f_in:
#    idx2word_freq = utils.load_idx2word_freq(f_in)

#max_ind = len(idx2word_freq)

#if max_ind >= 2147483648:
#    print("Will cause overflow")
#    sys.exit()


with open(corpus_input_name) as f_in:
    all_features, all_targets = load_save_w_ind(f_in, args.max_sent_num, args.max_sent_len)

#corpus_size = len(w_ind_feature)
#args.max_target_num+args.max_sent_len


#for i in range(0,len(w_ind_feature)):
#    output_i = i
#    feature_list = w_ind_feature[i]
#    current_len = min(args.max_sent_len,len(feature_list))
#
#    all_features[output_i,-current_len:] = torch.tensor(feature_list[-current_len:],dtype = store_type)
#    target_w_list = w_ind_target[i][:-1] #we should skip <eos>
#    all_targets[output_i, :len(target_w_list) ] = torch.tensor(target_w_list, dtype = store_type)
    
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

#del w_ind_feature
#del w_ind_target

#print("{} / {} needs to randomly select targets".format(random_selection_num,corpus_size))

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
