import argparse
import gzip
import os
import time
import sys
sys.path.insert(0, sys.path[0]+'/..')
from utils import Dictionary, str2bool


#map words to index (and set the max sentence number), 
#map low freq words into <unk>
#add end of sentence tokens (for transformer to generate output embedding), 
#output dataset, dictionary (start with <null>, <unk>, and <eos>), word frequency,  print total number of words, total number of filtered words

parser = argparse.ArgumentParser(description='Preprocessing step 1')
parser.add_argument('--data', type=str, default='./data/raw/binary-wackypedia-1-4-ukwac-.gz',
                    help='location of the data corpus')
#/iesl/canvas/xiangl/data/bookcorpus/books_large_p1.txt
parser.add_argument('--save', type=str, default='./data/processed/wackypedia/',
                    help='path to save the output data')
parser.add_argument('--min_freq', type=int, default='0',
                    help='map to <unk> if observe less than this number')
parser.add_argument('--min_sent_length', type=int, default='-1',
                    help='skip the sentence if sentence length is less than this number')
parser.add_argument('--max_sent_num', type=int, default='100000000000000',
                    help='load only this number of sentences from input corpus')
parser.add_argument('--lowercase', type=str2bool, nargs='?', default=False,
                    help='whether make all the words in corpus lowercased')
parser.add_argument('--eos', type=str2bool, nargs='?', default=False,
                    help='whether append eos')
parser.add_argument('--ignore_unk', type=str2bool, nargs='?', default=False,
                    help='when lower than min_freq, do not append unk')
parser.add_argument('--input_vocab', type=str, default='',
                    help='input an exiting vocab dictionary')
parser.add_argument('--update_dict', type=str2bool, nargs='?', default=True,
                    help='Whether we want to update the dictionary if we encounter new words')
parser.add_argument('--output_file_name', type=str, default='corpus_index',
                    help='The file name store the indices')

args = parser.parse_args()

print(args)

if len(args.input_vocab) == 0:
    assert args.update_dict

if len(args.input_vocab) > 0:
    assert args.min_freq == 0
    if 'meta' in args.data:
        if args.lowercase:
            assert '_uncased' in args.input_vocab
        else:
            assert '_cased' in args.input_vocab
        
start_time = time.time()

if args.data[-3:] == '.gz':
    my_open = gzip.open
    byte_mode = True
else:
    my_open = open
    byte_mode = False


w_ind_corpus = []

dict_c = Dictionary(byte_mode)
if len(args.input_vocab) > 0:
    with open(args.input_vocab) as f_in:
        dict_c.load_dict(f_in)

total_num_w = 0
filtered_sent_num = 0
with my_open(args.data, 'r') as f_in:
    for line in f_in:
        w_list_org = line.rstrip().split()
        if len(w_list_org) > 0 and len(w_list_org) < args.min_sent_length:
            filtered_sent_num += 1
            continue
        w_ind_list = []
        for w in w_list_org:
            if args.lowercase:
                w = w.lower()
            if args.update_dict:
                w_ind = dict_c.dict_check_add(w)
            else:
                w_ind = dict_c.dict_check(w, args.ignore_unk)
                #if w_ind == 0:
                #    print(w)
                
            w_ind_list.append(w_ind)
            total_num_w += 1
        if args.eos:
            dict_c.append_eos(w_ind_list)
        w_ind_corpus.append(w_ind_list)
        if len(w_ind_corpus) % 1000000 == 0:
            print(len(w_ind_corpus))
            sys.stdout.flush()
        if len(w_ind_corpus) >= args.max_sent_num:
            break

print("total number of lines: "+str(len(w_ind_corpus)))
elapsed = time.time() - start_time
print("time of loading file: "+str(elapsed)+'s')

if len(args.input_vocab) == 0:
    compact_mapping, total_freq_filtering = dict_c.densify_index(args.min_freq, args.ignore_unk)
    print("{}/{} tokens are filtered".format(total_freq_filtering, total_num_w) )
else:
    compact_mapping = list(range( len(dict_c.ind_l2_w_freq) ))


if args.update_dict:
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    corpus_output_name = args.save + args.output_file_name
    dictionary_output_name = args.save + "dictionary_index"

    with open(dictionary_output_name, 'w') as f_out:
        dict_c.store_dict(f_out)
else:
    corpus_output_name = args.save


with open(corpus_output_name, 'w') as f_out:
    for w_ind_list in w_ind_corpus:
         f_out.write(' '.join([str(compact_mapping[x]) for x in w_ind_list if compact_mapping[x] > 0])+'\n')

elapsed = time.time() - start_time
print("time of total word to index: "+str(elapsed)+'s')
