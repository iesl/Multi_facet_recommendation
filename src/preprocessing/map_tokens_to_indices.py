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
parser.add_argument('--min_freq', type=int, default='5',
                    help='map to <unk> if observe less than this number')
parser.add_argument('--min_sent_length', type=int, default='5',
                    help='skip the sentence if sentence length is less than this number')
parser.add_argument('--max_sent_num', type=int, default='100000000000000',
                    help='load only this number of sentences from input corpus')
parser.add_argument('--lowercase', type=str2bool, nargs='?', default=False,
                    help='whether make all the words in corpus lowercased')

args = parser.parse_args()

print(args)

start_time = time.time()

if args.data[-3:] == '.gz':
    my_open = gzip.open
    byte_mode = True
else:
    my_open = open
    byte_mode = False


w_ind_corpus = []
        
dict_c = Dictionary(byte_mode)

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
            w_ind = dict_c.dict_check_add(w)
            w_ind_list.append(w_ind)
            total_num_w += 1
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

compact_mapping, total_freq_filtering = dict_c.densify_index(args.min_freq)
print("{}/{} tokens are filtered".format(total_freq_filtering, total_num_w) )

if not os.path.exists(args.save):
    os.makedirs(args.save)

corpus_output_name = args.save + "corpus_index"
dictionary_output_name = args.save + "dictionary_index"

with open(dictionary_output_name, 'w') as f_out:
    dict_c.store_dict(f_out)

with open(corpus_output_name, 'w') as f_out:
    for w_ind_list in w_ind_corpus:
         f_out.write(' '.join([str(compact_mapping[x]) for x in w_ind_list])+'\n')

elapsed = time.time() - start_time
print("time of total word to index: "+str(elapsed)+'s')
