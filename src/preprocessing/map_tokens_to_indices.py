import argparse
import gzip
import os
import time
import sys

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
parser.add_argument('--max_sent_num', type=int, default='100000000000',
                    help='load only this number of sentences from input corpus')

args = parser.parse_args()

start_time = time.time()

if args.data[-3:] == '.gz':
    my_open = gzip.open
    byte_mode = True
else:
    my_open = open
    byte_mode = False

w_ind_corpus = []
w_d2_ind = {'[null]': 0, '<unk>': 1, '<eos>': 2}
ind_l2_w_freq = [ ['[null]',-1], ['<unk>',0], ['<eos>',0] ]
UNK_IND = 1
EOS_IND = 2

total_num_w = 0
filtered_sent_num = 0
with my_open(args.data, 'r') as f_in:
    for line in f_in:
        w_list_org = line.rstrip().split()
        if len(w_list_org) < args.min_sent_length:
            filtered_sent_num += 1
            continue
        w_ind_list = []
        for w in w_list_org:
            if w not in w_d2_ind:
                w_ind = len(w_d2_ind)
                w_d2_ind[w] = w_ind
                if byte_mode:
                    ind_l2_w_freq.append([w.decode('utf-8'), 1])
                else:
                    ind_l2_w_freq.append([w, 1])
            else:
                w_ind = w_d2_ind[w]
                ind_l2_w_freq[w_ind][1] += 1 
            w_ind_list.append(w_ind)
            total_num_w += 1
        w_ind_list.append(2) # append <eos>
        w_ind_corpus.append(w_ind_list)
        ind_l2_w_freq[EOS_IND][1] += 1
        if len(w_ind_corpus) % 1000000 == 0:
            print(len(w_ind_corpus))
            sys.stdout.flush()
        if len(w_ind_corpus) >= args.max_sent_num:
            break

print("total number of lines: "+str(len(w_ind_corpus)))
elapsed = time.time() - start_time
print("time of loading file: "+str(elapsed)+'s')

vocab_size = len(ind_l2_w_freq)
unk_ind_list = [False]*vocab_size

total_num_filtering = 0
total_freq_filtering = 0
for i, (w, w_freq) in enumerate(ind_l2_w_freq[3:]):
    if w_freq < args.min_sent_length:
        unk_ind_list[i] = True
        ind_l2_w_freq[i].append('unk')
        total_num_filtering += 1
        total_freq_filtering += w_freq

ind_l2_w_freq[UNK_IND][1] = total_freq_filtering #update <unk> frequency

print("{}/{} word types are filtered".format(total_num_filtering, vocab_size) )
print("{}/{} tokens are filtered".format(total_freq_filtering, total_num_w) )

if not os.path.exists(args.save):
    os.makedirs(args.save)

corpus_output_name = args.save + "corpus_index"
dictionary_output_name = args.save + "dictionary_index"

with open(dictionary_output_name, 'w') as f_out:
    for i in range(vocab_size):
        #print(ind_l2_w_freq[i])
        ind_l2_w_freq[i][1] = str(ind_l2_w_freq[i][1])
        f_out.write('\t'.join(ind_l2_w_freq[i])+'\n')

with open(corpus_output_name, 'w') as f_out:
    for w_ind_list in w_ind_corpus:
         f_out.write(' '.join([str(x) if not unk_ind_list[x] else str(UNK_IND) for x in w_ind_list])+'\n')

elapsed = time.time() - start_time
print("time of total word to index: "+str(elapsed)+'s')
