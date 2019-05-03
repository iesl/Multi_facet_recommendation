#import data

#data_path_1 = "./data/wikitext-2"
#data_path_1 = "./data/wikitext-103"
#data_path_1 = "./data/raw/binary-wackypedia-1-4-ukwac-.gz"
#data_path_1 = "data/processed/wackypedia/dictionary_index"
#data_path_1 = "data/processed/wiki2016/dictionary_index"
#data_path_2 = "./data/penn"
#data_path_2 = "./data/penn"
#data_path_2 = "/iesl/canvas/xiangl/data/bookcorpus/books_large_p1.txt"
#data_path_2 = "data/processed/bookp1/dictionary_index"
data_path_2 = ""
#emb_file_in_path = "/iesl/data/word_embedding/glove.840B.300d.txt"
#emb_file_out_path = "resources/glove.840B.300d_filtered_wac_bookp1.txt"
#emb_file_out_path = "resources/glove.840B.300d_filtered_wiki2016.txt"
#emb_file_in_path = "/iesl/data/word_embedding/GoogleNews-vectors-negative300.txt"
#emb_file_out_path = "resources/Google-vec-neg300_filtered_wac_bookp1.txt"
#emb_file_out_path = "resources/Google-vec-neg300_filtered_wiki2016.txt"

import sys
import getopt
sys.path.insert(0, sys.path[0]+'/..')
import utils

help_msg = '-f <dict_file_path> -s <second_dict_path> -e <embedding_file_path> -o <output_file_path>'


try:
    opts, args = getopt.getopt(sys.argv[1:], "f:s:e:o:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-f"):
        data_path_1 = arg
    elif opt in ("-s"):
        data_path_2 = arg
    elif opt in ("-e"):
        emb_file_in_path = arg
    elif opt in ("-o"):
        emb_file_out_path = arg

#def find_word_dict(data_path):
#    corpus = data.Corpus(data_path)
#    return corpus.dictionary.word2idx

def load_word_dict(data_path):
    d = set()
    with open(data_path) as f_in:
        for line in f_in:
            fields = line.rstrip().split('\t')
            #fields = line.rstrip().split(' ')
            if len(fields) == 3:
                d.add(fields[0])
    return d

d1 = load_word_dict(data_path_1)

if len(data_path_2) > 0:
    d2 = load_word_dict(data_path_2)
    d1.update(d2)

#def load_emb_file(emb_file):
#    with open(emb_file) as f_in:
#        word2emb = {}
#        for line in f_in:
#            word_val = line.rstrip().split(' ')
#            word = word_val[0]
#            #val = [float(x) for x word_val[1:]]
#            val = word_val[1:]
#            word2emb[word] = val
#            #emb_size = len(val)
#    return word2emb

word2emb, emb_size = utils.load_emb_file(emb_file_in_path, convert_np = False)

with open(emb_file_out_path,'w') as f_out:
    for w in d1:
        if w in word2emb:
            f_out.write(w+' '+' '.join(word2emb[w])+'\n')
