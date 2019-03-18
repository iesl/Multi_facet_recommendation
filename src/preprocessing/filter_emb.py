#import data

#data_path_1 = "./data/wikitext-2"
#data_path_1 = "./data/wikitext-103"
#data_path_1 = "./data/raw/binary-wackypedia-1-4-ukwac-.gz"
data_path_1 = "data/processed/wackypedia/dictionary_index"
#data_path_2 = "./data/penn"
#data_path_2 = "./data/penn"
#data_path_2 = "/iesl/canvas/xiangl/data/bookcorpus/books_large_p1.txt"
data_path_2 = "data/processed/bookp1/dictionary_index"
#emb_file_in_path = "/iesl/canvas/hschang/glove.840B.300d.txt"
#emb_file_out_path = "resources/glove.840B.300d_filtered_wac_bookp1.txt"
emb_file_in_path = "/iesl/canvas/hschang/GoogleNews-vectors-negative300.txt"
emb_file_out_path = "resources/Google-vec-neg300_filtered_wac_bookp1.txt"

#def find_word_dict(data_path):
#    corpus = data.Corpus(data_path)
#    return corpus.dictionary.word2idx

def load_word_dict(data_path):
    d = set()
    with open(data_path) as f_in:
        for line in f_in:
            fields = line.rstrip().split('\t')
            if len(fields) == 2:
                d.add(fields[0])
    return d

d1 = load_word_dict(data_path_1)
d2 = load_word_dict(data_path_2)

d1.update(d2)

def load_emb_file(emb_file):
    with open(emb_file) as f_in:
        word2emb = {}
        for line in f_in:
            word_val = line.rstrip().split(' ')
            word = word_val[0]
            #val = [float(x) for x word_val[1:]]
            val = word_val[1:]
            word2emb[word] = val
            #emb_size = len(val)
    return word2emb

word2emb = load_emb_file(emb_file_in_path)

with open(emb_file_out_path,'w') as f_out:
    for w in d1:
        if w in word2emb:
            f_out.write(w+' '+' '.join(word2emb[w])+'\n')
