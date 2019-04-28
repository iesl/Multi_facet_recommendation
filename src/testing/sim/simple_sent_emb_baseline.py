import gensim
from gensim.models import KeyedVectors
from scipy.spatial import distance
from scipy.stats import pearsonr
import numpy as np

input_path = "./dataset_testing/STS/stsbenchmark/sts-dev_org"
#emb_file_path = "./resources/word2vec_wiki2016_min100_d400_w5.txt"
#emb_file_path = "./resources/Google-vec-neg300_filtered_wiki2016_min500.txt"
#emb_file_path = "./resources/Google-vec-neg300_filtered_wiki2016_min100.txt"
#emb_file_path = "./resources/Google-vec-neg300_filtered_wiki2016.txt"
#emb_file_path = "./resources/word2vec_wiki2016_min100_filtered_google.txt"
#emb_file_path = "./resources/lexvec.commoncrawl.ngramsubwords.300d.W.pos.vectors"
emb_file_path = "./resources/lexvec.enwiki+newscrawl.300d.W.pos.vectors"
#emb_file_path = "./resources/paragram_300_ws353/paragram_300_ws353.txt"
#emb_file_path = "./resources/paragram_300_sl999/paragram_300_sl999.txt"
#emb_file_path = "./resources/apnews_dbow/doc2vec.bin"
gt_file_name = "./dataset_testing/STS/stsbenchmark/sts-dev.csv"

#lower_words = True
lower_words = False

print(emb_file_path)

#w2v = gensim.models.KeyedVectors.load_word2vec_format(emb_file_path, binary=False, unicode_errors='ignore' )
w2v = KeyedVectors.load_word2vec_format(emb_file_path, binary=False, unicode_errors='ignore' )
#w2v = gensim.models.Word2Vec.load(emb_file_path)

def load_STS_file(f_in):
    output_list = []
    for line in f_in:
        #print(line.rstrip().split('\t'))
        fields = line.rstrip().split('\t')
        genre, source, source_year, org_idx, score, sent_1, sent_2 = fields[:7]
        output_list.append([sent_1, sent_2, float(score), genre+'-'+source])
    return output_list

with open(gt_file_name) as f_in:
    testing_list = load_STS_file(f_in)

#org_sent_list = []
#proc_sent_list = []
org_sent_d2_proc_sent = {}
with open(input_path) as f_in:
    for line in f_in:
        org_sent, proc_sent = line.rstrip().split('\t')
        org_sent_d2_proc_sent[org_sent] = proc_sent
        #org_sent_list.append(org_sent)
        #proc_sent_list.append(proc_sent)

emb_size = w2v.vector_size
oov_count = 0
source_d2_info = {}

def sent_to_emb(org_sent_d2_proc_sent, sent, w2v):
    global oov_count
    sent_proc = org_sent_d2_proc_sent[sent]
    w_list = sent_proc.split()
    sent_emb = np.zeros(emb_size)
    oov_count_sent = 0
    for w in w_list:
        if lower_words:
            w = w.lower()
        if w in w2v.wv:
            #if np.sum(np.isnan(w2v.wv[w])) > 0:
            if np.sum(w2v.wv[w]) == 0:
                print(w)
                oov_count += 1
                oov_count_sent += 1
                continue
            sent_emb += w2v.wv[w]
        else:
            oov_count += 1
            oov_count_sent += 1
    return sent_emb, oov_count_sent

gt_list = []
pred_list = []
for sent_1, sent_2, score_gt, source in testing_list:
    sent_1_emb, oov_count_sent_1 = sent_to_emb(org_sent_d2_proc_sent, sent_1, w2v)
    sent_2_emb, oov_count_sent_2 = sent_to_emb(org_sent_d2_proc_sent, sent_2, w2v)
    if np.sum(sent_1_emb) != 0 and np.sum(sent_2_emb) != 0:
        score_pred = 1 - distance.cosine(sent_1_emb, sent_2_emb)
    else:
        score_pred = 1
        print(sent_1)
        print(sent_2)
        print(sent_1_emb)
        print(sent_2_emb)
    if source not in source_d2_info:
        source_d2_info[source] = [ 0, [], [] ]
    source_d2_info[source][0] += oov_count_sent_1 + oov_count_sent_2
    source_d2_info[source][1].append(score_pred)
    source_d2_info[source][2].append(score_gt)
    pred_list.append(score_pred)
    gt_list.append(score_gt)

print(pearsonr(pred_list,gt_list))
print(oov_count)

for source in source_d2_info:
    print(source)
    print(source_d2_info[source][0])
    print(pearsonr(source_d2_info[source][1],source_d2_info[source][2]))
