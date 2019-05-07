from gensim.models import word2vec
import time
input_data_path='./data/raw/wiki2016.txt'
#output_data_path='./resources/word2vec_wiki2016_min100.txt'
#output_data_path='./resources/word2vec_wiki2016_min100_d400_w5.txt'
#output_data_path='./resources/word2vec_wiki2016_min100_d300_w5_s5.txt'
output_data_path='./resources/word2vec_sg_wiki2016_lower_min100_w5_s5.txt'
lowercase = True

data = []
#max_sent_num=512000
max_sent_num=10000000000000
with open(input_data_path) as f_in:
    for line in f_in:
        sent_str = line.rstrip()
        if lowercase:
            sent_str = sent_str.lower()
        w_list = sent_str.split()
        data.append(w_list)
        if len(data)>max_sent_num:
            break
print data[:10]
emb_dim = 300
#emb_dim = 400

print "training word2vec"
t = time.time()
#model = word2vec.Word2Vec(data, size=emb_dim, window=10, min_count=100, workers=8, iter=5)
#model = word2vec.Word2Vec(data, size=emb_dim, window=5, min_count=100, workers=15, iter=10, sample=1e-5)
model = word2vec.Word2Vec(data, sg = 1, size=emb_dim, window=5, min_count=100, workers=15, iter=10, sample=1e-5)
model.wv.save_word2vec_format(output_data_path, binary=False)
elapsed = time.time() - t
print "total spent time", elapsed
