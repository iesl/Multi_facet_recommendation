import gensim
from scipy.spatial import distance
from scipy.stats import spearmanr

#emb_file_path = "/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/resources/Google-vec-neg300_filtered_wac_bookp1.txt"
#emb_file_path = "/iesl/data/word_embedding/GoogleNews-vectors-negative300.txt"
#emb_file_path = "/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/resources/glove.840B.300d_filtered_wac_bookp1.txt"
#emb_file_path = "./models/Wacky-20190329-154753/target_emb.txt"
#emb_file_path = "./models/wiki2016-20190409-112932/target_emb.txt"
emb_file_path = "/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/resources/Google-vec-neg300_filtered_wiki2016.txt"
ratings_file = "dataset_testing/SCWS/ratings.txt"

w2v = gensim.models.KeyedVectors.load_word2vec_format(emb_file_path, binary=False)

gt_rating = []
pred_rating = []

with open(ratings_file) as f_in:
    oov_query_count = 0
    for line in f_in:
        fields = line.rstrip().split('\t')
        w1 = fields[1]
        w2 = fields[3]
        rating = float(fields[-11])
        try:
            global_emb1 = w2v.wv[w1]
            global_emb2 = w2v.wv[w2]
        except:
            oov_query_count += 1
            continue
        sim = 1 - distance.cosine(global_emb1, global_emb2)
        gt_rating.append(rating)
        pred_rating.append(sim)
print(oov_query_count)
print(spearmanr(gt_rating, pred_rating))
