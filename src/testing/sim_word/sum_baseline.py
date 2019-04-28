import gensim
from scipy.spatial import distance

embedding_dir = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/resources/"

#lowercase = False
embedding_file_name = embedding_dir + "glove.840B.300d_filtered_wiki2016_min100.txt"
#embedding_file_name = embedding_dir + "lexvec_wiki2016_min100"
#embedding_file_name = embedding_dir + "word2vec_wiki2016_min100.txt"

#lower
#embedding_file_name = embedding_dir + "lexvec_enwiki_wiki2016_min100"
#embedding_file_name = embedding_dir + "paragram_wiki2016_min100"
#embedding_file_name = embedding_dir + "glove.42B.300d_filtered_wiki2016_nchunk_lower_min100.txt"

#w2v = gensim.models.KeyedVectors.load_word2vec_format(emb_file_path, binary=False)

dataset_dir = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/"
dataset_list = [ [dataset_dir + "SemEval2013/en.trainSet", "SemEval2013" ], [dataset_dir + "SemEval2013/en.testSet", "SemEval2013"], [dataset_dir + "Turney2012/jair.data", "Turney"] ]


