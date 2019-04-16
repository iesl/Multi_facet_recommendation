import gensim

# Load Google's pre-trained Word2Vec model.
model_wv = gensim.models.KeyedVectors.load_word2vec_format('/iesl/canvas/hschang/GoogleNews-vectors-negative300.bin', binary=True)
model_wv.save_word2vec_format("/iesl/canvas/hschang/GoogleNews-vectors-negative300.txt")
