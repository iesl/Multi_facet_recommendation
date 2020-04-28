import sys
sys.path.insert(0, sys.path[0]+'/..')
import getopt
from scibert.tokenization_bert import BertTokenizer
model_name = 'scibert-scivocab-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(model_name)

help_msg = '-i <input_path> -c <out_corpus_path> -d <out_dict_path>' 

out_corpus_path = ''
out_dict_path = ''

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:c:d:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-i"):
        input_path = arg
    elif opt in ("-c"):
        out_corpus_path = arg
    elif opt in ("-d"):
        out_dict_path = arg


corpus_list = []
idx_d2_freq = {}
with open(input_path) as f_in:
    for line in f_in:
        tokenized_text = line.rstrip()
        indexed_tokens = bert_tokenizer.convert_tokens_to_ids(tokenized_text.split())
        #print(tokenized_text, indexed_tokens)
        for idx in indexed_tokens:
            freq = idx_d2_freq.get(idx,0)
            idx_d2_freq[idx] = freq + 1
        corpus_list.append(' '.join(map(str,indexed_tokens)))

if len(out_corpus_path) > 0:
    with open(out_corpus_path, 'w') as f_out:
        f_out.write( '\n'.join(corpus_list) )

if len(out_dict_path) > 0:
    vocab_dict = bert_tokenizer.vocab
    idx_l2_token = [''] * len(vocab_dict)
    for word in vocab_dict:
        idx = vocab_dict[word]
        idx_l2_token[idx] = word
    with open(out_dict_path, 'w') as f_out:
        for idx, freq in sorted(idx_d2_freq.items(), key = lambda x: x[1], reverse = True):
            word = idx_l2_token[idx]
            f_out.write(word+'\t'+str(freq)+'\t'+str(idx)+ '\n')
