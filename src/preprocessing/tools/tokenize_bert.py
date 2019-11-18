import sys
bert_dir = '/iesl/canvas/hschang/language_modeling/pytorch-pretrained-BERT'
#bert_dir = '/mnt/nfs/scratch1/hschang/language_modeling/pytorch-pretrained-BERT'
sys.path.insert(1, bert_dir)
from pytorch_pretrained_bert import BertTokenizer, BertModel

BERT_model_path = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(BERT_model_path, cache_dir = bert_dir + '/cache_dir/', do_lower_case = False)

input_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016.txt"
output_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_bert_tok.txt"

f_out = open(output_path, 'w')

with open(input_path) as f_in:
    for line in f_in:
        tokenized_text = tokenizer.tokenize('[CLS] ' + line + ' [SEP]')
        f_out.write(' '.join(tokenized_text)+'\n')
        
f_out.close()
