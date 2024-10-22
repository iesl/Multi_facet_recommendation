import json
import pandas as pd
import os
import sys
import csv
import getopt

help_msg = '-i <paper_data_path> -m <meta_ml_path> -o <output_path>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:m:o:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-i"):
        paper_data_path = arg
    elif opt in ("-m"):
        meta_ml_path = arg
    elif opt in ("-o"):
        output_path = arg

tokenizer_mode = 'scapy'
#tokenizer_mode = 'scibert'

if tokenizer_mode == 'scapy':
    from spacy.lang.en import English
    nlp = English()
else:
    import sys
    sys.path.insert(0, sys.path[0]+'/../..')
    from scibert.tokenization_bert import BertTokenizer
    model_name = 'scibert-scivocab-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)

csv.field_size_limit(sys.maxsize)

#meta_ml_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_ml_ext.tsv'
#paper_data_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/paper_data_ml_ext.json'
#output_path = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/gorc/all_paper_data'
##output_path = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/gorc/all_paper_data_scibert'
##meta_ml_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_ml_ext_org.tsv'
##paper_data_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/paper_data_ml_ext_org.json'
##output_path = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/gorc/all_paper_org_data'

def parse_list_str(list_in):
    return list_in.replace('[','').replace(']','').replace("'",'').replace('"','').split(', ')

meta_ml_df = pd.read_csv(meta_ml_path, delimiter='\t', engine='python', quoting=csv.QUOTE_NONE )

with open(paper_data_path) as f_in:
    id_title_abstract_author_list = json.load(f_in)
print(len(id_title_abstract_author_list))

id_d2_title_abstract = {}
for pid, title, abstract, author in id_title_abstract_author_list:
    id_d2_title_abstract[pid] = [title, abstract]

meta_ml_df.authors = meta_ml_df.authors.apply(parse_list_str)
meta_ml_df.inbound_citations = meta_ml_df.inbound_citations.apply(parse_list_str)

output_list = []
for index, row in meta_ml_df.iterrows():
    pid = row['pid']
    title, abstract = id_d2_title_abstract[pid]
    inbound = row['inbound_citations']
    authors = row['authors']
    if tokenizer_mode == 'scapy':
        w_list_title = [w.text for w in nlp.tokenizer( title ) ] + ['<SEP>']
    elif tokenizer_mode == 'scibert':
        w_list_title = tokenizer.tokenize('[CLS] ' + title + ' [SEP]')
    #w_list_title = [w.text for w in nlp.tokenizer( title ) ] + ['<SEP>']
    w_list_title = ' '.join(w_list_title).split()
    if abstract is not None:
        if tokenizer_mode == 'scapy':
            w_list_abstract = [w.text for w in nlp.tokenizer( abstract ) ] + ['<SEP>']
        elif tokenizer_mode == 'scibert':
            w_list_abstract = tokenizer.tokenize('[CLS] ' + abstract + ' [SEP]') #We will finetune scibert, so don't need to use sentence segmentation
        #w_list_abstract = [w.text for w in nlp.tokenizer( abstract ) ] + ['<SEP>']
        w_list_abstract = ' '.join(w_list_abstract).split()
    else:
        w_list_abstract = []

    type_list = ['0']*len(w_list_title) + ['1']*len(w_list_abstract)
    output_list.append([' '.join(w_list_title + w_list_abstract), ' '.join(type_list), ','.join(authors), ','.join(inbound)])

with open(output_path, 'w') as f_out:
    for output in output_list:
        f_out.write('\t'.join(output)+'\n')
