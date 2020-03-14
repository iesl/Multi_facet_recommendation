import re
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

import sys
sys.path.insert(0, sys.path[0]+'/../..')
from scibert.tokenization_bert import BertTokenizer

nlp = English()
#tokenizer = Tokenizer(nlp.vocab)

#lowercase = True
lowercase = False

convert_numbers = True
#convert_numbers = False

num_token = "<NUM>"

file_format = 'a'
#file_format = 't'

scibert_tokenization = True
if scibert_tokenization:
    model_name = 'scibert-scivocab-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)

if file_format == 'a':
    file_path = "/iesl/canvas/hschang/recommendation/ctrsr_datasets/citeulike-a/raw-data.csv"
    if scibert_tokenization:
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/citeulike-a/paper_text_idx_scibert"
    else:
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/citeulike-a/paper_text"
else:
    file_path = "/iesl/canvas/hschang/recommendation/ctrsr_datasets/citeulike-t/rawtext.dat"
    if scibert_tokenization:
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/citeulike-t/paper_text_idx_scibert"
    else:
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/citeulike-t/paper_text"

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def tokenize_paper(input_text):
    w_list_org = [w.text for w in nlp.tokenizer( input_text.replace('{','').replace('}','') ) ]
    if convert_numbers:
        w_list = []
        for w in w_list_org:
            if is_number(w):
                w_list.append(num_token)
            else:
                w_list.append(w)
    else:
        w_list = w_list_org
    output_text = ' '.join(w_list)
    if lowercase:
        output_text = out_sent.lower()
    return output_text

def tokenize_paper_scibert(input_text):
    proc_text = input_text.replace('{','').replace('}','')
    w_idx_list = tokenizer.encode(proc_text)
    output_text = ' '.join([str(x) for x in w_idx_list])
    return output_text
#filter_set = set(['<b>','</b>','</font','style>'])
#
#def cleanhtml(raw_html):
#    cleanr = re.compile('<.*?>')
#    cleantext = re.sub(cleanr, '', raw_html)
#    return cleantext

output_list = []

if file_format == 'a':
    import pandas as pd
    df = pd.read_csv(file_path, encoding = 'ISO-8859-1')
    title_list = df['raw.title'].tolist()
    abstract_list = df['raw.abstract'].tolist()
    for title, abstract in zip(title_list, abstract_list):
        if scibert_tokenization:
            out_sent = tokenize_paper_scibert(title + ' ' + abstract)
        else:
            out_sent = tokenize_paper(title + ' ' + abstract)
        output_list.append(out_sent)

else:
    with open(file_path) as f_in:
        for i, line in enumerate(f_in):
            if i % 2 == 0:
                assert line[0] == '#'
                continue
            #w_list = cleanhtml(line.rstrip()).split()
            #print(w_list)
            #sent_filtered = ' '.join([x for x in w_list if x not in filter_set])
            if scibert_tokenization:
                out_sent = tokenize_paper_scibert(line.rstrip())
            else:
                out_sent = tokenize_paper(line.rstrip())
            output_list.append(out_sent)

with open(output_path,'w') as f_out:
    for lemma_sent in output_list:
        f_out.write(lemma_sent+'\n')
        
    #for org_sent, lemma_sent in output_list:
    #    f_out.write(org_sent+'\t'+lemma_sent+'\n')

