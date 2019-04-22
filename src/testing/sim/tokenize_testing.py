import re
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer

nlp = English()
#tokenizer = Tokenizer(nlp.vocab)

#file_path = "/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/SCWS/scws_only_sent"
#output_path = "/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/SCWS/scws_org"

#file_path = "/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/stsbenchmark/sts-dev_only_sent"
#output_path = "/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/stsbenchmark/sts-dev_org"

file_path = "/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/WiC_dataset/dev/dev.data_only_sent"
output_path = "/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/WiC_dataset/dev/dev.data_org"

filter_set = set(['<b>','</b>','</font','style>'])


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

output_list = []

with open(file_path) as f_in:
    for line in f_in:
        w_list = cleanhtml(line.rstrip()).split()
        #print(w_list)
        sent_filtered = ' '.join([x for x in w_list if x not in filter_set])
        out_sent = ' '.join([w.text for w in nlp.tokenizer(sent_filtered)])
        output_list.append([line.rstrip(), out_sent])

with open(output_path,'w') as f_out:
    for org_sent, lemma_sent in output_list:
        f_out.write(org_sent+'\t'+lemma_sent+'\n')

