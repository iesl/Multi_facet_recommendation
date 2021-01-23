import gzip
import json
import pandas as pd
import os
import sys
import csv
import getopt

csv.field_size_limit(sys.maxsize)

help_msg = '-i <paper_dir> -m <meta_ml_path> -o <paper_data_out>'

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
        paper_dir = arg
    elif opt in ("-m"):
        meta_ml_path = arg
    elif opt in ("-o"):
        paper_data_out = arg

#paper_dir = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/20190928/papers'
#meta_ml_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_ml_ext_org.tsv'
#paper_data_out = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/paper_data_ml_ext_org.json'
##meta_ml_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_ml_ext.tsv'
##meta_ml_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_cs.tsv'
##meta_ml_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_ml.tsv'
##paper_data_out = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/paper_data_cs.json'
##paper_data_out = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/paper_data_ml_ext.json'

meta_ml_df = pd.read_csv(meta_ml_path, delimiter='\t', engine='python', quoting=csv.QUOTE_NONE )

batch_num_d2_pids = {k: set(v) for k,v in meta_ml_df.groupby("batch_num")["pid"]}

id_title_abstract_author_list = []

for batch_num in batch_num_d2_pids:
    print(batch_num)
    file_path = os.path.join(paper_dir, str(batch_num)+'.jsonl.gz')
    pid_set = batch_num_d2_pids[batch_num]
    #print(pid_set)
    with gzip.open(file_path) as f_in:
        for line in f_in:
            paper_data = json.loads(line)
            paper_id = int(paper_data['paper_id'])
            if paper_id not in pid_set:
                continue
            title = paper_data["metadata"]['title']
            abstract = paper_data["metadata"]['abstract']
            authors = paper_data["metadata"]['authors']
            id_title_abstract_author_list.append([paper_id, title, abstract, authors])
    #break

with open(paper_data_out, 'w') as outfile:
    json.dump(id_title_abstract_author_list, outfile)
