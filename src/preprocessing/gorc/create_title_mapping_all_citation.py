import pandas as pd
import os
import csv
import sys

#import sys
#sys.path.insert(0, sys.path[0]+'/../..')
#import utils

num_special_tok = 3
def load_paper_ids(f_in):
    id_list = []
    for i, line in enumerate(f_in):
        if i < num_special_tok:
            continue
        fields = line.rstrip().split('\t')
        id_list.append(int(fields[0]))
    id_pd = pd.DataFrame(id_list,columns =['pid'])
    return id_pd

csv.field_size_limit(sys.maxsize)

meta_dir = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/20190928/metadata'
id_dict = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/processed/gorc_org_uncased_min_5/tag/dictionary_index'
id_title_output = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/gorc/id_title_full_org'
file_list = os.listdir(meta_dir)

with open(id_dict) as f_in:
    id_pd = load_paper_ids(f_in)

#meta_all_df = pd.DataFrame()
meta_all_list = []

for i, filename in enumerate(file_list):
    file_path = os.path.join(meta_dir, filename)
    print('processing ', file_path)
    meta_pd = pd.read_csv(file_path, delimiter='\t', error_bad_lines=False, engine='python', quoting=csv.QUOTE_NONE ) #encoding='ISO-8859-1')
    meta_pd_filtered = meta_pd.merge(id_pd, how='inner', on='pid')
    #try:
    #    meta_pd_filtered = meta_pd[meta_pd['mag_fos'].str.contains('Computer science')]
    #except:
    #    idx = meta_pd['mag_fos'].str.contains('Computer science')
    #    print(meta_pd[idx.isna()])
    #    idx[idx.isna()] = False
    #    meta_pd_filtered = meta_pd[idx]
    print("{} size {}, after filtering {}".format( filename, len(meta_pd.index), len(meta_pd_filtered.index) ))
    sys.stdout.flush()
    meta_all_list.append(meta_pd_filtered[ ['pid', 'title'] ])
    #meta_all_df = meta_all_df.append(meta_pd_filtered)
    #print("total length ", len(meta_all_df.index))
    #if i > 3:
    #    break
    #if 'Computer science' in category:
meta_all_df = pd.concat(meta_all_list)
meta_all_df.to_csv(id_title_output, sep='\t')
