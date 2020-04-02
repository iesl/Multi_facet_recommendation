import json
import pandas as pd
import os
import sys
import csv
import ast

csv.field_size_limit(sys.maxsize)

meta_ml_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_ml_ext.tsv'
ID_title_path = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/gorc/id_title'
author_title_path = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/gorc/author_titles'

def parse_list_str(list_in):
    return list_in.replace('[','').replace(']','').replace("'",'').replace('"','').split(', ')

meta_ml_df = pd.read_csv(meta_ml_path, delimiter='\t', engine='python', quoting=csv.QUOTE_NONE )

meta_id_title_df = meta_ml_df[['pid','title']].sort_values(by=['pid'])
meta_id_title_df.to_csv(ID_title_path,sep='\t',index=False)

#meta_ml_df.authors = meta_ml_df.authors.apply(ast.literal_eval)
meta_ml_df.authors = meta_ml_df.authors.apply(parse_list_str)

author_d2_titles = {}
for index, row in meta_ml_df.iterrows():
    for author in row['authors']:
        author = author.strip()
        if len(author) == 0:
            continue
        titles = author_d2_titles.get(author, [])
        titles.append(row['title'])
        author_d2_titles[author] = titles

with open(author_title_path, 'w') as f_out:
    author_titles_sorted = sorted(author_d2_titles.items(), key = lambda x:x[0])
    f_out.write('author\tpapers\n')
    for author, titles in author_titles_sorted:
        f_out.write(author+'\t'+' | '.join(titles)+'\n')
