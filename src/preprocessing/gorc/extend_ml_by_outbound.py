import pandas as pd
import csv
import sys
#import ast
import json
import numpy as np
import getopt

csv.field_size_limit(sys.maxsize)

help_msg = '-c <meta_cs_path> -m <meta_ml_path> -o <meta_output>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "c:m:o:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-c"):
        meta_cs_path = arg
    elif opt in ("-m"):
        meta_ml_path = arg
    elif opt in ("-o"):
        meta_output = arg

#meta_cs_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_cs.tsv'
#meta_ml_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_ml.tsv'
#meta_output = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_ml_ext_org.tsv'
##meta_cs_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/20190928/metadata/0.tsv'
##meta_ml_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_ml.tsv'
##meta_output = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/test.tsv'

def parse_list_str(list_in):
    #print(list_in)
    return list_in.strip('[]').replace("'",'').split(', ')

meta_ml_df = pd.read_csv(meta_ml_path, delimiter='\t', engine='python', quoting=csv.QUOTE_NONE )
#meta_ml_df.outbound_citations = meta_ml_df.outbound_citations.apply(ast.literal_eval)
#meta_ml_df.outbound_citations = meta_ml_df.outbound_citations.apply(json.loads)
meta_ml_df.outbound_citations = meta_ml_df.outbound_citations.apply(parse_list_str)
print(meta_ml_df.outbound_citations)
#outbound_set = list(set(meta_ml_df.outbound_citations.sum()))
outbound_set = [int(x) for x in np.concatenate(meta_ml_df.outbound_citations) if len(x) > 0]
#outbound_set += meta_ml_df.pid.tolist()
outbound_set = list(set(outbound_set))
print(len(outbound_set))
outbound_df = pd.DataFrame({'pid': outbound_set})

meta_cs_df = pd.read_csv(meta_cs_path, delimiter='\t', engine='python', quoting=csv.QUOTE_NONE )
meta_ml_ext_df = meta_cs_df.merge(outbound_df, how='inner', on='pid')
print("hitting pid ratio: {}".format(float(len(meta_ml_ext_df.index))/len(outbound_set)))
meta_ml_ext_df.to_csv(meta_output, sep='\t')
