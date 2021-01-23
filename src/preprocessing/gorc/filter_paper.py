import pandas as pd
import os
import csv
import sys
import getopt

csv.field_size_limit(sys.maxsize)

help_msg = '-i <meta_dir> -o <meta_output>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:o:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-i"):
        meta_dir = arg
    elif opt in ("-o"):
        meta_output = arg

#meta_dir = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/20190928/metadata'
#meta_output = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_cs.tsv'
file_list = os.listdir(meta_dir)

#meta_all_df = pd.DataFrame()
meta_all_list = []

for i, filename in enumerate(file_list):
    file_path = os.path.join(meta_dir, filename)
    print('processing ', file_path)
    meta_pd = pd.read_csv(file_path, delimiter='\t', error_bad_lines=False, engine='python', quoting=csv.QUOTE_NONE ) #encoding='ISO-8859-1')
    try:
        meta_pd_filtered = meta_pd[meta_pd['mag_fos'].str.contains('Computer science')]
    except:
        idx = meta_pd['mag_fos'].str.contains('Computer science')
        print(meta_pd[idx.isna()])
        idx[idx.isna()] = False
        meta_pd_filtered = meta_pd[idx]
    print("{} size {}, after filtering {}".format( filename, len(meta_pd.index), len(meta_pd_filtered.index) ))
    meta_all_list.append(meta_pd_filtered)
    #meta_all_df = meta_all_df.append(meta_pd_filtered)
    #print("total length ", len(meta_all_df.index))
    #if i > 10:
    #    break
    #if 'Computer science' in category:
meta_all_df = pd.concat(meta_all_list)
meta_all_df.to_csv(meta_output, sep='\t')
