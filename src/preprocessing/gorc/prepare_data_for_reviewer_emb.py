import numpy as np
import os
import json
from unicodedata import normalize

import sys
import getopt

help_msg = '-i <paper_dir> -o <output_path>'

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
        paper_dir = arg
    elif opt in ("-o"):
        output_path = arg

tokenizer_mode = 'scapy'
#tokenizer_mode = 'scibert'

if tokenizer_mode == 'scapy':
    from spacy.lang.en import English
    nlp = English()
elif tokenizer_mode == 'scibert':
    import pysbd #scibert uses scispacy and scispacy uses pysbd (https://github.com/allenai/scispacy/blob/master/scispacy/custom_sentence_segmenter.py)
    import sys
    sys.path.insert(0, sys.path[0]+'/../..')
    from scibert.tokenization_bert import BertTokenizer
    model_name = 'scibert-scivocab-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    seg = pysbd.Segmenter(language="en", clean=False)

expertise_file = ""
dataset = 'new'
#dataset = 'UAI2019'
#dataset = 'ICLR2020'
#dataset = 'ICLR2020_test'
#dataset = 'ICLR2019'
#dataset = 'ICLR2018'
#dataset = 'NeurIPS2019'
#dataset = 'NeurIPS2020'
#dataset = 'NeurIPS2020_final'
#dataset = 'NeurIPS2020_final_review'
#dataset = 'NeurIPS2020_fixed_ac'
#dataset = 'NeurIPS2020_fixed_review'

reviewer_mapping_file = ""
if dataset == 'NeurIPS2020_fixed_review':
    paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_fixed_review/source_data/archives"
    expertise_file = ""
    reviewer_mapping_file = ""
    #reviewer_mapping_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_final_review/source_data/neurips20_review_profile_1.csv"
    if tokenizer_mode == 'scapy':    
        #output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_fixed_ac/all_reviewer_paper_data"
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_fixed_review/all_reviewer_paper_data"

elif dataset == 'NeurIPS2020_fixed_ac':
    paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_fixed_ac/source_data/archives"
    expertise_file = ""
    reviewer_mapping_file = ""
    #reviewer_mapping_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_final_review/source_data/neurips20_review_profile_1.csv"
    if tokenizer_mode == 'scapy':    
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_fixed_ac/all_reviewer_paper_data"

elif dataset == 'NeurIPS2020_final_review':
    paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_final_review/source_data/archives"
    expertise_file = ""
    reviewer_mapping_file = ""
    #reviewer_mapping_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_final_review/source_data/neurips20_review_profile_1.csv"
    if tokenizer_mode == 'scapy':    
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_final_review/all_reviewer_paper_data"

elif dataset == 'NeurIPS2020_final':
    paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_final/source_data/archives"
    expertise_file = ""
    reviewer_mapping_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_final/source_data/neurips20_ac_profile.csv"
    if tokenizer_mode == 'scapy':    
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_final/all_reviewer_paper_data"

elif dataset == 'NeurIPS2020':
    paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020/source_data/archives"
    expertise_file = ""
    if tokenizer_mode == 'scapy':    
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020/all_reviewer_paper_data"
    elif tokenizer_mode == 'scibert':
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020/all_reviewer_paper_data_scibert"

elif dataset == 'NeurIPS2019':
    paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/source_data/archives"
    expertise_file = ""
    if tokenizer_mode == 'scapy':    
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/all_reviewer_paper_data"
    elif tokenizer_mode == 'scibert':
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/all_reviewer_paper_data_scibert"
elif dataset == 'UAI2019':
    paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/UAI2019/source_data/archives"
    expertise_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/UAI2019/source_data/profiles_expertise/profiles_expertise.json"
    if tokenizer_mode == 'scapy':    
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/UAI2019/all_reviewer_paper_data"
    elif tokenizer_mode == 'scibert':
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/UAI2019/all_reviewer_paper_data_scibert"
elif dataset == 'ICLR2020_test':
    paper_dir = "/iesl/canvas/hschang/code/Multi_facet_recommendation/data/raw/openreview/ICLR2020/source_data/archives"
    output_path = "/iesl/canvas/hschang/code/Multi_facet_recommendation/data/raw/openreview/ICLR2020/all_reviewer_paper_data_test"

elif dataset == 'ICLR2020':
    paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020/source_data/archives"
    expertise_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020/source_data/profiles_expertise/profiles_expertise.json"
    if tokenizer_mode == 'scapy':
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020/all_reviewer_paper_data"
    elif tokenizer_mode == 'scibert':
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020/all_reviewer_paper_data_scibert"
elif dataset == 'ICLR2019':
    paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2019/source_data/archives"
    expertise_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2019/source_data/profiles_expertise/profiles_expertise.json"
    if tokenizer_mode == 'scapy':
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2019/all_reviewer_paper_data"
    elif tokenizer_mode == 'scibert':
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2019/all_reviewer_paper_data_scibert"
elif dataset == 'ICLR2018':
    paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2018/source_data/archives"
    expertise_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2018/source_data/profiles_expertise/profiles_expertise.json"
    if tokenizer_mode == 'scapy':
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2018/all_reviewer_paper_data"
    elif tokenizer_mode == 'scibert':
        output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2018/all_reviewer_paper_data_scibert"


reviewer_d2_expertise = {}
if len(expertise_file) > 0:
    with open(expertise_file) as f_in:
        all_expertise = json.load(f_in)
        for reviewer_name in all_expertise:
            keyword_list = []
            if all_expertise[reviewer_name] is None:
                reviewer_d2_expertise[reviewer_name] = []
                continue
            for fields in all_expertise[reviewer_name]:
                keyword_list += fields["keywords"]
            reviewer_d2_expertise[reviewer_name] = keyword_list

reviewer_d2_real_name = {}
if len(reviewer_mapping_file) > 0:
    with open(reviewer_mapping_file) as f_in:
        for line in f_in:
            #real_name, or_name = line.strip().split(',')
            or_name, real_name = line.strip().split(',')
            reviewer_d2_real_name[normalize('NFC',or_name)] = real_name

#print(reviewer_d2_real_name)

paper_id_d2_features_type_author_other = {}
all_files = os.listdir(paper_dir)
for file_name in all_files:
    author_name = file_name.replace('.jsonl','')
    if len(reviewer_d2_real_name) > 0:
        try:
            author_name = reviewer_d2_real_name[normalize('NFC',author_name)]
        except:
            print(author_name)
            continue
    expertise = reviewer_d2_expertise.get(author_name,[])
    reviewer_full_name = (author_name + '|' + '+'.join(expertise)).replace(' ','_')
    with open( os.path.join(paper_dir, file_name) ) as f_in:
        for line in f_in:
            paper_data = json.loads(line)
            paper_id = paper_data['id']
            if paper_id in paper_id_d2_features_type_author_other:
                paper_id_d2_features_type_author_other[paper_id][2].append(reviewer_full_name)
            else:
                if "abstract" in paper_data['content']:
                    abstract = paper_data['content']["abstract"]
                else:
                    abstract = None
                title = paper_data['content']["title"]
                if "authors" not in paper_data['content']:
                    author_list = [author_name]
                else:
                    author_list = paper_data['content']["authors"]
                if "authorids" not in paper_data['content']:
                    author_id_list = [author_name]
                else:
                    author_id_list = paper_data['content']["authorids"]
                author_full_str = ','.join(['|'.join(x) for x in zip(author_list, author_id_list)]).replace(' ','_').replace('\t','_')
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
                        w_list_abstract = []
                        #try:
                        for sent in seg.segment(abstract):
                            w_list_abstract += tokenizer.tokenize('[CLS] ' + sent + ' [SEP]')
                        #except:
                        #    print(abstract)
                        #    sys.exit(1)
                    #w_list_abstract = [w.text for w in nlp.tokenizer( abstract ) ] + ['<SEP>']
                    w_list_abstract = ' '.join(w_list_abstract).split()
                else:
                    w_list_abstract = []

                type_list = ['0']*len(w_list_title) + ['1']*len(w_list_abstract)
                paper_id_d2_features_type_author_other[paper_id] = [' '.join(w_list_title + w_list_abstract), ' '.join(type_list), [reviewer_full_name], author_full_str, paper_id]

with open(output_path, 'w') as f_out:
    for paper_id in paper_id_d2_features_type_author_other:
        paper_info = paper_id_d2_features_type_author_other[paper_id]
        paper_info[2] = ','.join(paper_info[2])
        f_out.write('\t'.join(paper_info)+'\n')
