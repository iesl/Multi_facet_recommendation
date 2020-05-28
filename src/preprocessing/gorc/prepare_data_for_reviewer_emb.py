import numpy as np
import os
import json

#tokenizer_mode = 'scapy'
tokenizer_mode = 'scibert'

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

#dataset = 'UAI2019'
#dataset = 'ICLR2020'
#dataset = 'ICLR2019'
#dataset = 'ICLR2018'
dataset = 'NeurIPS2019'

if dataset == 'NeurIPS2019':
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

paper_id_d2_features_type_author_other = {}
all_files = os.listdir(paper_dir)
for file_name in all_files:
    author_name = file_name.replace('.jsonl','')
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
                author_list = paper_data['content']["authors"]
                author_id_list = paper_data['content']["authorids"]
                author_full_str = ','.join(['|'.join(x) for x in zip(author_list, author_id_list)]).replace(' ','_')
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
