import numpy as np
import os
import json
import sys
import getopt

help_msg = '-i <paper_dir> -e <expertise_file> -s <user_tag_source> -a <assignment_file> -b <bid_file> -o <output_path>'

expertise_file = ""
assignment_file = ""
bid_file = ""

#user_tag_source = 'bid'
user_tag_source = 'assignment'

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:e:s:a:b:o:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-i"):
        paper_dir = arg
    elif opt in ("-e"):
        expertise_file = arg
    elif opt in ("-s"):
        user_tag_source = arg
    elif opt in ("-a"):
        assignment_file = arg
    elif opt in ("-b"):
        bid_file = arg
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


dataset = 'new'
#dataset = 'UAI2019'
#dataset = 'ICLR2020'
#dataset = 'ICLR2020_test'
#dataset = 'ICLR2019'
#dataset = 'ICLR2018'
#dataset = 'NeurIPS2019'

if dataset == 'NeurIPS2019':
    paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/source_data/submissions"
    expertise_file = ""
    assert user_tag_source == 'bid'
    if user_tag_source == 'bid':
        bid_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/source_data/bids/bids.json"
        if tokenizer_mode == 'scapy':
            output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019_bid_score/all_submission_bid_data"
        elif tokenizer_mode == 'scibert':
            output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019_bid_score/all_submission_bid_data_scibert"
elif dataset == 'UAI2019':
    paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/UAI2019/source_data/submissions"
    expertise_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/UAI2019/source_data/profiles_expertise/profiles_expertise.json"
    #assignment_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/UAI2019/source_data/assignments/assignments.json"
    bid_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/UAI2019/source_data/bids/bids.json"
    #output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/UAI2019/all_submission_paper_data"
    #output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/UAI2019/all_submission_paper_data_scibert"
    #output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/UAI2019_bid_score/all_submission_bid_data"
    output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/UAI2019_bid_score/all_submission_bid_data_scibert"
    #output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/UAI2019_bid_low/all_submission_bid_data"
    #output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/UAI2019_bid_high/all_submission_bid_data"
elif dataset == 'ICLR2020_test':
    paper_dir = "/iesl/canvas/hschang/code/Multi_facet_recommendation/data/raw/openreview/ICLR2020/source_data/submissions"
    output_path = "/iesl/canvas/hschang/code/Multi_facet_recommendation/data/raw/openreview/ICLR2020/all_submission_paper_data_test"
    
elif dataset == 'ICLR2020':
    paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020/source_data/submissions"
    expertise_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020/source_data/profiles_expertise/profiles_expertise.json"
    if user_tag_source == 'assignment':
        assignment_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020/source_data/assignments/assignments.json"
        if tokenizer_mode == 'scapy':
            output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020/all_submission_paper_data"
        elif tokenizer_mode == 'scibert':
            output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020/all_submission_paper_data_scibert"
    elif user_tag_source == 'bid':
        bid_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020/source_data/bids/bids.json"
        if tokenizer_mode == 'scapy':
            output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020_bid_score/all_submission_bid_data"
        elif tokenizer_mode == 'scibert':
            output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020_bid_score/all_submission_bid_data_scibert"
    #output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020_bid_high/all_submission_bid_data"
    #output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020_bid_low/all_submission_bid_data"
elif dataset == 'ICLR2019':
    paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2019/source_data/submissions"
    expertise_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2019/source_data/profiles_expertise/profiles_expertise.json"
    if user_tag_source == 'assignment':
        assignment_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2019/source_data/assignments/assignments.json"
        if tokenizer_mode == 'scapy':
            output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2019/all_submission_paper_data"
        elif tokenizer_mode == 'scibert':
            output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2019/all_submission_paper_data_scibert"
    elif user_tag_source == 'bid':
        bid_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2019/source_data/bids/bids.json"
        if tokenizer_mode == 'scapy':
            output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2019_bid_score/all_submission_bid_data"
        elif tokenizer_mode == 'scibert':
            output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2019_bid_score/all_submission_bid_data_scibert"
elif dataset == 'ICLR2018':
    paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2018/source_data/submissions"
    expertise_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2018/source_data/profiles_expertise/profiles_expertise.json"
    if user_tag_source == 'assignment':
        assignment_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2018/source_data/assignments/assignments.json"
        if tokenizer_mode == 'scapy':
            output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2018/all_submission_paper_data"
        elif tokenizer_mode == 'scibert':
            output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2018/all_submission_paper_data_scibert"
    elif user_tag_source == 'bid':
        bid_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2018/source_data/bids/bids.json"
        if tokenizer_mode == 'scapy':
            output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2018_bid_score/all_submission_bid_data"
        elif tokenizer_mode == 'scibert':
            output_path = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2018_bid_score/all_submission_bid_data_scibert"

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

paper_id_d2_reviewers = {}
if user_tag_source == 'bid':
    if dataset == 'NeurIPS2019':
        score_map_dict = {"2-not_willing": '2', "3-in_a_pinch": '3', "4-willing": '4', "5-eager": '5'}
    elif dataset == 'ICLR2018':
        score_map_dict = {"No bid": "0", "I cannot review": '1', "I can probably review but am not an expert": '2', "I can review": '3', "I want to review": '4'}
    else:
        score_map_dict = {"No Bid": "0", "Very Low": '1', "Low": '2', "Neutral": '3', "High": '4', "Very High": '5'}
    with open(bid_file) as f_in:
        all_bids = json.load(f_in)
        for reviewer_name in all_bids:
            expertise = reviewer_d2_expertise.get(reviewer_name,[])
            reviewer_full_name = (reviewer_name + '|' + '+'.join(expertise)).replace(' ','_')
            for fields in all_bids[reviewer_name]:
                preference = fields["tag"]
                #if 'High' in preference:
                #    continue
                #if 'Low' in preference:
                #    continue
                paper_id = fields["forum"]
                reviewers = paper_id_d2_reviewers.get(paper_id,[])
                reviewers.append( [ reviewer_full_name, score_map_dict[preference]] )
                paper_id_d2_reviewers[paper_id] = reviewers

elif user_tag_source == 'assignment' and len(assignment_file) > 0:
    with open(assignment_file) as f_in:
        all_assignments = json.load(f_in)
        for reviewer_name in all_assignments:
            expertise = reviewer_d2_expertise.get(reviewer_name,[])
            reviewer_full_name = (reviewer_name + '|' + '+'.join(expertise)).replace(' ','_')
            for fields in all_assignments[reviewer_name]:
                paper_id = fields["head"]
                reviewers = paper_id_d2_reviewers.get(paper_id,[])
                reviewers.append(reviewer_full_name)
                paper_id_d2_reviewers[paper_id] = reviewers

paper_id_d2_features_type_author_other = {}
all_files = os.listdir(paper_dir)
for file_name in all_files:
    id_name = file_name.replace('.jsonl','')
    #reviewers = paper_id_d2_reviewers[id_name]
    reviewers = paper_id_d2_reviewers.get(id_name,[])

    with open( os.path.join(paper_dir, file_name) ) as f_in:
        paper_data = json.load(f_in)
        paper_id = paper_data['id']
        assert paper_id == id_name
        if "abstract" in paper_data['content']:
            abstract = paper_data['content']["abstract"]
            if dataset == 'NeurIPS2019':
                abstract = abstract.replace('\\textbf{','').replace('\\textrm{','').replace('\\textit{','').replace('\\text{','').replace('\\mathcal{','').replace('\\mathbb{','').replace('\\mathrm{','').replace('{','').replace('}','').replace('$','')
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
        w_list_title = ' '.join(w_list_title).split()
        if abstract is not None:
            if tokenizer_mode == 'scapy':
                w_list_abstract = [w.text for w in nlp.tokenizer( abstract ) ] + ['<SEP>']
            elif tokenizer_mode == 'scibert':
                w_list_abstract = []
                for sent in seg.segment(abstract):
                    w_list_abstract += tokenizer.tokenize('[CLS] ' + sent + ' [SEP]')
                #print(seg.segment(text))
            w_list_abstract = ' '.join(w_list_abstract).split()
        else:
            w_list_abstract = []

        type_list = ['0']*len(w_list_title) + ['1']*len(w_list_abstract)
        if user_tag_source == 'bid':
            reviewers_name, bid_score = zip(*reviewers)
            paper_id_d2_features_type_author_other[paper_id] = [' '.join(w_list_title + w_list_abstract), ' '.join(type_list), ','.join(reviewers_name), author_full_str, paper_id, ','.join(bid_score)]
        else:
            paper_id_d2_features_type_author_other[paper_id] = [' '.join(w_list_title + w_list_abstract), ' '.join(type_list), ','.join(reviewers), author_full_str, paper_id]

with open(output_path, 'w') as f_out:
    for paper_id in paper_id_d2_features_type_author_other:
        paper_info = paper_id_d2_features_type_author_other[paper_id]
        f_out.write('\t'.join(paper_info)+'\n')
