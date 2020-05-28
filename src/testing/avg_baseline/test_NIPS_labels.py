import os
import numpy as np
from sklearn.metrics import average_precision_score
import sys

label_file_path = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/source_data/NeurIPS_2019_labels.tsv'
#score_file_dir = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/to_NeurIPS2019_2/to_NeurIPS2019/'
score_file_dir = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/to_NeurIPS2019_2/new/'

#using pandas join seems to be easier

author_d2_paper_d2_label = {}

with open(label_file_path) as f_in:
    for i, line in enumerate(f_in):
        if i == 0:
            continue

        label, last_name, user_id, paper_id, title, abstract = line.rstrip().split('\t')
        paper_d2_label = author_d2_paper_d2_label.get(user_id,{})
        paper_d2_label[paper_id] = int(label)
        author_d2_paper_d2_label[user_id] = paper_d2_label

def compute_MAP(author_d2_paper_d2_label_score):
    AP_list = []
    total_eval_count = 0
    for user_id in author_d2_paper_d2_label_score:
        label_score = author_d2_paper_d2_label_score[user_id].values()
        label, score = zip(*label_score)
        label_sum = sum(label)
        if label_sum == 0 or label_sum == len(label):
            #either all labels are 0 or 1
            continue
        AP = average_precision_score(label, score)
        AP_list.append(AP)
        total_eval_count += len(label)
    #print(AP_list)
    return np.mean(AP_list), total_eval_count
    
file_list = os.listdir(score_file_dir)
for file_name in file_list:
    author_d2_paper_d2_label_score = {}
    with open(score_file_dir+'/'+file_name) as f_in:
        for i, line in enumerate(f_in):
            paper_id, user_id, score = line.rstrip().split(',')
            if user_id not in author_d2_paper_d2_label:
                continue
            if paper_id not in author_d2_paper_d2_label[user_id]:
                continue

            paper_d2_label_score = author_d2_paper_d2_label_score.get(user_id,{})
            paper_d2_label_score[paper_id] = [author_d2_paper_d2_label[user_id][paper_id],float(score)]
            author_d2_paper_d2_label_score[user_id] = paper_d2_label_score
    MAP, total_eval_count = compute_MAP(author_d2_paper_d2_label_score)
    print(file_name, MAP, total_eval_count)
    sys.stdout.flush()
