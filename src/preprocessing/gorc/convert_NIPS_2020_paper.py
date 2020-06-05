import csv
import json
import os
import random
from spacy.lang.en import English
nlp = English()

#input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_final/source_data/neurips20_abstracts.csv"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_final/all_submission_paper_data"
input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_final_review/source_data/Papers.csv"
output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_final_review/all_submission_paper_data"

#output_list = []

        
def clean_text(abstract):
    return abstract.replace('\t',' ').replace('\n',' ')

f_out = open(output_file,'w')

with open(input_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for i, row in enumerate(spamreader):
        if i==0:
            continue
        paper_id, title, abstract = row
        title = clean_text(title)
        abstract = clean_text(abstract)
        author_list = ['r'+str(random.randint(1,10))+'|', 'r'+str(random.randint(1,10))+'|']
        w_list_title = [w.text for w in nlp.tokenizer( title ) ] + ['<SEP>']
        w_list_title = ' '.join(w_list_title).split()
        w_list_abstract = [w.text for w in nlp.tokenizer( abstract ) ] + ['<SEP>']
        w_list_abstract = ' '.join(w_list_abstract).split()
        type_list = ['0']*len(w_list_title) + ['1']*len(w_list_abstract)
        output_list = [' '.join(w_list_title + w_list_abstract), ' '.join(type_list), ','.join(author_list), ','.join(author_list), paper_id]
        f_out.write('\t'.join(output_list)+'\n')
