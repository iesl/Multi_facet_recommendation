import csv
import json
import os

input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/source_data/OpenReviewTestData/meta.csv"
output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/source_data/paper_data_spector.json"

output_dict = {}

with open(input_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for i, row in enumerate(spamreader):
        if i==0:
            continue
        paper_id, title, abstract, authors = row
        author_list = authors.replace(' ','').split(';')
        paper_data = {}
        paper_data['paper_id'] = paper_id
        paper_data['title'] = title
        paper_data['abstract'] = abstract.replace('\n',' ')
        paper_data['authors'] = author_list
        output_dict[paper_id] = paper_data

with open(output_file,'w') as f_out:
    json.dump(output_dict,f_out,indent = 1)
            #line = json.dumps(paper_data)
            #f_out.write(line)
        #line = json.dumps(paper_data)
        #output_list.append(line)

#with open(output_file, 'w') as f_out:
#    f_out.write('\n'.join(output_list)+'\n') 
