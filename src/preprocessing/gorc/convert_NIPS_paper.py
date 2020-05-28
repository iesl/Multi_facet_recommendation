import csv
import json
import os

input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/source_data/OpenReviewTestData/meta.csv"
output_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/source_data/submissions"

#output_list = []

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(input_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for i, row in enumerate(spamreader):
        if i==0:
            continue
        paper_id, title, abstract, authors = row
        author_list = authors.replace(' ','').split(';')
        paper_data = {}
        paper_data['id'] = paper_id
        content = {}
        content['title'] = title
        content['abstract'] = abstract.replace('\n',' ')
        content['authors'] = author_list
        content['authorids'] = author_list
        paper_data['content'] = content
        with open(output_dir+'/'+paper_id + '.jsonl','w') as f_out:
            json.dump(paper_data,f_out)
            #line = json.dumps(paper_data)
            #f_out.write(line)
        #line = json.dumps(paper_data)
        #output_list.append(line)

#with open(output_file, 'w') as f_out:
#    f_out.write('\n'.join(output_list)+'\n') 
