import csv
import json
import os
import sys
import getopt

help_msg = '-i <paper_dir> -o <output_file>'

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
        output_file = arg

##paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/source_data/archives"
##output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/source_data/paper_data_spector_train.json"

#paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020/source_data/archives"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020/source_data/paper_data_spector_train.json"
#paper_dir = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020/source_data/submissions"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020/source_data/paper_data_spector.json"

output_dict = {}

all_files = os.listdir(paper_dir)
for file_name in all_files:
    author_name = file_name.replace('.jsonl','')
    with open( os.path.join(paper_dir, file_name) ) as f_in:
        for line in f_in:
            paper_data = json.loads(line)
            paper_data_out = {}
            paper_id = paper_data['id']
            if "abstract" in paper_data['content']:
                abstract = paper_data['content']["abstract"]
                paper_data_out['abstract'] = abstract
            title = paper_data['content']["title"]
            paper_data_out['title'] = title
            if "authors" not in paper_data['content']:
                author_list = [author_name]
            else:
                author_list = paper_data['content']["authors"]
            paper_data_out['authors'] = author_list
            paper_data_out['paper_id'] = paper_id
            output_dict[paper_id] = paper_data_out

with open(output_file,'w') as f_out:
    json.dump(output_dict,f_out,indent = 1)
            #line = json.dumps(paper_data)
            #f_out.write(line)
        #line = json.dumps(paper_data)
        #output_list.append(line)

#with open(output_file, 'w') as f_out:
#    f_out.write('\n'.join(output_list)+'\n') 
