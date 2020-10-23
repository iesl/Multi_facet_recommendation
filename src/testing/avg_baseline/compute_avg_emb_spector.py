import json
import numpy as np
import sys
import getopt

help_msg = '-i <input> -o <output>'

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
        input_file = arg
    elif opt in ("-o"):
        output_file = arg

#input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/NeurIPS2019_emb_spector_raw.jsonl"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/NeurIPS2019_emb_spector_norm.txt"
#input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/NeurIPS2019_emb_spector_raw_train.jsonl"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/NeurIPS2019_emb_spector_train_norm.txt"
#input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/NeurIPS2019_emb_spector_raw_train_duplicate.jsonl"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/NeurIPS2019_emb_spector_train_duplicate_norm.txt"
#input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/ICLR2020_emb_spector_raw.jsonl"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/ICLR2020_emb_spector_norm.txt"
#input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/ICLR2020_emb_spector_raw_train.jsonl"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/gen_log/ICLR2020_emb_spector_train_norm.txt"

f_out = open(output_file, 'w')

paper_emb_size_default = 768

with open(input_file) as f_in:
    for line in f_in:
        paper_data = json.loads(line.rstrip())
        paper_emb_size = len(paper_data['embedding'])
        assert paper_emb_size == 0 or paper_emb_size == paper_emb_size_default
        if paper_emb_size == 0:
            paper_emb_norm = np.zeros(paper_emb_size_default)
        else:
            paper_emb = np.array(paper_data['embedding'])
            paper_emb_norm = paper_emb/(np.linalg.norm(paper_emb)+0.000000000001)
        f_out.write(' '.join(map(str,paper_emb_norm.tolist()))+'\n')

f_out.close()
