import csv
import json

input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/source_data/OpenReviewTestData/bids.csv"
output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/source_data/bids/bids.json"
score_map_dict = {"2": "2-not_willing", "3": "3-in_a_pinch", '4': "4-willing", '5': "5-eager"}

output_dict = {}

with open(input_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for i, row in enumerate(spamreader):
        paper_id, reviewer, bid_score = row
        if bid_score == '1':
            continue
        bid_score_show = score_map_dict[bid_score]
        bid_list = output_dict.get(reviewer,[])
        bid_list.append({"forum": paper_id, "tag": bid_score_show, "signature": reviewer})
        output_dict[reviewer] = bid_list

with open(output_file, 'w') as f_out:
    json.dump(output_dict,f_out, indent=4)
