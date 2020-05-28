
#input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/processed/NeurIPS2019_bid_score_gorc_uncased/paper_id_train"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/processed/NeurIPS2019_bid_score_gorc_uncased/paper_id_train_uniq"
#assume paper_text_data have the same order as paper_id_train
#paper_text_data = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2019/all_reviewer_paper_data_scibert" 
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/processed/NeurIPS2019_bid_score_scibert_gorc_uncased/paper_id_train_uniq"
paper_text_data = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/ICLR2020/all_reviewer_paper_data_scibert" 
output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/processed/ICLR2020_bid_score_scibert_gorc_uncased/paper_id_train_uniq"


paper_feature_set = set()
f_out = open(output_file, 'w')

#with open(input_file) as f_in:
with open(paper_text_data) as f_in:
    for line in f_in:
        feature, feature_type, reviewer, authors, paper_id = line.rstrip().split('\t')
        feature = tuple(feature.lower().strip().split()[:512])
        if feature not in paper_feature_set:
            f_out.write(paper_id+'\n')
            paper_feature_set.add(feature)
            

f_out.close()
