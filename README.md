Assuming you want to run ICLR2020

## Preprocessing:
You will need 
`data/raw/openreview/ICLR2020/source_data/archives`
`data/raw/openreview/ICLR2020/source_data/profiles_expertise/profiles_expertise.json`
`data/raw/openreview/ICLR2020/source_data/submissions`
`data/raw/openreview/ICLR2020/source_data/assignments/assignments.json`
`data/raw/openreview/ICLR2020/source_data/bids/bids.json`

Run (modify the path if necessary)
```
src/preprocessing/gorc/prepare_data_for_reviewer_emb.py
src/preprocessing/gorc/prepare_data_for_assignment_testing.py #(user_tag_source = 'assignment')
src/preprocessing/gorc/prepare_data_for_assignment_testing.py #(user_tag_source = 'bid')
```

You will need 
`./data/raw/openreview/ICLR2020/all_reviewer_paper_data` (The output files generated above)
`./data/raw/openreview/ICLR2020/all_submission_paper_data` (The output files generated above)
`data/processed/gorc_fix_uncased_min_5/feature/dictionary_index` (The model file is on gypsum. download from `/mnt/nfs/scratch1/hschang/recommend/Multi_facet_recommendation/data/processed/gorc_fix_uncased_min_5/feature/dictionary_index` )

Run (modify the path if necessary)
```
bin/preprocessing_openreview.sh
bin/preprocessing_openreview_bid_score.sh
```

## SPECTOR:
Change the path and run the following two codes
`src/preprocessing/gorc/convert_paper_train_spector.py`
`/iesl/canvas/hschang/recommendation/specter/run.sh` (The code is on blake2. Need to download specter repository from https://github.com/allenai/specter)


## PREDICT:
You will need 
`./data/raw/openreview/ICLR2020/all_reviewer_paper_data` (from preprocessing)
`./data/processed/ICLR2020_fix_gorc_uncased` (from preprocessing)
`./data/processed/ICLR2020_bid_score_fix_gorc_uncased` (from preprocessing)
`./gen_log/ICLR2020_emb_spector_raw.jsonl` (from SPECTER)
`./gen_log/ICLR2020_emb_spector_raw_train.jsonl` (from SPECTER)
`./models/gorc_fix-20200722-104422` (The model file is on gypsum. download from `/mnt/nfs/scratch1/hschang/recommend/Multi_facet_recommendation/models/gorc_fix-20200722-104422` )

Run (modify the path if necessary)
```
bin/testing_for_ICLR2020.sh
```

