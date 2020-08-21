#Run all codes without testing

##Download specter
```
git clone git@github.com:allenai/specter.git
cd specter
wget https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/specter/archive.tar.gz
tar -xzvf archive.tar.gz 
```

##Setup python
After install anaconda3, go to the specter repo
```
conda create --name specter_ours python=3.7 setuptools 
conda activate specter_ours
conda install pytorch cudatoolkit=10.1 -c pytorch 
pip install -r requirements.txt
conda install filelock
```

##Prepare data
You will need to prepare and put the data to proper places
All papers reviewers wrote `data/raw/openreview/ICLR2020/source_data/archives`
All submission papers `data/raw/openreview/ICLR2020/source_data/submissions`
Our model `./models/gorc_fix-20200722-104422`
Our dictionary file `./data/processed/gorc_fix_uncased_min_5/feature/dictionary_index`

##Check the configuration
In `./bin/testing_for_new_conference.sh`
Modify `SPECTER_FOLDER` to point to your specter repo
If you want to put above data into a different folder, change the paths in the INPUT section (but we assume that the folder `TEXT_DATA_DIR` exist)
If you like, change the output path `OUTPUT_CSV`
If you want to use CPU instead of GPU, change `CUDA_DEVICE_IDX` to be -1

##Run the code
cd to this repo
`./bin/testing_for_new_conference.sh`



#Run all codes with testing (old)
Assuming you want to run ICLR2020

## Preprocessing:
You will need 
- `data/raw/openreview/ICLR2020/source_data/archives`
- `data/raw/openreview/ICLR2020/source_data/profiles_expertise/profiles_expertise.json`
- `data/raw/openreview/ICLR2020/source_data/submissions`
- `data/raw/openreview/ICLR2020/source_data/assignments/assignments.json`
- `data/raw/openreview/ICLR2020/source_data/bids/bids.json`

Run (modify user_tag_source and the path if necessary)
```
src/preprocessing/gorc/prepare_data_for_reviewer_emb.py
src/preprocessing/gorc/prepare_data_for_assignment_testing.py #(user_tag_source = 'assignment')
src/preprocessing/gorc/prepare_data_for_assignment_testing.py #(user_tag_source = 'bid')
```

You will need 
- `./data/raw/openreview/ICLR2020/all_reviewer_paper_data` (The output files generated above)
- `./data/raw/openreview/ICLR2020/all_submission_paper_data` (The output files generated above)
- `data/processed/gorc_fix_uncased_min_5/feature/dictionary_index` (The model file is on gypsum. download from  `/mnt/nfs/scratch1/hschang/recommend/Multi_facet_recommendation/data/processed/gorc_fix_uncased_min_5/feature/dictionary_index` )

Run (modify the path if necessary)
```
bin/preprocessing_openreview.sh
bin/preprocessing_openreview_bid_score.sh
```

## SPECTER:
Change the path and run the following two codes
```
src/preprocessing/gorc/convert_paper_train_spector.py
/iesl/canvas/hschang/recommendation/specter/run.sh #The code is on blake2. Need to download specter repository from https://github.com/allenai/specter
```


## PREDICT:
You will need 
- `./data/raw/openreview/ICLR2020/all_reviewer_paper_data` (from preprocessing)
- `./data/processed/ICLR2020_fix_gorc_uncased` (from preprocessing)
- `./data/processed/ICLR2020_bid_score_fix_gorc_uncased` (from preprocessing)
- `./gen_log/ICLR2020_emb_spector_raw.jsonl` (from SPECTER)
- `./gen_log/ICLR2020_emb_spector_raw_train.jsonl` (from SPECTER)
- `./models/gorc_fix-20200722-104422` (The model file is on gypsum. download from `/mnt/nfs/scratch1/hschang/recommend/Multi_facet_recommendation/models/gorc_fix-20200722-104422` )

Run (modify the path if necessary)
```
bin/testing_for_ICLR2020.sh
```

