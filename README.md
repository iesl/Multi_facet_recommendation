# Train the model

Could be skipped by downloading the pretrained model here

## Prepare the training data

- Download S2ORC from AI2 (https://github.com/allenai/s2orc)
- Download embeddings-basic-cbow.txt from [here](https://drive.google.com/file/d/1q1vVMwT7EiwpGBQlNI3SYX462geXjA0Z/view?usp=sharing)
- Update the input (and output) paths of `./bin/collect_filter_gorc.sh`, and run the code to collect the papers related to machine learning from S2ORC
- Update the input (and output) paths of `./bin/preprocessing_gorc_clean.sh`, and run the code to convert the tokenize the inputs. 

## Train

- Update the input (and output) paths of `./bin/training.sh`, and run the code on a machine with a GPU with at least 11GB memory to train a model


# Estimate the Paper Similarity (and Performing Quantitative Evaluation Using Assignment or Bids)

## Download SPECTER
```
git clone https://github.com/allenai/specter.git
cd specter
wget https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/specter/archive.tar.gz
tar -xzvf archive.tar.gz
```
copy `./src/testing/avg_baseline/embed_abs_path.py` in Multi_facet_recommendation to `./scripts/embed_abs_path.py` in SPECTER repo

## Setup python
After install anaconda3, go to the SPECTER repo
```
conda create --name specter_ours python=3.7 setuptools 
conda activate specter_ours
conda install pytorch cudatoolkit=10.1 -c pytorch 
pip install -r requirements.txt
python setup.py install
conda install filelock
```

## Prepare data
Due to the sensitivity of bidding and paper assignment information, we cannot release the actual review data. In order to let the users understand the input format this repo expects, we fabricate some small examples in the `data_sample_example` folder.

For a new conference, you will need to prepare and put the following data to proper places
- All papers reviewers wrote (see an example in `data_sample_example/raw/openreview/ICLR2020/source_data/archives`)
- All submission papers (see an example in `data_sample_example/raw/openreview/ICLR2020/source_data/submissions`)
- Our model `./models/gorc_fix-20200722-104422` 
- Our dictionary file `./data/processed/gorc_fix_uncased_min_5/feature/dictionary_index`
- (the above two items could be downloaded from [here](https://drive.google.com/file/d/1q1vVMwT7EiwpGBQlNI3SYX462geXjA0Z/view?usp=sharing))

For a old conference, if you want to evaluate the performance, prepare the following two additional files
- All bidding record (see an example in `data_sample_example/raw/openreview/ICLR2020/source_data/bids/bids.json)
- All paper-reviewer assignment record (see an example in `data_sample_example/raw/openreview/ICLR2020/source_data/assignments/assignments.json)

Optionally, you can also input the expertise keywords of authors to make visualization/debugging easier
- Expertise keywords (see an example in `data_sample_example/raw/openreview/ICLR2020/source_data/profiles_expertise/profiles_expertise.json)


## Check the configuration
In `./bin/testing.sh`
- Modify `SPECTER_FOLDER` to point to your SPECTER repo
- Modify `PY_PATH` to use the python you just prepared
- If you want to put above data into a different folder, change the paths in the INPUT section (but we assume that the folder `TEXT_DATA_DIR` exist)
- If you have bidding files and assignment files, you can set `OLD_CONFERENCE="Y"` to get the quantitative evaluation
- If you like, change the output path `OUTPUT_CSV`

## Run the code
cd to this repo (Multi_facet_recommendation)

Run `./bin/testing.sh` on a machine with a GPU with at least 11GB memory
