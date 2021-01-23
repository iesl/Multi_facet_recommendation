#!/bin/bash
set -e

PY_PATH="~/anaconda3/bin/python"

##INPUT
GORC_META_ALL="/iesl/data/word_embedding/gorc/s2-gorc/gorc/20190928/metadata"
GORC_PAPER_ALL="/iesl/data/word_embedding/gorc/s2-gorc/gorc/20190928/papers"


##OUTPUT
OUTPUT_DIR="./data/raw/gorc_raw"
OUTPUT_FILE="${OUTPUT_DIR}/all_paper_data"


##Path for intermediate files
META_CS="${OUTPUT_DIR}/metadata_cs.tsv"
ARXIV_ML_ID="${OUTPUT_DIR}/arXiv_id_ML"
META_ML="${OUTPUT_DIR}/metadata_ml.tsv"
META_ML_EXT="${OUTPUT_DIR}/metadata_ml_ext.tsv"
PAPER_JSON="${OUTPUT_DIR}/paper_data_ml_ext.json"

mkdir -p $OUTPUT_DIR

eval $PY_PATH src/preprocessing/gorc/collect_arXiv_subject_efficient.py -o $ARXIV_ML_ID
eval $PY_PATH src/preprocessing/gorc/filter_paper.py -i $GORC_META_ALL -o $META_CS
eval $PY_PATH src/preprocessing/gorc/get_ML_meta.py -i $META_CS -a $ARXIV_ML_ID -o $META_ML
eval $PY_PATH src/preprocessing/gorc/extend_ml_by_outbound.py -c $META_CS -m $META_ML -o $META_ML_EXT
eval $PY_PATH src/preprocessing/gorc/collect_data_from_meta.py -i $GORC_PAPER_ALL -m $META_ML_EXT -o $PAPER_JSON
eval $PY_PATH src/preprocessing/gorc/collect_all_data.py -i $PAPER_JSON -m $META_ML_EXT -o $OUTPUT_FILE

#meta_dir = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/20190928/metadata'
#meta_output = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_cs.tsv'
#
#filter_paper.py
#
#output_path = './arXiv_id_ML'
#
#collect_arXiv_subject_efficient.py
#
#meta_input = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_cs.tsv'
#arxiv_map_path = '/iesl/data/word_embedding/gorc/s2-gorc/arXiv_id_ML'
#meta_output = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_ml.tsv'
#
#get_ML_meta.py
#
#meta_cs_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_cs.tsv'
#meta_ml_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_ml.tsv'
#meta_output = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_ml_ext.tsv'
#
#extend_ml_by_outbound.py
#
#paper_dir = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/20190928/papers'
#meta_ml_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_ml_ext.tsv'
#paper_data_out = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/paper_data_ml_ext.json'
#
#collect_data_from_meta.py
#
#meta_ml_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/metadata_ml_ext.tsv'
#paper_data_path = '/iesl/data/word_embedding/gorc/s2-gorc/gorc/paper_data_ml_ext.json'
#output_path = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/gorc/all_paper_data'
#
#collect_all_data.py
