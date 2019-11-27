#!/bin/bash

#sbatch --ntasks=40 --ntasks-per-node=10 --nodes=4 --cpus-per-task=1 --mem-per-cpu=20G ./bin/noun_chunk_all_wiki_ep.sh

#sacct -j 89262 --format=JobID,Start,End,Elapsed,NCPUS,nodelist,JobName
INPUT_DIR="/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_POS/"
OUTPUT_DIR="/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_nchunk_entities/"

mkdir -p $OUTPUT_DIR

for file_path in $INPUT_DIR/*; do
    file_name=`basename $file_path`
    output_file_name=$file_name
    echo --ntasks=1 --nodes=1 --exclusive -p cpu ~/anaconda2/bin/python src/preprocessing/tools/noun_chunking_wiki_entities.py -i $file_path -o $OUTPUT_DIR/$output_file_name
    srun --ntasks=1 --nodes=1 --exclusive -p cpu ~/anaconda2/bin/python src/preprocessing/tools/noun_chunking_wiki_entities.py -i $file_path -o $OUTPUT_DIR/$output_file_name  &
done
wait
