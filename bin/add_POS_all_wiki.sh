#!/bin/bash

#sbatch --ntasks=50 --ntasks-per-node=10 --nodes=5 --cpus-per-task=1 --mem-per-cpu=10G ./bin/add_POS_all_wiki.sh

#sacct -j 89262 --format=JobID,Start,End,Elapsed,NCPUS,nodelist,JobName
INPUT_DIR="/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_raw/"
OUTPUT_DIR="/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_POS/"

mkdir -p $OUTPUT_DIR

for file_path in $INPUT_DIR/*; do
    file_name=`basename $file_path`
    output_file_name=$file_name
    #output_file_name="${file_name//.json.gz/}"
    echo --ntasks=1 --nodes=1 --exclusive -p cpu ~/anaconda2/bin/python src/preprocessing/tools/add_POS_wiki.py -i $file_path -o $OUTPUT_DIR/$output_file_name
    srun --ntasks=1 --nodes=1 --exclusive -p cpu ~/anaconda2/bin/python src/preprocessing/tools/add_POS_wiki.py -i $file_path -o $OUTPUT_DIR/$output_file_name  &
    #srun --ntasks=1 --nodes=1 --mem=100G -p cpu ~/anaconda3/bin/python src/preprocessing/tokenize_wiki.py -i $file_path -o $OUTPUT_DIR/$output_file_name
    #srun --ntasks=1 --mem=100G -p cpu ~/anaconda3/bin/python src/preprocessing/tokenize_wiki.py -i $file_path -o $OUTPUT_DIR/$output_file_name
    #sleep 2
done
wait
