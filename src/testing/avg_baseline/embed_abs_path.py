"""
Script to run allennlp predict command for embedding papers 
This makes it easier to run the predict command without dealing with all the overrides

Example usage:

python scripts/run_predict.py \
--ids /path/to/papr/ids.txt \
--model /path/to/model/model.tar.gz \
--metadata /path/to/metadata/metadata.json \
--cuda-device 0 \
--batch-size 32 \
--output-file /path/to/output

"""
import subprocess

import argparse

import logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids', help='path to the paper ids file to embed')
    parser.add_argument('--model', help='path to the model')
    parser.add_argument('--metadata', help='path to the paper metadata')
    parser.add_argument('--output-file', help='path to the output file')
    parser.add_argument('--cuda-device', default=0, type=str)
    parser.add_argument('--batch-size', default=1, type=str)
    parser.add_argument('--vocab-dir', default='data/vocab/')
    parser.add_argument('--included-text-fields', default='abstract title')
    parser.add_argument('--weights-file', default=None)
    parser.add_argument('--py_path', default="~/anaconda3/bin/python")
    parser.add_argument('--specter_folder', default=".")

    args = parser.parse_args()

    overrides = f"{{'model':{{'predict_mode':'true','include_venue':'false'}},'dataset_reader':{{'type':'specter_data_reader','predict_mode':'true','paper_features_path':'{args.metadata}','included_text_fields': '{args.included_text_fields}'}},'vocabulary':{{'directory_path':'{args.vocab_dir}'}}}}"

    command = [
        args.py_path,
        args.specter_folder+'/specter/predict_command.py',
        'predict',
        args.model,
        args.ids,
        '--include-package',
        'specter',
        '--predictor',
        'specter_predictor',
        '--overrides',
        f'"{overrides}"',
        '--cuda-device',
        args.cuda_device,
        '--output-file',
        args.output_file,
        '--batch-size',
        args.batch_size,
        '--silent'
    ]
    if args.weights_file is not None:
        command.extend(['--weights-file', args.weights_file])

    logging.info('running command:')
    logging.info(' '.join(command))

    subprocess.run(' '.join(command), shell=True)

if __name__ == '__main__':
    main()
