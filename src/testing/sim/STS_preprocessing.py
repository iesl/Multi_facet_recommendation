import os
import sys

#years = ['2012', '2013', '2014', '2015', '2016']
#output_file = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/sts_all_years_test'
years = ['2012', '2013', '2014', '2015']
folder_prefix = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/sts-en-test-gs-'
output_prefix = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/sts_test_year_'
file_start = 'STS'
#years = ['2016']
#folder_prefix = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/sts-en-test-gs-'
#output_prefix = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/sts_test_year_'
#file_start = 'STS2016'
file_end = '.txt'
#years = ['2012']
#folder_prefix = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/sts-en-train-gs-'
#output_file = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/sts_2012_train'


for year in years:
    input_dir = folder_prefix + year
    files = os.listdir(input_dir)
    type_d2_sent_pairs = {}
    type_d2_gt = {}
    for file_name_raw in files:
        print("Processing "+ year + ' ' + file_name_raw)
        sys.stdout.flush()
        file_name = file_name_raw.replace(file_end,'')
        if 'ALL' in file_name or 'LICENSE' in file_name or file_end not in file_name_raw:
            print('ignore '+ file_name)
        elif file_name.startswith(file_start+'.input.'):
            type_name = file_name[len(file_start+'.input.'):]
            with open(input_dir + '/'+ file_name_raw) as f_in:
                sent_pair_list = [line.rstrip().split('\t') for line in f_in]
            type_d2_sent_pairs[type_name] = sent_pair_list
        elif file_name.startswith(file_start+'.gs.'):
            type_name = file_name[len(file_start+'.gs.'):]
            with open(input_dir + '/'+ file_name_raw) as f_in:
                #score_list = [float(line.rstrip()) for line in f_in]
                score_list = [(line.rstrip()) for line in f_in]
            type_d2_gt[type_name] = score_list
        else:
            print('ignore '+ file_name)
    output_list = []
    for type_name in type_d2_sent_pairs:
        for line_idx, (sent_pair, gt) in enumerate(zip(type_d2_sent_pairs[type_name], type_d2_gt[type_name])):
            if len(gt) == 0:
                continue
            output_list.append('\t'.join([type_name, type_name, year, str(line_idx), gt, sent_pair[0], sent_pair[1]]))
    output_file = output_prefix + year
    with open(output_file, 'w') as f_out:
        f_out.write('\n'.join(output_list)+'\n')
