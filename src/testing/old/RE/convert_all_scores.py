import os

input_folder = '/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/gen_log'
output_folder = '/iesl/canvas/hschang/TAC_2016/codes/torch-relation-extraction/results'

score_list = ['cos', 'kmeans_p2r', 'kmeans_r2p', 'kmeans_avg', 'SC_r2p', 'SC_p2r', 'SC_avg']
basis_list = ['b1', 'b3']
update_list = ['upd', 'no_upd']
#score_list = ['cos']
year_list = ['2012', '2013', '2014']

for basis_name in basis_list:
    for update_name in update_list:
        for score_idx, score_name in enumerate(score_list):
            for year_name in year_list:
                score_field = str(score_idx + 10)
                input_file_name = input_folder + '/' + basis_name + '/' + update_name + '/full_sentence_candidates_'+year_name+'_scored'
                current_output_dir = output_folder + '/' + basis_name + '_' + update_name + '_' + score_name
                os.makedirs(current_output_dir, exist_ok=True)
                output_file_name = current_output_dir + '/' + year_name + '_scored'
                command = 'awk -F $\'\\t\' \'{print $1"\\t"$2"\\t"$3"\\t"$4"\\t0\\t0\\t0\\t0\\t"$' +score_field+ '}\' ' + input_file_name + ' > ' + output_file_name
                print(command)
                os.system(command)


