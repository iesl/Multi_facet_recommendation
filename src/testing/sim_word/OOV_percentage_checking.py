import getopt
import sys
import numpy as np

help_msg = '-d <dict_file_path> -t <dataset_file_path>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "d:t:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-d"):
        dict_file_name = arg
    elif opt in ("-t"):
        dataset_file_name = arg

def load_word_dict(f_in):
    d = {}
    max_ind = 0 #if the dictionary is densified, max_ind is the same as len(d)
    for i, line in enumerate(f_in):
        fields = line.rstrip().split('\t')
        d[fields[0]] = [int(fields[2]),int(fields[1]), len(fields)]
        max_ind = int(fields[2])

    return d, max_ind

with open(dict_file_name) as f_in:
    w_d2_info, max_ind = load_word_dict(f_in)

print(dataset_file_name)

def accumulate_oov(total_count, w, w_d2_info, unfixable_oov_count, fixable_oov_freq_list):
    total_count += 1
    if w not in w_d2_info:
        unfixable_oov_count += 1
    elif w_d2_info[w][2] == 4:
        fixable_oov_freq_list.append(w_d2_info[w][1])
    return total_count, unfixable_oov_count

def print_OOV_info(fixable_oov_freq_list, unfixable_oov_count, total_count):
    fixable_oov_count = len(fixable_oov_freq_list)
    print("Total OOV percentage ", (unfixable_oov_count+fixable_oov_count)/float(total_count)  )
    print("Fixable OOV percentage ", (fixable_oov_count)/float(total_count)  )
    print("Average frequency of fixable OOV", np.mean(fixable_oov_freq_list)  )

with open(dataset_file_name) as f_in:
    total_count = 0
    unfixable_oov_count = 0
    #fixable_oov_count = 0
    fixable_oov_freq_list = []
    head_count = 0
    head_unfixable_oov_count = 0
    head_fixable_oov_freq_list = []
    for i, line in enumerate(f_in):
        fields = line.rstrip().split('\t')
        w = fields[0]
        total_count, unfixable_oov_count = accumulate_oov(total_count, w, w_d2_info, unfixable_oov_count, fixable_oov_freq_list)
        if len(fields) > 1:
            head_count, head_unfixable_oov_count = accumulate_oov(head_count, w, w_d2_info, head_unfixable_oov_count, head_fixable_oov_freq_list)
    print_OOV_info(fixable_oov_freq_list, unfixable_oov_count, total_count)
    print_OOV_info(head_fixable_oov_freq_list, head_unfixable_oov_count, head_count)
