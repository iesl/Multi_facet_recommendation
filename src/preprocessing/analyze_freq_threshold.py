import numpy as np

input_dict_file = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/processed/wiki2016/dictionary_index"
#input_dict_file = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/processed/wiki2016_nchunk_lower_min100/dictionary_index"

with open(input_dict_file) as f_in:
    freq_list = []
    for line in f_in:
        fields = line.rstrip().split('\t')
        freq_list.append(int(fields[1]))
    sorted_freq_list = sorted(freq_list[3:], reverse = True)
    sorted_freq_sum = np.cumsum(sorted_freq_list)
    #freq_opt = np.arange(100,1000,100)
    freq_opt = np.arange(50,500,50)
    for freq in freq_opt:
        OOV_idx= np.where(sorted_freq_list < freq)
        first_idx = OOV_idx[0][0]
        print(first_idx, sorted_freq_list[first_idx], sorted_freq_sum[first_idx]/float(sorted_freq_sum[-1]))
    vocab_opt = np.arange(100000,1000000,100000)
    for vocab_num in vocab_opt:
        print(vocab_num, sorted_freq_list[vocab_num], sorted_freq_sum[vocab_num]/float(sorted_freq_sum[-1]))

