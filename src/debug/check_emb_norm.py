import sys
sys.path.insert(0, sys.path[0]+'/..')
import utils
import torch

dictionary_input_name = "./data/processed/wackypedia/dictionary_index"

device = 'cpu'
emb_file = './resources/Google-vec-neg300_filtered_wac_bookp1.txt'

with open(dictionary_input_name) as f_in:
    idx2word_freq = utils.load_idx2word_freq(f_in)

external_emb, output_emb_size, oov_list = utils.load_emb_file(emb_file, device, idx2word_freq)

external_emb_norm = external_emb.norm(dim = 1)
external_emb_norm[external_emb_norm == 0] = 100
smallest_value, smallest_index = torch.topk(-external_emb_norm, 100)
print(smallest_value)
print(smallest_index)
