import sys
sys.path.insert(0, sys.path[0]+'/../..')
import torch
from utils import load_idx2word_freq

#emb_file_path = "./models/Wacky-20190329-154753/target_emb.pt"
#dict_file_path = "data/processed/wackypedia/dictionary_index"
#out_file_path = "./models/Wacky-20190329-154753/target_emb.txt"

emb_file_path = "./models/wiki2016-20190409-112932/target_emb.pt"
dict_file_path = "data/processed/wiki2016_min500/dictionary_index"
out_file_path = "./models/wiki2016-20190409-112932/target_emb.txt"


word_emb = torch.load( emb_file_path, map_location='cpu' )

with open(dict_file_path) as f_in:
    idx2word_freq = load_idx2word_freq(f_in)

print(word_emb.size(0))
print(len(idx2word_freq))
assert word_emb.size(0) == len(idx2word_freq)

with open(out_file_path, 'w') as f_out:
    f_out.write(str(len(idx2word_freq))+' 300\n')
    for i in range(len(idx2word_freq)):
        w = idx2word_freq[i][0]
        emb = word_emb[i,:].tolist()
        f_out.write( w + ' ' + ' '.join([str(x) for x in emb]) + '\n')
