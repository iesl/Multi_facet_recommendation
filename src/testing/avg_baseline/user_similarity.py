import sys
sys.path.insert(0, sys.path[0]+'/../../')

import utils
import torch 
import numpy as np

AC_name_file = '/mnt/nfs/scratch1/hschang/recommend/Multi_facet_recommendation/data/processed/NeurIPS2020_gorc_uncased/AC_name_list'
user_dict_path = "./data/processed/NeurIPS2020_gorc_uncased/user/dictionary_index"
#user_emb_file = './models/gorc-20200423-155956/user_emb_NeurIPS2020_always.pt'
#output_file = './gen_log/NeurIPS2020_AC_SAC_n5.csv'
user_emb_file = './models/gorc-20200423-155505/user_emb_NeurIPS2020_always.pt'
output_file = './gen_log/NeurIPS2020_AC_SAC_n1.csv'

use_gpu = True
device = torch.device("cuda" if use_gpu else "cpu")

with open(user_dict_path) as f_in:
    user_idx2word_freq = utils.load_idx2word_freq(f_in)
    
print(user_emb_file)
user_emb, user_emb_size = utils.load_emb_from_path(user_emb_file, device, user_idx2word_freq)

user_emb.requires_grad = False
user_norm_emb = user_emb / (0.000000000001 + user_emb.norm(dim = 1, keepdim=True) )
user_norm_emb.requires_grad = False

name_to_role = {}
AC_num = 0
SAC_num = 0
with open(AC_name_file) as f_in:
    for line in f_in:
        name, role = line.rstrip().split('\t')
        name_to_role[name] = role
        if role == 'AC':
            AC_num += 1
        else:
            assert  role == 'SAC'
            SAC_num += 1

SAC_list = []
AC_list = []
AC_tensor = torch.zeros((AC_num,user_emb_size),device = device, requires_grad = False)
SAC_tensor = torch.zeros((SAC_num,user_emb_size),device = device, requires_grad = False)
for i in range(len(user_idx2word_freq)):
    user, freq = user_idx2word_freq[i]
    if user[-1] == '|':
        user = user[:-1]
    if user not in name_to_role:
        print(user)
        continue
    #assert user[-1] == '|'
    #user = user[:-1]
    role = name_to_role[user]
    if role == 'AC':
        AC_tensor[len(AC_list),:] = user_norm_emb[i,:]
        AC_list.append(user)
    else:
        assert  role == 'SAC'
        SAC_tensor[len(SAC_list),:] = user_norm_emb[i,:]
        SAC_list.append(user)

user_sim_np = torch.mm(AC_tensor, SAC_tensor.t()).cpu().numpy()
with open(output_file, 'w') as f_out:
    for i, AC_name in enumerate(AC_list):
        for j, SAC_name in enumerate(SAC_list):
            f_out.write(AC_name+','+SAC_name+','+str(user_sim_np[i,j])+'\n')
