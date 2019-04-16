import torch

subsampling_ratio = 2

input_file = "/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/data/processed/wiki2016_min100/tensors/train.pt"
output_file = "/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/data/processed/wiki2016_min100/tensors/train_sub"+str(subsampling_ratio)+".pt"

with open(input_file,'rb') as f_in:
    feature, target = torch.load(f_in)

def subsampling(tensor_in, subsampling_ratio):
    sample_num = tensor_in.size(0)
    return tensor_in[0:sample_num:subsampling_ratio,:].clone()

print(feature.size(0))
feature_sub = subsampling(feature, subsampling_ratio)
target_sub = subsampling(target,subsampling_ratio)
print(feature_sub.size(0))

with open(output_file,'wb') as f_out:
    torch.save([feature_sub,target_sub],f_out)
