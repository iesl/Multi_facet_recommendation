import torch
from allennlp.commands.elmo import ElmoEmbedder
import sys
import json
import sys
sys.path.insert(0,sys.path[0]+'/../..')
from utils import load_idx2word_freq, load_word_dict
from utils_testing import compute_freq_prob

print(torch.version.cuda)
print(torch.backends.cudnn.version())

weighted_by_prob = True
#weighted_by_prob = False

weighted_by_w_sim = True
#weighted_by_w_sim = False

root_dir = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/"
root_dir = "/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/"

freq_dict_path = root_dir+"/data/processed/wiki2016_min100/dictionary_index"

#input_path = root_dir+"/dataset_testing/phrase/WikiSRS_rel_sim_test"
#input_path = root_dir+"/dataset_testing/phrase/WikiSRS_rel_sim_test_upper"
#output_path = root_dir+"/gen_log/ELMo_WikiSRS_rel_sim_phrase_test.json"

#input_path = root_dir+"/dataset_testing/SNLI/snli_1.0_dev_org"
#output_path = root_dir+"/gen_log/ELMo_snli-dev_cased.json"
#input_path = root_dir+"/dataset_testing/SNLI/snli_1.0_test_org"
#output_path = root_dir+"/gen_log/ELMo_snli-test_cased.json"
#input_path = root_dir+"/dataset_testing/STS/stsbenchmark/sts-dev_org"
#output_path = root_dir+"/gen_log/ELMo_large_sts-dev_cased.json"
#input_path = root_dir+"/dataset_testing/STS/stsbenchmark/sts-train_org"
#output_path = root_dir+"/gen_log/ELMo_large_sts-train_cased.json"
input_path = root_dir+"/dataset_testing/STS/stsbenchmark/sts-test_org"
w_sim_path = root_dir+"/gen_log/STS_test_wiki2016_glove_trans_n1_bsz200_ep2_2_final.json"
#w_sim_path = root_dir+"/gen_log/STS_test_wiki2016_glove_trans_n10_bsz200_ep2_0_final.json"
output_path = root_dir+"/gen_log/ELMo_w_sim_n1_sts-test_cased.json"
#input_path = root_dir+"/dataset_testing/STS/stsbenchmark/sts-dev_org"
#w_sim_path = root_dir+"/gen_log/STS_dev_wiki2016_glove_trans_n10_bsz200_ep2_0_final.json"
#w_sim_path = root_dir+"/gen_log/STS_dev_wiki2016_glove_trans_n1_bsz200_ep2_2_final.json"
#output_path = root_dir+"/gen_log/ELMo_w_sim_n1_sts-dev_cased.json"
#input_path = root_dir+"/dataset_testing/STS/sts_train_2012_org"
#output_path = root_dir+"/gen_log/ELMo_sts_2012_train_cased.json"
#input_path = root_dir+"/dataset_testing/STS/sts_test_all_org"
#output_path = root_dir+"/gen_log/ELMo_sts_2012-6_test_cased.json"
#input_path = root_dir+"/dataset_testing/phrase/BiRD_test"
#output_path = root_dir+"/gen_log/ELMo_BiRD_phrase_test.json"
#input_path = root_dir+"/dataset_testing/phrase/SemEval2013_Turney2012_phrase_test_org_unique"
#output_path = root_dir+"/gen_log/ELMo_large_SemEval2013_Turney2012_phrase_test.json"
#input_path = root_dir+"/dataset_testing/phrase/HypeNet_WordNet_val_org_unique"
#output_path = root_dir+"/gen_log/ELMo_large_HypeNet_WordNet_phrase_val.json"
#input_path = root_dir+"/dataset_testing/phrase/HypeNet_WordNet_test_org_unique"
#output_path = root_dir+"/gen_log/ELMo_large_HypeNet_WordNet_phrase_test.json"

elmo = ElmoEmbedder(cuda_device=0)
#tokens = ["I", "ate", "an", "apple", "for", "breakfast"]
#vectors = elmo.embed_sentence(tokens)
#print(vectors)

word_d2_idx_freq = None
if weighted_by_prob:
    with open(freq_dict_path) as f_in:
        word_d2_idx_freq, max_ind = load_word_dict(f_in)
    compute_freq_prob(word_d2_idx_freq)
    alpha = 0.0001

sent_d2_w_sim = {}
if weighted_by_w_sim:
    with open(w_sim_path) as f_in:
        input_json= json.load(f_in)
        for input_dict in input_json:
            org_sent = input_dict['org_sent']
            w_sim = input_dict['w_imp_sim']
            sent_d2_w_sim[org_sent] = [float(x) for x in w_sim[:-1]] #exclude <eos> at the end

org_sent_list = []
proc_sent_list = []
freq_prob_list = []
w_sim_list = []
with open(input_path) as f_in:
    for line in f_in:
        org_sent, proc_sent = line.rstrip().split('\t')
        org_sent_list.append(org_sent)
        proc_sent_list.append(proc_sent.split())
        if word_d2_idx_freq is not None:
            freq_prob_list_i = []
            for w in proc_sent.split():
                if w in word_d2_idx_freq:
                    w_idx, freq, freq_prob = word_d2_idx_freq[w]
                    freq_prob_list_i.append( alpha / (alpha + freq_prob) )
                else:
                    freq_prob_list_i.append(0)
            if sum(freq_prob_list_i) == 0:
                print(line)
            freq_prob_list.append(freq_prob_list_i)
        if len(sent_d2_w_sim) > 0:
            w_sim_i = sent_d2_w_sim[org_sent]
            w_sim_list.append(w_sim_i)
 
batch_size = 100       
output_list = []
for i in range(len(org_sent_list)):
    storing_idx = i % batch_size
    if storing_idx == 0:
        sys.stdout.write(str(i)+' ')
        sys.stdout.flush()
        length_list = []
        org_sent_inner = []
        proc_sent_inner = []
        freq_prob_inner = []
        w_sim_inner = []
        #freq_prob_tensor = torch.empty(batch_size, max_len, device = device, dtype = torch.float)
    org_sent_inner.append(org_sent_list[i])
    proc_sent_inner.append(proc_sent_list[i])
    sent_len = len(proc_sent_list[i])
    length_list.append(sent_len)
    if len(freq_prob_list) > 0:
        freq_prob_inner.append(freq_prob_list[i])
    if len(w_sim_list) > 0:
        w_sim_inner.append(w_sim_list[i])
        #freq_prob_tensor[storing_idx, :sent_len] = torch.tensor(freq_prob_list[i], device = device)
    if storing_idx == batch_size - 1 or i == len(org_sent_list)-1:
        activations, mask = elmo.batch_to_embeddings(proc_sent_inner)
    
        for inner_i in range(len(org_sent_inner)):
            sent_len = length_list[inner_i]
            first_layer = activations[inner_i,0,:sent_len,:]
            last_layer = activations[inner_i,-1,:sent_len,:]
            weights_inner = torch.tensor([1.0]*sent_len, device = 'cuda')
            if len(freq_prob_list) > 0:
                weights_inner *= torch.tensor(freq_prob_inner[inner_i], device = 'cuda')
            if len(w_sim_list) > 0:
                weights_inner *= torch.tensor(w_sim_inner[inner_i], device = 'cuda')
            avg_emb = torch.sum(activations[inner_i,-1,:sent_len,:]*weights_inner.unsqueeze(-1) , dim = 0).squeeze() / (torch.sum(weights_inner) + 1e-12)

            #if len(freq_prob_list) > 0:
            #    freq_prob_tensor = torch.tensor(freq_prob_inner[inner_i], device = 'cuda')
            #    avg_emb = torch.sum(activations[inner_i,-1,:sent_len,:]*freq_prob_tensor.unsqueeze(-1) , dim = 0).squeeze() / (torch.sum(freq_prob_tensor) + 1e-12)
            #else:
            #    avg_emb = last_layer.mean(dim=0)
            proc_emb_last = torch.cat([ last_layer[0,:], last_layer[sent_len-1,:], last_layer[0,:] - last_layer[sent_len-1,:], last_layer[0,:]*last_layer[sent_len-1,:] ])
            proc_emb_first = torch.cat([ first_layer[0,:], first_layer[sent_len-1,:], first_layer[0,:] - first_layer[sent_len-1,:], first_layer[0,:]*first_layer[sent_len-1,:] ])
            proc_emb = torch.cat([proc_emb_last, proc_emb_first])
            output_list.append([ org_sent_inner[inner_i], proc_sent_inner[inner_i], avg_emb.tolist(), proc_emb.tolist() ])

with open(output_path, 'w') as outf:
    json.dump(output_list, outf, indent = 1)
