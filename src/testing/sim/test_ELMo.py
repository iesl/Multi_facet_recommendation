import torch
from allennlp.commands.elmo import ElmoEmbedder

print(torch.version.cuda)
print(torch.backends.cudnn.version())

input_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/phrase/WikiSRS_rel_sim_test"
output_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/gen_log/ELMo_WikiSRS_rel_sim_phrase_test.json"
#input_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/stsbenchmark/sts-dev_org"
#output_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/gen_log/BERT_large_sts-dev_cased.json"
#input_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/stsbenchmark/sts-train_org"
#output_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/gen_log/BERT_large_sts-train_cased.json"
#input_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/STS/stsbenchmark/sts-test_org"
#output_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/gen_log/BERT_large_sts-test_cased.json"

#output_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/gen_log/BERT_large_sts-test_cased.json"
#input_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/phrase/BiRD_test"
#output_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/gen_log/ELMo_base_BiRD_phrase_test.json
#input_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/phrase/SemEval2013_Turney2012_phrase_test_org_unique"
#output_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/gen_log/BERT_large_SemEval2013_Turney2012_phrase_test.json"
#input_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/phrase/HypeNet_WordNet_val_org_unique"
#output_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/gen_log/BERT_large_HypeNet_WordNet_phrase_val.json"
#input_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/phrase/HypeNet_WordNet_test_org_unique"
#output_path = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/gen_log/BERT_large_HypeNet_WordNet_phrase_test.json"

elmo = ElmoEmbedder(cuda_device=0)
#tokens = ["I", "ate", "an", "apple", "for", "breakfast"]
#vectors = elmo.embed_sentence(tokens)
#print(vectors)

org_sent_list = []
proc_sent_list = []

with open(input_path) as f_in:
    for line in f_in:
        org_sent, proc_sent = line.rstrip().split('\t')
        org_sent_list.append(org_sent.split())
        proc_sent_list.append(proc_sent.split())
 
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
    org_sent_inner.append(org_sent_list[i])
    proc_sent_inner.append(proc_sent_list[i])
    sent_len = len(proc_sent_list[i])
    length_list.append(sent_len)
    if storing_idx == batch_size - 1 or i == len(org_sent_list)-1:
        activations, mask = elmo.batch_to_embeddings(proc_sent_inner)
    
        for inner_i in range(len(org_sent_inner)):
            sent_len = length_list[inner_i]
            first_layer = activations[inner_i,0,:sent_len,:]
            last_layer = activations[inner_i,-1,:sent_len,:]
            avg_emb = last_layer.mean(dim=0)
            proc_emb_last = torch.cat([ last_layer[0,:], last_layer[sent_len,:], last_layer[0,:] - last_layer[sent_len,:], last_layer[0,:]*last_layer[sent_len,:] ])
            proc_emb_first = torch.cat([ first_layer[0,:], first_layer[sent_len,:], first_layer[0,:] - first_layer[sent_len,:], first_layer[0,:]*first_layer[sent_len,:] ])
            proc_emb = torch.cat([proc_emb_last, proc_emb_first])
            output_list.append([ org_sent_inner[inner_i], proc_sent_inner[inner_i], avg_emb.tolist(), proc_emb.tolist() ])

with open(output_path, 'w') as outf:
    json.dump(output_list, outf, indent = 1)
