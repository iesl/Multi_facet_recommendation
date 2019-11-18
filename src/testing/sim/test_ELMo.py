import torch
from allennlp.commands.elmo import ElmoEmbedder

print(torch.version.cuda)

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
tokens = ["I", "ate", "an", "apple", "for", "breakfast"]
vectors = elmo.embed_sentence(tokens)
print(vectors)
