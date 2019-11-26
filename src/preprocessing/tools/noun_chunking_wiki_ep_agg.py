import os
from collections import Counter
import sys

input_sent_dir = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_POS"
input_entity_dir = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_nchunk_entities"
#input_entity_dir = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/temp"
#output_dir = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_nchunk_ep_agg"
output_dir = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_nchunk_entity_agg"

f_out_sent = open(output_dir + '/wiki2016_sents', 'w')
#f_out_ep = open(output_dir + '/wiki2016_ep', 'w')
f_out_ent = open(output_dir + '/wiki2016_ent', 'w')
f_out_idx = open(output_dir + '/wiki2016_idx', 'w')

max_span = 2

current_idx = 0
#ep_d2_idx = {}
#idx_l2_ep = []
ent_d2_idx = {}
idx_l2_ent = []
wiki_pieces = os.listdir(input_entity_dir)
for file_name in wiki_pieces:
    print("Processing "+ file_name)
    sys.stdout.flush()
    sent_list = []
    with open(input_sent_dir + '/' + file_name) as f_in_s:
        for line in f_in_s:
            fields = line.rstrip().split('\t',1)
            #print(fields)
            if len(fields) == 1:
                sent_list.append('')
            else:
                sent, pos = fields
                sent = sent.replace('\t',' ')
                sent_list.append(sent)
    sent_ep_list = []
    with open(input_entity_dir + '/' + file_name) as f_in_e:
        for line_idx, line in enumerate(f_in_e):
            if line_idx % 10000 == 0:
                print(str(line_idx)+' ',)
                sys.stdout.flush()
            fields = line.rstrip().split('\t')
            if len(fields) <= 2:
                continue
            sent_idx = int(fields[0]) + current_idx
            ent_count = Counter(fields[1:])
            ent_list = []
            for j in range(1,len(fields)):
                ent = fields[j]
                if ent_count[ent] > 1:
                    continue
                ent_list.append(ent)
            if len(ent_list) <= 1:
                continue
            ent_idx_list = []
            for ent in ent_list:
                if ent not in ent_d2_idx:
                    ent_idx = len(idx_l2_ent)
                    ent_d2_idx[ent] = ent_idx
                    idx_l2_ent.append(ent)
                else:
                    ent_idx = ent_d2_idx[ent]
                ent_idx_list.append(ent_idx)
            for j in range( len(ent_idx_list) - 1 ):
                for k in range( j+1, min(j+1+max_span,len(ent_idx_list)) ):
                    sent_ep_list.append( [ent_idx_list[j], ent_idx_list[k], sent_idx] )
                    
            #ep_line_set = set()
            #for j in range(1,len(fields)):
            #    ent = fields[j]
            #    if ent_count[ent] > 1:
            #        continue
            #    for k in range( j+1,min(j+1+max_span,len(fields)) ):
            #        ent_next = fields[k]
            #        if ent_count[ent_next] > 1:
            #            continue
            #        ep = (ent, ent_next)
            #        #if ep in ep_line_set:
            #        #    continue
            #        #else:
            #        #    ep_line_set.add(ep)
            #        if ep not in ep_d2_idx:
            #            ep_idx = len(idx_l2_ep)
            #            ep_d2_idx[ep] = ep_idx
            #            idx_l2_ep.append(ep)
            #        else:
            #            ep_idx = ep_d2_idx[ep]
            #        sent_ep_list.append( [ep_idx, sent_idx] )
            #        #print(sent_ep_list)
            #sys.exit()
    for ent1_idx, ent2_idx, sent_idx in sent_ep_list:
        f_out_idx.write(str(ent1_idx)+'\t'+str(ent2_idx)+'\t'+str(sent_idx)+'\n')
    #for ep_idx, sent_idx in sent_ep_list:
    #    f_out_idx.write(str(ep_idx)+'\t'+str(sent_idx)+'\n')

    for sent in sent_list:
        f_out_sent.write(sent+'\n')
        current_idx += 1

for ent in idx_l2_ent:
    f_out_ent.write(ent+'\n')
#for ep in idx_l2_ep:
#    f_out_ep.write('\t'.join(ep)+'\n')

f_out_idx.close()
f_out_ent.close()
f_out_sent.close()
