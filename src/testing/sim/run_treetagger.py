import sys
sys.path.insert(0, sys.path[0]+'/treetagger-python')
from treetagger import TreeTagger
import re
import string

tt = TreeTagger(path_to_treetagger='/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/src/testing/sim/tree_tagger')
print(tt.get_installed_lang())

file_path = "/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/SCWS/scws_only_sent"
output_path = "/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/SCWS/scws_org_lemma"

filter_set = set(['<b>','</b>','</font','style>'])


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

output_list = []

with open(file_path) as f_in:
    for line in f_in:
        w_list = cleanhtml(line.rstrip()).split()
        #print(w_list)
        sent_filtered = ' '.join([x for x in w_list if x not in filter_set])
        output_list.append([line.rstrip(), sent_filtered])

org_sent, input_sent = zip(*output_list)
tagged_results = tt.tag(input_sent)
w_ind = 0
for i_out, sent in enumerate(input_sent):
    lemma_sent = []
    w_ind += 2 #skip ['(', 'PUL', '('], ["'", 'PUQ', "'"] or [',', 'PUN', ','], ["'", 'POS', "'"]
    #print(sent)
    for w in sent.split():
        #try: 
        #    tagged_results[w_ind][1]
        #except:
        #    print(w)
        #    print(sent)
        #    print(tagged_results[w_ind])
        if tagged_results[w_ind][1] == 'CJC' and tagged_results[w_ind][0] == '\\':
            w_ind += 1 
        #if w == "'s" and tagged_results[w_ind][0] == "'":
        #    lemma_sent.append("'s")
        #    w_ind += 2
        #    continue
        if w != tagged_results[w_ind][0].replace('\\',''): #handle the case where the tagger split the token such as MD.
            lemma_sent.append(w)
            current_tok = tagged_results[w_ind][0].replace('\\','')
            #print(w)
            #print(sent)
            #print(tagged_results[w_ind])
            #print(tagged_results[w_ind-5:w_ind+5])
            max_split = 5
            for i in range(max_split):
                w_ind += 1
                current_tok += tagged_results[w_ind][0].replace('\\','')
                if current_tok == w:
                    break
            w_ind += 1
            if i == max_split - 1:
                sys.exit(1)
            continue

        if w != tagged_results[w_ind][0].replace('\\',''):
            print(w)
            print(tagged_results[w_ind-5:w_ind+5])
            print(tagged_results[w_ind])
            #print(tagged_results[:10])
            print(sent)
            sys.exit(1)
        if tagged_results[w_ind][2] != '<unknown>':
            if str.isupper(w[0]) and tagged_results[w_ind-1][1] != 'SENT' and tagged_results[w_ind-1][1] != 'POS':
                lemma_sent.append(w)
            else:
                opt_ind = tagged_results[w_ind][2].find('|')
                if opt_ind>0:
                    lemma_sent.append(tagged_results[w_ind][2][:opt_ind])
                else:
                    lemma_sent.append(tagged_results[w_ind][2])
        w_ind += 1
    w_ind += 1 #skip ["'", 'POS', "'"] 
    output_list[i_out][1] = ' '.join(lemma_sent)

#tagged_words = tt.tag(sent_filtered)
#w_list, pos_list, lemma_list = zip(*tagged_words)
#lemma_sent = ' '.join(lemma_list)
#output_list.append([line.rstrip(), lemma_sent])

with open(output_path,'w') as f_out:
    for org_sent, lemma_sent in output_list:
        f_out.write(org_sent+'\t'+lemma_sent+'\n')
