import nltk
import getopt
import sys

help_msg = '-i <input_file_path> -o <output_file_path>'


try:
    opts, args = getopt.getopt(sys.argv[1:], "i:o:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-i"):
        input_file_name = arg
    elif opt in ("-o"):
        output_file_name = arg

#input_file_name = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016_POS/enwik0"
#output_file_name = "temp_chunk"
#max_line_num = 10000

grammar = "NP: {<JJ.*>*<VBG>*<NN.*>+}"
cp = nltk.RegexpParser(grammar)

f_out = open(output_file_name,'w')

with open(input_file_name) as f_in:
    total_line_num = 0
    for line_idx, line in enumerate(f_in):
        line = line.rstrip()
        if len(line) > 0:
            #try:
            sent, pos = line.rsplit('\t',1)
            sent = sent.replace('\t',' ')
            #except:
            #    print(line)
            w_list = sent.split()
            pos_list = pos.split()
            sent_pos= list(zip(w_list, pos_list))
            tree = cp.parse(sent_pos)
            iob_tags = nltk.tree2conlltags(tree)
            entity_list = []
            #entity_set = set()
            for w_idx, w_info in enumerate(iob_tags):
                if w_info[2] != 'B-NP':
                    continue
                end_idx = w_idx
                for j in range(w_idx+1, len(iob_tags) ):
                    if iob_tags[j][2] != 'I-NP':
                        break
                    else:
                        end_idx = j
                end_idx += 1
                feature_w = w_list[w_idx:end_idx]
                entity_list.append(feature_w)
                #if tuple(feature_w) not in entity_set:
                #    entity_list.append(feature_w)
                #    entity_set.add(tuple(feature_w))
            if len(entity_list) > 0:
                f_out.write(str(line_idx) + '\t' + '\t'.join([' '.join(entity) for entity in entity_list]) + '\n')
        total_line_num += 1
        if total_line_num % 10000 == 0:
            sys.stdout.write(str(total_line_num)+' ')
            sys.stdout.flush()
        #if total_line_num > max_line_num:
        #    break

f_out.close()
