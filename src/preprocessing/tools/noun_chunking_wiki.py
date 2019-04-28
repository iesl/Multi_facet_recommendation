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
window_size = 5

grammar = "NP: {<JJ.*>*<VBG>*<NN.*>+}"
cp = nltk.RegexpParser(grammar)

f_out = open(output_file_name,'w')

with open(input_file_name) as f_in:
    total_line_num = 0
    for line in f_in:
        line = line.rstrip()
        if len(line) > 0:
            #try:
            sent, pos = line.rsplit('\t',1)
            sent = sent.replace('\t',' ')
            #except:
            #    print(line)
            w_list = sent.split()
            pos_list = pos.split()
            sent_pos= zip(w_list, pos_list)
            tree = cp.parse(sent_pos)
            iob_tags = nltk.tree2conlltags(tree)
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
                ctx_start_idx = max(0,w_idx-window_size)
                ctx_end_idx = min(len(iob_tags),end_idx+window_size)
                target_w = w_list[ctx_start_idx:w_idx] + w_list[end_idx:ctx_end_idx]
                f_out.write(' '.join(feature_w) + '\n')
                f_out.write(' '.join(target_w) + '\n')    
        total_line_num += 1
        if total_line_num % 10000 == 0:
            sys.stdout.write(str(total_line_num)+' ')
            sys.stdout.flush()
        #if total_line_num > max_line_num:
        #    break

f_out.close()
