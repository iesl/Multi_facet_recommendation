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

#input_file_name = "/iesl/canvas/hschang/language_modeling/NSD_for_sentence_embedding/data/raw/wiki2016.txt"
#output_file_name = "temp_POS"
#max_line_num = 1000000

f_out = open(output_file_name,'w')

with open(input_file_name) as f_in:
    total_line_num = 0
    for line in f_in:
        
        line = line.rstrip()
        tokens = line.split()
        if len(tokens) > 0:
            sent_pos = nltk.pos_tag(tokens)
            #print(sent_pos)
            sent, pos = zip(*sent_pos)
            f_out.write(line + '\t' + ' '.join(pos)+'\n')
        else:
            f_out.write('\n')
        total_line_num += 1
        if total_line_num % 10000 == 0:
            sys.stdout.write(str(total_line_num)+' ')
            sys.stdout.flush()
        #if total_line_num > max_line_num:
        #    break

f_out.close()
