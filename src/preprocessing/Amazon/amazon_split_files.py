import sys
import getopt

#remove the sentences which do not have correct title
#remove the length mismatch

help_msg = '-i <input_path> -f <out_feature_path> -y <out_type_path> -u <out_user_path> -t <out_tag_path> -b <out_bid_path>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:f:y:u:t:b:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-i"):
        input_path = arg
    elif opt in ("-f"):
        out_feature_path = arg
    elif opt in ("-y"):
        out_type_path = arg
    elif opt in ("-u"):
        out_user_path = arg
    elif opt in ("-t"):
        out_tag_path = arg
    elif opt in ("-b"):
        out_bid_path = arg

feature_list = []
type_list = []
user_list = []
tag_list = []
bid_score_list = []
cut_line_num = 0
total_line_num = 0
mismatch_num = 0
sucessful_num = 0
weird_title_num = 0
with open(input_path) as f_in:
    for line in f_in:
        fields = line.split('\t')
        total_line_num += 1
        if len(fields) != 4 and len(fields) != 5:
            cut_line_num += 1
            continue
        if len(fields) == 4:
            meta_info, type_info, user_list_str, user_list_high_str = fields
        else:
            meta_info, type_info, user_list_str, user_list_high_str, bid_score_str = fields
        #meta_info = ' '.join(meta_info.split())
        weird_title_prefix = "var aPageStart"
        if meta_info[:len(weird_title_prefix)] == weird_title_prefix:
            weird_title_num += 1
            continue
        if len(user_list_high_str) > 0 and user_list_high_str[-1] == '\n':
            user_list_high_str = user_list_high_str[:-1]
        w_num = len(meta_info.split(' '))
        t_num = len(type_info.split(' '))
        if w_num != t_num:
            print(w_num, t_num)
            print(meta_info)
            print(type_info)
            sys.exit(1)
            mismatch_num += 1
            continue
        sucessful_num += 1
        feature_list.append(meta_info)
        type_list.append(type_info)
        user_list.append(user_list_str.replace(' ','_').replace(',',' '))
        tag_list.append(user_list_high_str.replace(' ','_').replace(',',' '))
        if len(fields) == 5:
            bid_score = bid_score_str.replace(',',' ')
            #bid_score = list(map( int,bid_score_str.split(',') ))
            bid_score_list.append(bid_score)
            assert len(bid_score_list[-1].split(' ')) == len(user_list[-1].split(' ')), print(bid_score_list[-1], user_list[-1], len(user_list))

print("weird title rate: ", weird_title_num / float(total_line_num-cut_line_num))
print("cut line rate: ", cut_line_num / float(total_line_num))
print("mismatch rate: ", mismatch_num / float(sucessful_num+mismatch_num))

with open(out_feature_path, 'w') as f_out:
    for feature in feature_list:
        f_out.write(feature+'\n')

with open(out_type_path, 'w') as f_out:
    for type_info in type_list:
        f_out.write(type_info+'\n')

with open(out_user_path, 'w') as f_out:
    for user in user_list:
        f_out.write(user+'\n')

with open(out_tag_path, 'w') as f_out:
    for tag in tag_list:
        f_out.write(tag+'\n')

if len(bid_score_list) > 0:
    with open(out_bid_path, 'w') as f_out:
        for bid in bid_score_list:
            f_out.write(bid+'\n')
