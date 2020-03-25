from spacy.lang.en import English
import json
import gzip
import sys
import getopt

help_msg = '-a <asin_user_file> -m <meta_data_file> -o <output_file>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "a:m:o:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-a"):
        asin_user_file = arg
    elif opt in ("-m"):
        meta_data_file = arg
    elif opt in ("-o"):
        output_file = arg

#asin_user_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/book/asin_to_user.tsv"
#meta_data_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/book/meta_Books.json.gz"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/book/all_book_data"
#asin_user_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/toy/asin_to_user.tsv"
#meta_data_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/toy/meta_Toys_and_Games.json.gz"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/toy/all_toy_data"
#asin_user_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/movie/asin_to_user.tsv"
#meta_data_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/movie/meta_Movies_and_TV.json.gz"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/movie/all_movie_data"
#asin_user_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/phone/asin_to_user.tsv"
#meta_data_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/phone/meta_Cell_Phones_and_Accessories.json.gz"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/phone/all_phone_data"
#asin_user_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/cloth/asin_to_user.tsv"
#meta_data_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/cloth/meta_Clothing_Shoes_and_Jewelry.json.gz"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/cloth/all_cloth_data"
#asin_user_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/electronics/asin_to_user.tsv"
#meta_data_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/electronics/meta_Electronics.json.gz"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/electronics/all_electronics_data"

nlp = English()

#lowercase = True
lowercase = False

convert_numbers = True
#convert_numbers = False

num_token = "<NUM>"
sep_token = "<SEP>"
f_sep_token = "<f-SEP>"

record_info_list = ["title", "brand", "main_cat", "category", "feature", "description"]

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def tokenize_text(input_text):
    w_list_org = [w.text for w in nlp.tokenizer( input_text.replace('{','').replace('}','') ) ]
    if convert_numbers:
        w_list = []
        for w in w_list_org:
            if is_number(w):
                w_list.append(num_token)
            else:
                w_list.append(w)
    else:
        w_list = w_list_org
    if lowercase:
        w_list = [x.lower() for x in w_list]
    #output_text = ' '.join(w_list)
    #if lowercase:
    #    output_text = out_sent.lower()
    #return output_text
    return w_list

asin_list = []
asin_d2_user_all = {}
with open(asin_user_file) as f_in:
    for line in f_in:
        asin, user_list, user_list_high = line.split('\t')
        if len(user_list_high) > 0 and user_list_high[-1] == '\n':
            user_list_high = user_list_high[:-1]
        asin_list.append(asin)
        asin_d2_user_all[asin] = [user_list, user_list_high]

asin_set = set(asin_list)
asin_d2_meta = {}

num_skip = 0

with gzip.open(meta_data_file) as f_in:
    for line_count, line in enumerate(f_in):
        if line_count % 10000 == 0:
            sys.stdout.write(str(line_count)+' ')
            sys.stdout.flush()
        #product_data = json.loads(line.replace(b'\\t',b' ').replace(b'\\n',b' ').rstrip())
        product_data = json.loads(line.rstrip())
        #if b'\\n' in line or b'\\t' in line:
        #    print(product_data)
        #    num_skip += 1
        #    continue
        if product_data['asin'] not in asin_set:
            continue
        asin = product_data['asin']
        product_text = []
        product_type = []
        for j, info_name in enumerate(record_info_list):
            if info_name in product_data:
                info = product_data[info_name]
                    
                if info_name == 'title' or info_name == "description" or info_name == "feature":
                    if type(info) == list:
                        output_text = []
                        for info_i in info:
                            output_text_i = tokenize_text(info_i)
                            output_text_i.append(f_sep_token)
                            output_text += output_text_i
                    else:
                        output_text = tokenize_text(info)
                elif info_name == 'brand' or info_name == 'main_cat' or info_name == "category":
                    if type(info) == list:
                        output_text = [ x + ' ' + f_sep_token for x in info]
                    else:
                        output_text = [info]
                output_text.append(sep_token)
                output_text_clean = ' '.join(output_text).replace('\t', ' ').replace('\n', ' ').split()
                #output_text_clean = ' '.join(output_text_clean).split(' ')
                output_type = [str(j)]*len( output_text_clean )
                product_text+=output_text_clean
                product_type+=output_type
        asin_d2_meta[asin] = [' '.join(product_text), ' '.join(product_type)]
        #if line_count > 10000:
        #    break
#print("skip rate: ", num_skip/line_count)
not_find_meta_count = 0
with open(output_file,'w') as f_out:
    for asin in asin_list:
        if asin not in asin_d2_meta:
            not_find_meta_count += 1
            continue
        f_out.write('\t'.join(asin_d2_meta[asin] + asin_d2_user_all[asin]) + '\n')
print("not find meta count", not_find_meta_count)
