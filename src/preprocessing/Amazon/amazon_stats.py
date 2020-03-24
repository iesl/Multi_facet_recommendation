
import numpy as np

help_msg = '-i <input_file> -o <output_file>'

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
        input_file = arg
    elif opt in ("-o"):
        output_file = arg

#input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/book/Books.csv"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/book/asin_to_user.tsv"
#input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/toy/Toys_and_Games.csv"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/toy/asin_to_user.tsv"
#input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/movie/Movies_and_TV.csv"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/movie/asin_to_user.tsv"
#input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/phone/Cell_Phones_and_Accessories.csv"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/phone/asin_to_user.tsv"
#input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/cloth/Clothing_Shoes_and_Jewelry.csv"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/cloth/asin_to_user.tsv"
#input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/electronics/Electronics.csv"
#output_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/electronics/asin_to_user.tsv"

user_d2_product = {}
user_d2_product_high = {}
with open(input_file) as f_in:
    for line in f_in:
        asin, user, rate = line.rstrip().split(',')
        asin_list = user_d2_product.get(user, [])
        asin_list.append(asin)
        user_d2_product[user] = asin_list
        if float(rate) >= 4:
            asin_list = user_d2_product_high.get(user, [])
            asin_list.append(asin)
            user_d2_product_high[user] = asin_list


print("Number of user before filtering", len(user_d2_product))
user_count = []
#all_asin_list = []
asin_d2_count = {}
asin_d2_user = {}
asin_high_d2_user = {}
for user in user_d2_product:
    asin_list = user_d2_product[user]
    asin_list_high = user_d2_product_high.get(user,[])

    count = len(asin_list)
    if count > 5:
        user_count.append([user,count])
        #all_asin_list += asin_list
        for asin in asin_list:
            user_count_a = asin_d2_count.get(asin,0)
            asin_d2_count[asin] = user_count_a + 1
            user_list = asin_d2_user.get(asin,[])
            user_list.append(user)
            asin_d2_user[asin] = user_list
        for asin_high in asin_list_high:
            user_list = asin_high_d2_user.get(asin_high,[])
            user_list.append(user)
            asin_high_d2_user[asin_high] = user_list
            
print("Number of user after filtering", len(user_count))
#user_count_sorted = sorted(user_count, key=lambda x:x[1],reverse = True)
user_list, count_list = zip(*user_count)
#count_list = user_count.values()
#asin_set = set(all_asin_list)
print("Number of rating after filtering", sum(count_list))
print("ASIN count per user", np.quantile(count_list, np.linspace(0,1,10) ))

with open(output_file, 'w') as f_out:
    for asin in asin_d2_user:
        user_list = asin_d2_user[asin]
        user_list_high = asin_high_d2_user.get(asin, [])
        f_out.write(asin + '\t' + ','.join(user_list) + '\t' + ','.join(user_list_high)  + '\n' )
        
#print(len(asin_set))
print("Number of asin after filtering", len(list(asin_d2_count.keys())))
print("user count per ASIN", np.quantile(list(asin_d2_count.values()), np.linspace(0,1,10)) )

