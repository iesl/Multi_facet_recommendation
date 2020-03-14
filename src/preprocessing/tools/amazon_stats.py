
import numpy as np
input_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/amazon/book/Books.csv"

user_d2_product = {}
with open(input_file) as f_in:
    for line in f_in:
        asin, user, rate = line.rstrip().split(',')
        asin_list = user_d2_product.get(user, [])
        asin_list.append(asin)
        user_d2_product[user] = asin_list

print(len(user_d2_product))
user_count = []
all_asin_list = []
asin_d2_count = {}
for user in user_d2_product:
    asin_list = user_d2_product[user]
    count = len(asin_list)
    if count > 5:
        user_count.append([user,count])
        all_asin_list += asin_list
        for asin in asin_list:
            user_count_a = asin_d2_count.get(asin,0)
            asin_d2_count[asin] = user_count_a + 1
print(len(user_count))
#user_count_sorted = sorted(user_count, key=lambda x:x[1],reverse = True)
user_list, count_list = zip(*user_count)
#count_list = user_count.values()
asin_set = set(all_asin_list)
print(sum(count_list))

print(np.quantile(count_list, np.linspace(0,1,10) ))

print(len(asin_set))
print( np.quantile(list(asin_d2_count.values()), np.linspace(0,1,10)) )
