import csv

openreview_name_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_final/AC_name_list"
NIPS_name_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_final/source_data/neurips20_ac_list.csv"

with open(openreview_name_file) as f_in:
    ac_name_list_or = f_in.readlines()
#print(ac_name_list_or)

name_d2_email = {}
with open(NIPS_name_file, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for i, row in enumerate(spamreader):
        if i==0:
            continue
        f_name, l_name, email = row
        author_name = f_name.replace(' ','_') + '_' + l_name.replace(' ','_')
        name_d2_email[author_name] = email

for ac_name_ in ac_name_list_or:
    ac_name = ac_name_[1:len(ac_name_)-len('1.jsonl\tAC\n')]
    if ac_name not in name_d2_email:
        print(ac_name)
#print(name_d2_email)
