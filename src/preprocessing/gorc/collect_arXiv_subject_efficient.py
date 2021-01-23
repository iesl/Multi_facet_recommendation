import requests
import pandas as pd
import csv
import xml.etree.ElementTree as ET
import getopt
import sys

help_msg = '-o <output_path>'

try:
    opts, args = getopt.getopt(sys.argv[1:], "o:")
except getopt.GetoptError:
    print(help_msg)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print(help_msg)
        sys.exit()
    elif opt in ("-o"):
        output_path = arg

domain_list = ['cs.AI', 'cs.CL', 'cs.CV', 'cs.IR', 'cs.IT', 'cs.LG', 'cs.MA', 'cs.NE', 'cs.RO', 'stats.ML']
#domain_list = ['cs.NE']
#output_path = './arXiv_id_ML'

batch_size = 1000

all_cat_id = ['domain\tid']
for domain in domain_list:
    i = 0
    print(domain)
    while(1):
        print(i*batch_size,)
        url = 'http://export.arxiv.org/api/query?search_query=cat:'+domain+'&start='+str(i*batch_size)+'&max_results='+str(batch_size)
        i += 1
        cat_id = []

        myfile = requests.get(url)
        root = ET.fromstring(myfile.content)
        #print(root[0])
        prefix = 'http://arxiv.org/abs/'
        for subject in root.iter('{http://www.w3.org/2005/Atom}id'):
            if subject.text[:len(prefix)] != prefix:
                continue
            cat_id.append( domain + '\t'+subject.text[len(prefix):] )
            #print(subject.text[len(prefix):])
            #subject_list = []
            #for subject in root.iter('{http://purl.org/dc/elements/1.1/}subject'):
            #    #print(subject.text)
            #    subject_list.append(subject.text)
            #id_subject.append([arxiv_id, '|'.join(subject_list)])
            #print(id_subject)
        if len(cat_id) == 0:
            break
        else:
            all_cat_id+=cat_id
with open(output_path,'w') as f_out:
    f_out.write('\n'.join(all_cat_id) )
