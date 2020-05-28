import os 

#AC_dir_parent = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020/source_data/'
#AC_dir = AC_dir_parent + 'archivesACs/'
#SAC_dir = AC_dir_parent + 'archivesSACs/'
#out_AC_names = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020/AC_name_list'

AC_dir_parent = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_final/source_data/'
AC_dir = AC_dir_parent + 'archives/'
#AC_dir = AC_dir_parent + 'archivesACs/'
#SAC_dir = AC_dir_parent + 'archivesSACs/'
out_AC_names = '/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/openreview/NeurIPS2020_final/AC_name_list'

AC_file_list = os.listdir(AC_dir)
#SAC_file_list = os.listdir(SAC_dir)
#print(SAC_file_list)

def dump_names(f_out, AC_file_list, ac_tag):
    for AC_name in AC_file_list:
        f_out.write('{}\t{}\n'.format(AC_name, ac_tag))

with open(out_AC_names, 'w') as f_out:
    dump_names(f_out, AC_file_list, "AC")
    #dump_names(f_out, SAC_file_list, "SAC")
