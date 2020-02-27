
file_format = 'a'
#file_format = 't'

if file_format == 'a':
    input_users_file = "/iesl/canvas/hschang/recommendation/ctrsr_datasets/citeulike-a/users.dat"
    input_tags_file = "/iesl/canvas/hschang/recommendation/ctrsr_datasets/citeulike-a/item-tag.dat"
    input_tags_name_file = "/iesl/canvas/hschang/recommendation/ctrsr_datasets/citeulike-a/tags.dat"
else:
    input_users_file = "/iesl/canvas/hschang/recommendation/ctrsr_datasets/citeulike-t/users.dat"
    input_tags_file = "/iesl/canvas/hschang/recommendation/ctrsr_datasets/citeulike-t/tag-item.dat"
    input_tags_name_file = "/iesl/canvas/hschang/recommendation/ctrsr_datasets/citeulike-t/tags.dat"

if file_format == 'a':
    output_target_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/citeulike-a/user_and_tags"
    output_user_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/citeulike-a/user"
    output_tag_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/citeulike-a/tags"
else:
    output_target_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/citeulike-t/user_and_tags"
    output_user_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/citeulike-t/user"
    output_tag_file = "/iesl/canvas/hschang/recommendation/Multi_facet_recommendation/data/raw/citeulike-t/tags"

user_prefix = 'U-'

def load_tags(f_in):
    #idx_d2_tag = ['NULL']
    idx_d2_tag = []
    for line in f_in:
        w = line.rstrip()
        idx_d2_tag.append(w)
    return idx_d2_tag

def read_paper_tags(f_in, idx_d2_tag):
    paper_idx_l2_tag_list = []
    for line in f_in:
        if line.rstrip() == '0':
            paper_idx_l2_tag_list.append([])
        else:
            tag_list = []
            for tag_idx_str in line.rstrip().split():
                tag = idx_d2_tag[int(tag_idx_str)]
                tag_list.append(tag)
            paper_idx_l2_tag_list.append(tag_list)
    return paper_idx_l2_tag_list
        
def read_tag_papers(f_in, idx_d2_tag, remove_first_item):
    paper_idx_d2_tag_list = {}
    max_paper_idx = -1
    for tag_idx, line in enumerate(f_in):
        tag = idx_d2_tag[tag_idx]
        for i, paper_idx_str in enumerate(line.rstrip().split()):
            if i == 0 and remove_first_item:
                continue
            paper_idx = int(paper_idx_str)
            tag_list = paper_idx_d2_tag_list.get(paper_idx, [])
            tag_list.append( tag )
            paper_idx_d2_tag_list[paper_idx] = tag_list
            if paper_idx > max_paper_idx:
                max_paper_idx = paper_idx

    paper_num = max_paper_idx + 1
    paper_idx_l2_tag_list = []
    for paper_idx in range(paper_num):
        if paper_idx in paper_idx_d2_tag_list:
            paper_idx_l2_tag_list.append(paper_idx_d2_tag_list[paper_idx])
        else:
            print('warning, paper id {} cannot be found in the tag file'.format(paper_idx) )
            paper_idx_l2_tag_list.append([])

    return paper_idx_l2_tag_list    

def read_user_papers(f_in, paper_num, remove_first_item):
    paper_idx_d2_user_list = {}
    for user_idx, line in enumerate(f_in):
        for i, paper_idx_str in enumerate(line.rstrip().split()):
            if i == 0 and remove_first_item:
                continue
            paper_idx = int(paper_idx_str)
            user_list = paper_idx_d2_user_list.get(paper_idx, [])
            user_list.append(user_prefix + str(user_idx))
            paper_idx_d2_user_list[paper_idx] = user_list

    paper_idx_l2_user_list = []
    for paper_idx in range(paper_num):
        if paper_idx in paper_idx_d2_user_list:
            paper_idx_l2_user_list.append(paper_idx_d2_user_list[paper_idx])
        else:
            print('warning, paper id {} cannot be found in the user file'.format(paper_idx) )
            paper_idx_l2_user_list.append([])

    return paper_idx_l2_user_list    

with open(input_tags_name_file) as f_in:
    idx_d2_tag = load_tags(f_in)

if file_format == 'a':
    with open(input_tags_file) as f_in:
        paper_idx_l2_tag_list = read_paper_tags(f_in, idx_d2_tag)
else:
    with open(input_tags_file) as f_in:
        paper_idx_l2_tag_list = read_tag_papers(f_in, idx_d2_tag, remove_first_item = True)

with open(input_users_file) as f_in:
    paper_num = len(paper_idx_l2_tag_list)
    if file_format == 'a':
        remove_first_item = False
    else:
        remove_first_item = True
    paper_idx_l2_user_list = read_user_papers(f_in, paper_num, remove_first_item)

#paper_idx_l2_user_tag_list = list(zip(paper_idx_l2_user_list, paper_idx_l2_tag_list))
#print(paper_idx_l2_user_tag_list)
with open(output_tag_file, 'w') as f_out:
    for tag_list in paper_idx_l2_tag_list:
        f_out.write(' '.join(tag_list) + '\n' )

with open(output_user_file, 'w') as f_out:
    for user_list in paper_idx_l2_user_list:
        f_out.write(' '.join(user_list) + '\n' )

with open(output_target_file, 'w') as f_out:
    for user_list, tag_list in zip(paper_idx_l2_user_list, paper_idx_l2_tag_list):
        f_out.write(' '.join(user_list) + '\t' + ' '.join(tag_list) + '\n' )
    #for paper_idx in range(paper_num):
    #    user_list = paper_idx_l2_user_tag_list[paper_idx]
    #    f_out.write(' '.join(user_tag_list) + '\n' )
