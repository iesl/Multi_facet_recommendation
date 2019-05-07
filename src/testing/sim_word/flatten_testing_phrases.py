dataset_dir = "./dataset_testing/"

dataset_list = [ [dataset_dir + "phrase/SemEval2013/train/en.trainSet.negativeInstances-v2", dataset_dir + "phrase/SemEval2013/train/en.trainSet.positiveInstances-v2", "SemEval2013" ], [dataset_dir + "phrase/Turney2012/Turney_train.txt", "Turney"] ]

output_file = dataset_dir + "phrase/SemEval2013_Turney2012_phrase_org"

def load_Turney(file_name):
    #dataset = []
    all_phrases = []
    with open(file_name) as f_in:
        for line in f_in:
            line = line.rstrip().replace(' | ', '|')
            fields = line.split('|')
            for word in fields:
                all_phrases.append(word)
            bigram = fields[0].split()
            all_phrases.append(bigram[1]+' '+bigram[0])
            #candidates = [ (fields[1],1) ] + [ (x,0) for x in fields[4:] ] #assume that 1 is correct, and 2, 3 are the candidate we want to remove
            #random.shuffle(candidates)
            #dataset.append( [bigram, candidates] )
    return all_phrases

def load_pairs(file_name):
    all_phrases = []
    with open(file_name) as f_in:
        for line in f_in:
            unigram, bigram = line.rstrip().split('\t')
            all_phrases.append(unigram)
            all_phrases.append(bigram)
    return all_phrases

def load_SemEval(neg_file, pos_file):
    neg_phrase = load_pairs(neg_file)
    pos_phrase = load_pairs(pos_file)
    all_phrases = neg_phrase + pos_phrase
    return all_phrases

all_phrases = []
for file_info in dataset_list:
    file_type = file_info[-1]
    print("loading ", file_info)
    if file_type == "SemEval2013":
        SemEval_phrase = load_SemEval( file_info[0], file_info[1] ) 
        all_phrases += SemEval_phrase
    elif file_type == "Turney":
        Turney_phrase = load_Turney( file_info[0] ) 
        all_phrases += Turney_phrase

with open(output_file, 'w') as f_out:
    for phrase in all_phrases:
        f_out.write(phrase+'\t'+phrase+'\n')
