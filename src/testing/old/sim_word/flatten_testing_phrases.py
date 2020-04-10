dataset_dir = "./dataset_testing/"

#/mnt/nfs/scratch1/hschang/language_modeling/NSD_for_sentence_embedding/dataset_testing/HypeNet/rnd/val.tsv
#dataset_list = [ [dataset_dir + "phrase/SemEval2013/train/en.trainSet.negativeInstances-v2", dataset_dir + "phrase/SemEval2013/train/en.trainSet.positiveInstances-v2", "SemEval2013" ], [dataset_dir + "phrase/Turney2012/Turney_train.txt", "Turney"] ]
dataset_list = [ [dataset_dir + "phrase/SemEval2013/test/en.testSet.negativeInstances-v2", dataset_dir + "phrase/SemEval2013/test/en.testSet.positiveInstances-v2", "SemEval2013" ], [dataset_dir + "phrase/Turney2012/Turney_test.txt", "Turney"] ]
#dataset_list = [ [ dataset_dir + 'phrase/HypeNet/rnd/val.tsv', 'raw' , "hyper"], [ dataset_dir + 'phrase/WordNet/wordnet_valid.txt' , "POS",  "hyper" ] ]
#dataset_list = [ [ dataset_dir + 'phrase/HypeNet/rnd/test.tsv', 'raw' , "hyper"], [ dataset_dir + 'phrase/WordNet/wordnet_test.txt' , "POS",  "hyper" ] ]
#dataset_list = [ [ dataset_dir + 'phrase/HypeNet/rnd/val.tsv', 'raw' , "hyper"], [ dataset_dir + 'phrase/HypeNet/rnd/test.tsv', 'raw' , "hyper"] ]
#dataset_list = [ [ dataset_dir + 'phrase/word_hyper/BLESS.all', 'POS' , "hyper"], [ dataset_dir + 'phrase/word_hyper/EVALution.all' , "POS",  "hyper" ], [ dataset_dir + 'phrase/word_hyper/LenciBenotto.all' , "POS",  "hyper" ], [ dataset_dir + 'phrase/word_hyper/Weeds.all' , "POS",  "hyper" ] ]
#output_file = dataset_dir + "phrase/SemEval2013_Turney2012_phrase_org"
output_file = dataset_dir + "phrase/SemEval2013_Turney2012_phrase_test_org"
#output_file = dataset_dir + "phrase/HypeNet_WordNet_val_org"
#output_file = dataset_dir + "phrase/HypeNet_WordNet_test_org"
#output_file = dataset_dir + "phrase/HypeNet_val_test_org"
#output_file = dataset_dir + "phrase/word_hyper_org"

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

def processing_phrase(phrase_raw, POS_suffix):
    words = phrase_raw.split(',')
    output_list = []
    for i, w in enumerate(words):
        if POS_suffix and i == len(words) - 1:
            if w[-1] != 'n':
                return []
            else:    
                output_list.append(w[:-2])
        else:
            output_list.append(w)
    return ' '.join(output_list)

def load_hyper(file_name, POS_suffix):
    all_phrases = []
    not_noun_count = 0
    with open(file_name) as f_in:
        for line in f_in:
            fields = line.rstrip().split('\t')
            hypo_candidate = processing_phrase(fields[0], POS_suffix)
            hyper_candidate = processing_phrase(fields[1], POS_suffix)
            if len(hypo_candidate) == 0 or len(hyper_candidate) == 0:
                #one of the phrase contains words which are not nouns
                not_noun_count += 1
                continue
            all_phrases.append(hypo_candidate)
            all_phrases.append(hyper_candidate)
    print("Throw away "+str(not_noun_count)+" pairs which are not nouns and keep "+str(len(all_phrases)/2)+" pairs")
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
    elif file_type == "hyper":
        POS_suffix = False
        if file_info[1] == "POS":
            POS_suffix = True
        all_phrases += load_hyper(file_info[0], POS_suffix)

with open(output_file, 'w') as f_out:
    for phrase in all_phrases:
        f_out.write(phrase+'\t'+phrase+'\n')
