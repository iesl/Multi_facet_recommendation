import os, shutil
import torch
import torch.utils.data
import model as model_code
import torch.nn as nn
import numpy as np
import random

UNK_IND = 1
EOS_IND = 2

class Dictionary(object):
    def __init__(self, byte_mode=False):
        self.w_d2_ind = {'[null]': 0, '<unk>': 1, '<eos>': 2}
        self.ind_l2_w_freq = [ ['[null]',-1,0], ['<unk>',0,1], ['<eos>',0,2] ]
        self.num_special_token = len(self.ind_l2_w_freq)
        self.UNK_IND = UNK_IND
        self.EOS_IND = EOS_IND
        self.byte_mode = byte_mode

    def dict_check_add(self,w):
        if w not in self.w_d2_ind:
            w_ind = len(self.w_d2_ind)
            self.w_d2_ind[w] = w_ind
            if self.byte_mode:
                self.ind_l2_w_freq.append([w.decode('utf-8'), 1, w_ind])
            else:
                self.ind_l2_w_freq.append([w, 1, w_ind])
        else:
            w_ind = self.w_d2_ind[w]
            self.ind_l2_w_freq[w_ind][1] += 1
        return w_ind

    def append_eos(self,w_ind_list):
        w_ind_list.append(self.EOS_IND) # append <eos>
        self.ind_l2_w_freq[self.EOS_IND][1] += 1

    def densify_index(self,min_sent_length):
        vocab_size = len(self.ind_l2_w_freq)
        compact_mapping = [0]*vocab_size
        for i in range(self.num_special_token):
            compact_mapping[i] = i
        #compact_mapping[1] = 1
        #compact_mapping[2] = 2

        #total_num_filtering = 0
        total_freq_filtering = 0
        current_new_idx = self.num_special_token

        #for i, (w, w_freq, w_ind_org) in enumerate(self.ind_l2_w_freq[self.num_special_token:]):
        for i in range(self.num_special_token,vocab_size):
            w, w_freq, w_ind_org = self.ind_l2_w_freq[i]
            if w_freq < min_sent_length:
                compact_mapping[i] = self.UNK_IND
                self.ind_l2_w_freq[i][-1] = self.UNK_IND
                self.ind_l2_w_freq[i].append('unk')
                #total_num_filtering += 1
                total_freq_filtering += w_freq
            else:
                compact_mapping[i] = current_new_idx
                self.ind_l2_w_freq[i][-1] = current_new_idx
                current_new_idx += 1

        self.ind_l2_w_freq[self.UNK_IND][1] = total_freq_filtering #update <unk> frequency


        print("{}/{} word types are filtered".format(vocab_size - current_new_idx, vocab_size) )

        return compact_mapping, total_freq_filtering

    def store_dict(self,f_out):
        vocab_size = len(self.ind_l2_w_freq)
        for i in range(vocab_size):
            #print(ind_l2_w_freq[i])
            self.ind_l2_w_freq[i][1] = str(self.ind_l2_w_freq[i][1])
            self.ind_l2_w_freq[i][2] = str(self.ind_l2_w_freq[i][2])
            f_out.write('\t'.join(self.ind_l2_w_freq[i])+'\n')

def load_word_dict(f_in):
    d = {}
    max_ind = 0 #if the dictionary is densified, max_ind is the same as len(d)
    for i, line in enumerate(f_in):
        fields = line.rstrip().split('\t')
        if len(fields) == 3:
            d[fields[0]] = [int(fields[2]),int(fields[1])]
            max_ind = int(fields[2])

    return d, max_ind

def load_idx2word_freq(f_in):
    idx2word_freq = []
    for i, line in enumerate(f_in):
        fields = line.rstrip().split('\t')
        if len(fields) == 3:
            assert len(idx2word_freq) == int(fields[2])
            idx2word_freq.append([fields[0],int(fields[1])])            

    return idx2word_freq

class F2SetDataset(torch.utils.data.Dataset):
#will need to handle the partial data loading if the dataset size is larger than cpu memory
#We could also use this class to put all sentences with the same length together
    def __init__(self, feature, target, device):
        self.feature = feature
        self.target = target
        self.output_device = device

    def __len__(self):
        return self.feature.size(0)

    def __getitem__(self, idx):
        feature = torch.tensor(self.feature[idx, :], dtype = torch.long, device = self.output_device)
        if self.target is None:
            target = []
        else:
            target = torch.tensor(self.target[idx, :], dtype = torch.long, device = self.output_device)
        #debug target[-1] = idx
        return [feature, target]
        #return [self.feature[idx, :], self.target[idx, :]]

def create_data_loader(f_in, bsz, device):
    feature, target = torch.load(f_in)
    #print(feature)
    #print(target)
    dataset = F2SetDataset(feature, target, device)
    use_cuda = False
    if device == 'cude':
        use_cuda = True
    return torch.utils.data.DataLoader(dataset, batch_size = bsz, shuffle = True, pin_memory=use_cuda, drop_last=False)

def convert_sent_to_tensor(proc_sent_list, max_sent_len, word2idx):
    store_type = torch.int32
    
    num_sent = len(proc_sent_list)
    feature_tensor = torch.zeros( num_sent, max_sent_len, dtype = store_type)
    truncate_num = 0
    for output_i, proc_sent in enumerate(proc_sent_list):
        w_ind_list = []
        w_list = proc_sent.split()
        sent_len = min(len(w_list)+1, max_sent_len)
        if len(w_list) > max_sent_len-1:
            truncate_num += 1
        for w in w_list[:sent_len-1]:
            if w in word2idx:
                w_ind_list.append(word2idx[w][0])
            else:
                w_ind_list.append(UNK_IND)
        w_ind_list.append(0) #buggy preprocessing
        #w_ind_list.append(EOS_IND)
        #print(w_ind_list)
        feature_tensor[output_i,-sent_len:] = torch.tensor(w_ind_list, dtype = store_type)
    print("Truncation rate: ", truncate_num/float(len(proc_sent_list)) )
    return feature_tensor

def load_testing_sent(dict_path, input_path, max_sent_len, eval_bsz, device):
    with open(dict_path) as f_in:
        idx2word_freq = load_idx2word_freq(f_in)

    with open(dict_path) as f_in:
        word2idx, max_idx = load_word_dict(f_in)

    org_sent_list = []
    proc_sent_list = []
    with open(input_path) as f_in:
        for line in f_in:
            org_sent, proc_sent = line.rstrip().split('\t')
            org_sent_list.append(org_sent)
            proc_sent_list.append(proc_sent)
    
    feature_tensor = convert_sent_to_tensor(proc_sent_list, max_sent_len, word2idx)
    dataset = F2SetDataset(feature_tensor, None, device)
    use_cuda = False
    if device == 'cude':
        use_cuda = True
    dataloader_test = torch.utils.data.DataLoader(dataset, batch_size = eval_bsz, shuffle = False, pin_memory=use_cuda, drop_last=False)

    return dataloader_test, org_sent_list, idx2word_freq

def load_corpus(data_path, train_bsz, eval_bsz, device, tensor_folder = "tensors", training_file = "train.pt"):
    train_corpus_name = data_path + "/"+tensor_folder+"/" + training_file
    val_org_corpus_name = data_path +"/"+tensor_folder+"/val_org.pt"
    val_shuffled_corpus_name = data_path + "/"+tensor_folder+"/val_shuffled.pt"
    dictionary_input_name = data_path + "dictionary_index"

    with open(dictionary_input_name) as f_in:
        idx2word_freq = load_idx2word_freq(f_in)

    with open(train_corpus_name,'rb') as f_in:
        dataloader_train = create_data_loader(f_in, train_bsz, device)

    with open(val_org_corpus_name,'rb') as f_in:
        dataloader_val = create_data_loader(f_in, eval_bsz, device)

    with open(val_shuffled_corpus_name,'rb') as f_in:
        dataloader_val_shuffled = create_data_loader(f_in, eval_bsz, device)

    return idx2word_freq, dataloader_train, dataloader_val, dataloader_val_shuffled

def load_emb_file(emb_file, device, idx2word_freq):
    with open(emb_file) as f_in:
        word2emb = {}
        for line in f_in:
            word_val = line.rstrip().split(' ')
            if len(word_val) < 3:
                continue
            word = word_val[0]
            val = [float(x) for x in  word_val[1:]]
            word2emb[word] = val
            emb_size = len(val)
    num_w = len(idx2word_freq)
    #emb_size = len(word2emb.values()[0])
    #external_emb = torch.empty(num_w, emb_size, device = device, requires_grad = update_target_emb)
    external_emb = torch.empty(num_w, emb_size, device = device, requires_grad = False)
    OOV_num = 0
    OOV_freq = 0
    total_freq = 0
    for i in range(num_w):
        w = idx2word_freq[i][0]
        total_freq += idx2word_freq[i][1]
        if w in word2emb:
            val = torch.tensor(word2emb[w], device = device, requires_grad = False)
            #val = np.array(word2emb[w])
            external_emb[i,:] = val
        else:
            external_emb[i,:] = 0
            OOV_num += 1
            OOV_freq += idx2word_freq[i][1]

    print("OOV word type percentage: {}%".format( OOV_num/float(num_w)*100 ))
    print("OOV token percentage: {}%".format( OOV_freq/float(total_freq)*100 ))
    return external_emb, emb_size

def output_parallel_models(use_cuda, single_gpu, encoder, decoder):
    if use_cuda:
        if single_gpu:
            parallel_encoder = encoder.cuda()
            parallel_decoder = decoder.cuda()
        else:
            parallel_encoder = nn.DataParallel(encoder, dim=0).cuda()
            parallel_decoder = nn.DataParallel(decoder, dim=0).cuda()
            #parallel_decoder = decoder.cuda()
    else:
        parallel_encoder = encoder
        parallel_decoder = decoder
    return parallel_encoder, parallel_decoder

def loading_all_models(args, idx2word_freq, device, linear_mapping_dim):

    if len(args.emb_file) > 0:
        if args.emb_file[-3:] == '.pt':
            word_emb = torch.load( args.emb_file )
            output_emb_size = word_emb.size(1)
        else:
            word_emb, output_emb_size = load_emb_file(args.emb_file,device,idx2word_freq)
    else:
        output_emb_size = args.emsize

    ntokens = len(idx2word_freq)
    if args.en_model == "LSTM":
        external_emb = torch.tensor([0.])
        encoder = model_code.RNNModel_simple(args.en_model, ntokens, args.emsize, args.nhid, args.nlayers,
                       args.dropout, args.dropouti, args.dropoute, external_emb)
        decoder = model_code.RNNModel_decoder(args.de_model, args.nhid * 2, args.nhidlast2, output_emb_size, 1, args.n_basis, linear_mapping_dim = linear_mapping_dim, dropoutp= 0.5)
        #if use_position_emb:
        #    decoder = model_code.RNNModel_decoder(args.de_model, args.nhid * 2, args.nhidlast2, output_emb_size, 1, args.n_basis, linear_mapping_dim = 0, dropoutp= 0.5)
        #else:
        #    decoder = model_code.RNNModel_decoder(args.de_model, args.nhid * 2, args.nhidlast2, output_emb_size, 1, args.n_basis, linear_mapping_dim = args.nhid, dropoutp= 0.5)

    encoder.load_state_dict(torch.load(os.path.join(args.checkpoint, 'encoder.pt')))
    decoder.load_state_dict(torch.load(os.path.join(args.checkpoint, 'decoder.pt')))

    if len(args.emb_file) == 0:
        word_emb = encoder.encoder.weight.detach()

    word_norm_emb = word_emb / (0.000000000001 + word_emb.norm(dim = 1, keepdim=True) )
    word_norm_emb[0,:] = 0

    parallel_encoder, parallel_decoder = output_parallel_models(args.cuda, args.single_gpu, encoder, decoder)

    return parallel_encoder, parallel_decoder, encoder, decoder, word_norm_emb

def seed_all_randomness(seed,use_cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if not use_cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(seed)

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

def save_checkpoint(encoder, decoder, optimizer_e,  optimizer_d, external_emb, path):
    torch.save(encoder.state_dict(), os.path.join(path, 'encoder.pt'))
    try:
        torch.save(decoder.state_dict(), os.path.join(path, 'decoder.pt'))
    except:
        pass
    if external_emb.size(0) > 1:
        torch.save(external_emb, os.path.join(path, 'target_emb.pt'))
    torch.save(optimizer_e.state_dict(), os.path.join(path, 'optimizer_e.pt'))
    torch.save(optimizer_d.state_dict(), os.path.join(path, 'optimizer_d.pt'))
