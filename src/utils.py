import os, shutil
import torch

class Dictionary(object):
    def __init__(self, byte_mode=False):
        self.w_d2_ind = {'[null]': 0, '<unk>': 1, '<eos>': 2}
        self.ind_l2_w_freq = [ ['[null]',-1,0], ['<unk>',0,1], ['<eos>',0,2] ]
        self.num_special_token = len(self.ind_l2_w_freq)
        self.UNK_IND = 1
        self.EOS_IND = 2
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
    for i in range(num_w):
        w = idx2word_freq[i][0]
        if w in word2emb:
            val = torch.tensor(word2emb[w], device = device, requires_grad = False)
            #val = np.array(word2emb[w])
            external_emb[i,:] = val
        else:
            external_emb[i,:] = 0
    return external_emb, emb_size

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
