import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
#from weight_drop import WeightDrop

class RNNModel_decoder(nn.Module):

    def __init__(self, rnn_type, ninp, nhid, outd, nlayers, n_basis, linear_mapping_dim, dropoutp= 0.5):
        super(RNNModel_decoder, self).__init__()
        self.drop = nn.Dropout(dropoutp)
        self.n_basis = n_basis
        #self.layernorm = nn.InstanceNorm1d(n_basis, affine =False)
        #self.outd_sqrt = math.sqrt(outd)
        self.linear_mapping_dim = linear_mapping_dim
        position_emb_size = 0
        if n_basis > 0:
            if linear_mapping_dim > 0:
                self.init_linear_arr = nn.ModuleList([nn.Linear(ninp, linear_mapping_dim) for i in range(n_basis)])
                for i in range(n_basis):
                    #It seems that the LSTM only learns well when bias is larger than weights at the beginning
                    #If setting std in weight to be too large (e.g., 1), the loss might explode
                    self.init_linear_arr[i].bias.data.uniform_(-.5,.5)
                    self.init_linear_arr[i].weight.data.uniform_(-.1,.1)
            else:
                position_emb_size = 100
                self.poistion_emb = nn.Embedding( n_basis, position_emb_size )

        self.init_hid_linear_1 = nn.ModuleList([nn.Linear(ninp, nhid) for i in range(nlayers)])
        self.init_hid_linear_2 = nn.ModuleList([nn.Linear(ninp, nhid) for i in range(nlayers)])
        for i in range(nlayers):
            self.init_hid_linear_1[i].weight.data.uniform_(-.1,.1)
            self.init_hid_linear_1[i].bias.data.uniform_(-.5,.5)
            self.init_hid_linear_2[i].weight.data.uniform_(-.1,.1)
            self.init_hid_linear_2[i].bias.data.uniform_(-.5,.5)


        #self.out_linear = nn.Linear(nhid, outd, bias=False)
        self.out_linear = nn.Linear(nhid, outd)
        self.relu_layer = nn.ReLU()

        if rnn_type in ['LSTM', 'GRU']:
            if linear_mapping_dim > 0:
                self.rnn = getattr(nn, rnn_type)(linear_mapping_dim, nhid, nlayers, dropout=0)
            else:
                self.rnn = getattr(nn, rnn_type)(ninp+position_emb_size, nhid, nlayers, dropout=0)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp+position_emb_size, nhid, nlayers, nonlinearity=nonlinearity, dropout=0)
        
        self.coeff_nlayers = 1
        self.coeff_rnn = nn.LSTM(ninp+outd , nhid, num_layers = self.coeff_nlayers , bidirectional = True)
        self.coeff_out_linear = nn.Linear(nhid*2, 2)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        #necessary?
        initrange = 0.1
        self.out_linear.bias.data.zero_()
        self.out_linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_init, predict_coeff_sum = False):
        
        hidden_1 = torch.cat( [self.init_hid_linear_1[i](input_init).unsqueeze(dim = 0) for i in range(self.nlayers)], dim = 0 )
        hidden_2 = torch.cat( [self.init_hid_linear_2[i](input_init).unsqueeze(dim = 0) for i in range(self.nlayers)], dim = 0 )
        hidden = (hidden_1, hidden_2)

        input = input_init.expand(self.n_basis, input_init.size(0), input_init.size(1) )
        if self.n_basis == 0:
            emb = input
        else:
            if self.linear_mapping_dim > 0:
                emb_raw = torch.cat( [self.init_linear_arr[i](input_init).unsqueeze(dim = 0)  for i in range(self.n_basis) ] , dim = 0 )
                #emb = emb_raw
                emb = self.drop(emb_raw)
            else:
                batch_size = input.size(1)
                input_pos = torch.arange(self.n_basis,dtype=torch.long,device = input.get_device()).expand(batch_size,self.n_basis).permute(1,0)

                poistion_emb_input = self.poistion_emb(input_pos)
                poistion_emb_input = self.drop(poistion_emb_input)
                emb = torch.cat( ( poistion_emb_input,input), dim = 2  )

        output, hidden = self.rnn(emb, hidden)
        #output = self.drop(output)
        #output = self.out_linear(self.relu_layer(output))
        #output = self.layernorm(output.permute(1,0,2)).permute(1,0,2)
        #output /= self.outd_sqrt
        output = self.out_linear(output)
        #output = output / (0.000000000001 + output.norm(dim = 2, keepdim=True) )

        if not predict_coeff_sum:
            #output has dimension (n_seq_len, n_batch, n_emb_size)
            return output, hidden
        else:
            #bsz = input.size(1)
            #weight = next(self.parameters()) # we use this just to let the tensor have the same type and device as the first weight in this class
            #hidden_init = (weight.new_zeros(self.coeff_nlayers*2, bsz, self.nhid), weight.new_zeros(self.coeff_nlayers*2, bsz, self.nhid))
            coeff_input= torch.cat( (input, output), dim = 2)
            #coeff_output, coeff_hidden = self.coeff_rnn(coeff_input, hidden_init)
            coeff_output, coeff_hidden = self.coeff_rnn(coeff_input.detach()) #default hidden state is 0
            coeff_pred = self.coeff_out_linear(coeff_output)
            return output, hidden, coeff_pred

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

class RNNModel_simple(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout, dropouti, dropoute, external_emb):
        super(RNNModel_simple, self).__init__()
        #self.drop = nn.Dropout(dropout)
        self.lockdrop = LockedDropout()
        if len(external_emb) > 1:
            self.encoder = nn.Embedding.from_pretrained(external_emb, freeze = False)
            ntoken, ninp = external_emb.size()
        else:
            self.encoder = nn.Embedding(ntoken, ninp)
        self.use_dropout = True
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=0, bidirectional = True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=0)

        #self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropoute = dropoute
        self.dropouti = dropouti

    #def init_weights(self):
    #    initrange = 0.1
    #    self.encoder.weight.data.uniform_(-initrange, initrange)
    #    self.encoder.weight.data[0,:] = 0
    #    self.decoder.bias.data.zero_()

    def forward(self, input):
        
        bsz = input.size(1)
        hidden = self.init_hidden(bsz)

        #weight = next(self.parameters())
        #input_long = weight.new_tensor(input,dtype = torch.long)

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.use_dropout else 0)
        emb = self.lockdrop(emb, self.dropouti if self.use_dropout else 0)       
        #emb = self.drop(self.encoder(input))
        output_org, hidden = self.rnn(emb, hidden)
        #output = self.drop(output_org)
        output = self.lockdrop(output_org, self.dropout if self.use_dropout else 0)
        output_unpacked = output.view(output.size(0), bsz, 2, self.nhid)
        output_last = torch.cat( (output_unpacked[-1,:,0,:], output_unpacked[0,:,1,:]) , dim = 1)
        #output_last = output_last / (0.000000000001 + output_last.norm(dim = 1, keepdim=True) )
        #forward_mean = torch.mean(output_unpacked[:,:,0,:], dim = 0)
        #backward_mean = torch.mean(output_unpacked[:,:,1,:], dim = 0)
        #output_last = torch.cat( (forward_mean, backward_mean), dim = 1 )
        return output, hidden, output_last

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers*2, bsz, self.nhid),
                    weight.new_zeros(self.nlayers*2, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nhidlast, nlayers, 
                 dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, 
                 tie_weights=False, ldropout=0.5, n_experts=10):
        super(RNNModel, self).__init__()
        self.use_dropout = True
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)
        
        self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else nhidlast, 1, dropout=0) for l in range(nlayers)]
        if wdrop:
            self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop if self.use_dropout else 0) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.prior = nn.Linear(nhidlast, n_experts, bias=False)
        self.latent = nn.Sequential(nn.Linear(nhidlast, n_experts*ninp), nn.Tanh())
        self.decoder = nn.Linear(ninp, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights(tie_weights)

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.ldropout = ldropout
        self.dropoutl = ldropout
        self.n_experts = n_experts
        self.ntoken = ntoken

        size = 0
        for p in self.parameters():
            size += p.nelement()
        print('param size: {}'.format(size))

    def init_weights(self, tie_weights):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.encoder.weight.data[0,:] = 0
        self.decoder.bias.data.fill_(0)
        if tie_weights:
            self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False, return_prob=False):
        batch_size = input.size(1)

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if (self.training and self.use_dropout) else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti if self.use_dropout else 0)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            #print raw_output.size()
            #print hidden[l][0].size()
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth if self.use_dropout else 0)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout if self.use_dropout else 0)
        outputs.append(output)

        latent = self.latent(output)
        latent = self.lockdrop(latent, self.dropoutl if self.use_dropout else 0)
        logit = self.decoder(latent.view(-1, self.ninp))

        prior_logit = self.prior(output).contiguous().view(-1, self.n_experts)
        prior = nn.functional.softmax(prior_logit, -1)

        prob = nn.functional.softmax(logit.view(-1, self.ntoken), -1).view(-1, self.n_experts, self.ntoken)
        prob = (prob * prior.unsqueeze(2).expand_as(prob)).sum(1)

        if return_prob:
            model_output = prob
        else:
            log_prob = torch.log(prob.add_(1e-8))
            model_output = log_prob

        model_output = model_output.view(-1, batch_size, self.ntoken)

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()),
                 Variable(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else self.nhidlast).zero_()))
                for l in range(self.nlayers)]

if __name__ == '__main__':
    model = RNNModel('LSTM', 10, 12, 12, 12, 2)
    input = Variable(torch.LongTensor(13, 9).random_(0, 10))
    hidden = model.init_hidden(9)
    model(input, hidden)

    # input = Variable(torch.LongTensor(13, 9).random_(0, 10))
    # hidden = model.init_hidden(9)
    # print(model.sample(input, hidden, 5, 6, 1, 2, sample_latent=True).size())
