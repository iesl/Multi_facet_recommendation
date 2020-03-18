import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
import model_trans
import sys
#from weight_drop import WeightDrop

class MatrixReconstruction(nn.Module):
    def __init__(self, batch_size, ntopic, nbow, device):
        super(MatrixReconstruction, self).__init__()
        self.coeff = nn.Parameter(torch.randn(batch_size, ntopic, nbow, device=device, requires_grad=True))
        self.device = device
    
    def compute_coeff_pos(self):
        self.coeff.data = self.coeff.clamp(0.0, 1.0)
    
    def compute_coeff_pos_norm(self):
        self.coeff.data = self.coeff.clamp(min = 0.0)
        norm_value = torch.max( torch.ones(1, device=self.device) , self.coeff.data.sum(dim = 2, keepdim=True) )
        self.coeff.data = self.coeff.data / norm_value
        #Constraints the sum of all coefficient smaller than 1 -> directly normalize like clamping the maximal value 
        #If denominator < 1 -> = 1
        #-> When n_basis = 1, it is exactly the same as single embedding approach
    
    def forward(self, input):
        result = self.coeff.matmul(input)
        return result

class RNN_decoder(nn.Module):
    def __init__(self, model_type, emb_dim, ninp, nhid, nlayers, dropout_prob):
        super(RNN_decoder, self).__init__()
        if model_type in ['LSTM', 'GRU']:
            print("RNN decoder dropout:", dropout_prob)
            self.rnn = getattr(nn, model_type)(emb_dim, nhid, nlayers, dropout=dropout_prob)
            #if linear_mapping_dim > 0:
            #    self.rnn = getattr(nn, model_type)(linear_mapping_dim, nhid, nlayers, dropout=0)
            #else:
            #    self.rnn = getattr(nn, model_type)(ninp+position_emb_size, nhid, nlayers, dropout=0)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[model_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            #self.rnn = nn.RNN(ninp+position_emb_size, nhid, nlayers, nonlinearity=nonlinearity, dropout=0)
            self.rnn = nn.RNN(emb_dim, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout_prob)
        
        if model_type == 'LSTM':
            self.init_hid_linear_1 = nn.ModuleList([nn.Linear(ninp, nhid) for i in range(nlayers)])
            self.init_hid_linear_2 = nn.ModuleList([nn.Linear(ninp, nhid) for i in range(nlayers)])
            for i in range(nlayers):
                self.init_hid_linear_1[i].weight.data.uniform_(-.1,.1)
                self.init_hid_linear_1[i].bias.data.uniform_(-.5,.5)
                self.init_hid_linear_2[i].weight.data.uniform_(-.1,.1)
                self.init_hid_linear_2[i].bias.data.uniform_(-.5,.5)
        self.nlayers = nlayers
        self.model_type = model_type

    def forward(self, input_init, emb):
        hidden_1 = torch.cat( [self.init_hid_linear_1[i](input_init).unsqueeze(dim = 0) for i in range(self.nlayers)], dim = 0 )
        hidden_2 = torch.cat( [self.init_hid_linear_2[i](input_init).unsqueeze(dim = 0) for i in range(self.nlayers)], dim = 0 )
        hidden = (hidden_1, hidden_2)
        
        output, hidden = self.rnn(emb, hidden)
        return output

    #def init_hidden(self, bsz):
    #    weight = next(self.parameters())
    #    if self.model_type == 'LSTM':
    #        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
    #                weight.new_zeros(self.nlayers, bsz, self.nhid))
    #    else:
    #        return weight.new_zeros(self.nlayers, bsz, self.nhid)
    

class ext_emb_to_seq(nn.Module):
    def __init__(self, model_type_list, emb_dim, ninp, nhid, nlayers, n_basis, trans_layers, using_memory, add_position_emb, dropout_prob_trans, dropout_prob_lstm):
        super(ext_emb_to_seq, self).__init__()
        self.decoder_array = nn.ModuleList()
        input_dim = emb_dim
        self.trans_dim = None
        for model_type in model_type_list:
            if model_type == 'LSTM':
                model = RNN_decoder(model_type, input_dim, ninp, nhid, nlayers, dropout_prob_lstm)
                input_dim = nhid
                #output_dim = nhid
            elif model_type == 'TRANS':
                #model = model_trans.BertEncoder(model_type = model_type, hidden_size = input_dim, max_position_embeddings = n_basis, num_hidden_layers=trans_layers)
                if input_dim == 768:
                    num_attention_heads = 12
                else:
                    num_attention_heads = 10
                model = model_trans.Transformer(model_type = model_type, hidden_size = input_dim, max_position_embeddings = n_basis, num_hidden_layers=trans_layers, add_position_emb = add_position_emb,  decoder = using_memory, dropout_prob = dropout_prob_trans, num_attention_heads = num_attention_heads)
                self.trans_dim = input_dim
                #output_dim = input_dim
            else:
                print("model type must be either LSTM or TRANS")
                sys.exit(1)
            self.decoder_array.append( model )
        self.output_dim = input_dim

    def forward(self, input_init, emb, memory=None):
        hidden_states = emb
        for model in self.decoder_array:
            model_type = model.model_type
            if model_type == 'LSTM':
                hidden_states = model(input_init, hidden_states)
            elif model_type == 'TRANS':
                #If we want to use transformer by default at the end, we will want to reconsider reducing the number of permutes
                hidden_states = hidden_states.permute(1,0,2)
                hidden_states = model(hidden_states, memory_tensors = memory)
                hidden_states = hidden_states[0].permute(1,0,2)
        return hidden_states

#class RNNModel_decoder(nn.Module):
class EMB2SEQ(nn.Module):

    #def __init__(self, model_type_list, ninp, nhid, outd, nlayers, n_basis, linear_mapping_dim, dropoutp= 0.5, trans_layers=2, using_memory = False):
    def __init__(self, model_type_list, coeff_model, ninp, nhid, target_emb_sz, nlayers, n_basis, positional_option, dropoutp= 0.5, trans_layers=2, using_memory = False, dropout_prob_trans = 0.1,dropout_prob_lstm=0):
        #super(RNNModel_decoder, self).__init__()
        super(EMB2SEQ, self).__init__()
        self.drop = nn.Dropout(dropoutp)
        self.n_basis = n_basis
        #self.layernorm = nn.InstanceNorm1d(n_basis, affine =False)
        #self.outd_sqrt = math.sqrt(outd)
        #self.linear_mapping_dim = linear_mapping_dim
        #position_emb_size = 0
        #if n_basis > 0:
        #if linear_mapping_dim > 0:
        input_size = ninp
        add_position_emb = False
        if positional_option == 'linear':
            linear_mapping_dim = ninp
            self.init_linear_arr = nn.ModuleList([nn.Linear(ninp, linear_mapping_dim) for i in range(n_basis)])
            for i in range(n_basis):
                #It seems that the LSTM only learns well when bias is larger than weights at the beginning
                #If setting std in weight to be too large (e.g., 1), the loss might explode
                self.init_linear_arr[i].bias.data.uniform_(-.5,.5)
                self.init_linear_arr[i].weight.data.uniform_(-.1,.1)
            input_size = linear_mapping_dim
        elif positional_option == 'cat':
            position_emb_size = 100
            self.poistion_emb = nn.Embedding( n_basis, position_emb_size )
            self.linear_keep_same_dim = nn.Linear(position_emb_size + ninp, ninp)
            #input_size = position_emb_size + ninp
            input_size = ninp
        elif positional_option == 'add':
            input_size = ninp
            add_position_emb = True
            if model_type_list[0] == 'LSTM':
                self.poistion_emb = nn.Embedding( n_basis, ninp )
            else:
                self.scale_factor = math.sqrt(ninp)
                
        #self.relu_layer = nn.ReLU()
        
        self.positional_option = positional_option
        self.dep_learner = ext_emb_to_seq(model_type_list, input_size, ninp, nhid, nlayers, n_basis, trans_layers, using_memory, add_position_emb, dropout_prob_trans, dropout_prob_lstm)
        
        self.trans_dim = self.dep_learner.trans_dim

        #self.out_linear = nn.Linear(nhid, outd, bias=False)
        self.out_linear = nn.Linear(self.dep_learner.output_dim, target_emb_sz)
        #self.final = nn.Linear(target_emb_sz, target_emb_sz)
        self.final_linear_arr = nn.ModuleList([nn.Linear(target_emb_sz, target_emb_sz) for i in range(n_basis)])
        
        self.coeff_model = coeff_model
        if coeff_model == "LSTM":
            coeff_nlayers = 1
            #elf.coeff_rnn = nn.LSTM(ninp+outd , nhid, num_layers = coeff_nlayers , bidirectional = True)
            self.coeff_rnn = nn.LSTM(input_size+target_emb_sz , nhid, num_layers = coeff_nlayers , bidirectional = True)
            output_dim = nhid*2
        elif coeff_model == "TRANS":
            coeff_nlayers = 2
            self.coeff_trans = model_trans.Transformer(model_type = 'TRANS', hidden_size = input_size+target_emb_sz, max_position_embeddings = n_basis, num_hidden_layers=coeff_nlayers, add_position_emb = False,  decoder = False)
            #self.coeff_trans = model_trans.Transformer(model_type = 'TRANS', hidden_size = ninp+outd, max_position_embeddings = n_basis, num_hidden_layers=coeff_nlayers, add_position_emb = False,  decoder = False)
            output_dim = input_size+target_emb_sz
        elif coeff_model == "None":
            pass
            #output_dim = ninp+outd
            
        #self.coeff_out_linear = nn.Linear(nhid*2, 2)
        if coeff_model != "None":
            half_output_dim = int(output_dim / 2)
            self.coeff_out_linear_1 = nn.Linear(output_dim, half_output_dim)
            self.coeff_out_linear_2 = nn.Linear(half_output_dim, half_output_dim)
            self.coeff_out_linear_3 = nn.Linear(half_output_dim, 2)
        #self.coeff_out_linear_1 = nn.Linear(nhid*2, nhid)
        #self.coeff_out_linear_2 = nn.Linear(nhid, nhid)
        #self.coeff_out_linear_3 = nn.Linear(nhid, 2)

        self.init_weights()

        #self.model_type = model_type
        #self.nhid = nhid

    def init_weights(self):
        #necessary?
        initrange = 0.1
        self.out_linear.bias.data.zero_()
        self.out_linear.weight.data.uniform_(-initrange, initrange)
        #self.final.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_init, memory = None, predict_coeff_sum = False):
        
        #print(input_init.size())
        
        def prepare_posi_emb(input, poistion_emb, drop):
            batch_size = input.size(1)
            n_basis = input.size(0)
            input_pos = torch.arange(n_basis,dtype=torch.long,device = input.get_device()).expand(batch_size,n_basis).permute(1,0)
            poistion_emb_input = poistion_emb(input_pos)
            poistion_emb_input = drop(poistion_emb_input)
            return poistion_emb_input

        input = input_init.expand(self.n_basis, input_init.size(0), input_init.size(1) )
        #if self.n_basis == 0:
        #    emb = input
        #else:
        #    if self.linear_mapping_dim > 0:
        if self.positional_option == 'linear':
            emb_raw = torch.cat( [self.init_linear_arr[i](input_init).unsqueeze(dim = 0)  for i in range(self.n_basis) ] , dim = 0 )
            #emb = emb_raw
            emb = self.drop(emb_raw)
        elif self.positional_option == 'cat':
            #batch_size = input.size(1)
            #input_pos = torch.arange(self.n_basis,dtype=torch.long,device = input.get_device()).expand(batch_size,self.n_basis).permute(1,0)

            #poistion_emb_input = self.poistion_emb(input_pos)
            #poistion_emb_input = self.drop(poistion_emb_input)
            poistion_emb_input = prepare_posi_emb(input, self.poistion_emb, self.drop)
            emb = torch.cat( ( poistion_emb_input,input), dim = 2  )
            emb = self.linear_keep_same_dim(emb)
        elif self.positional_option == 'add':
            if self.dep_learner.decoder_array[0].model_type == "LSTM":
                poistion_emb_input = prepare_posi_emb(input, self.poistion_emb, self.drop)
                emb = input + poistion_emb_input
            else:
                emb = input * self.scale_factor

        output = self.dep_learner(input_init, emb, memory)
        #output = self.drop(output)
        #output = self.out_linear(self.relu_layer(output))
        #output = self.layernorm(output.permute(1,0,2)).permute(1,0,2)
        #output /= self.outd_sqrt
        output = self.out_linear(output)
        #output = self.final(output)
        output = torch.cat( [self.final_linear_arr[i](output[i,:,:]).unsqueeze(dim = 0)  for i in range(self.n_basis) ] , dim = 0 )
        #output = output / (0.000000000001 + output.norm(dim = 2, keepdim=True) )
        output_batch_first = output.permute(1,0,2)

        if not predict_coeff_sum:
            #output has dimension (n_batch, n_seq_len, n_emb_size)
            return output_batch_first
        else:
            #bsz = input.size(1)
            #weight = next(self.parameters()) # we use this just to let the tensor have the same type and device as the first weight in this class
            #hidden_init = (weight.new_zeros(self.coeff_nlayers*2, bsz, self.nhid), weight.new_zeros(self.coeff_nlayers*2, bsz, self.nhid))
            #coeff_input= torch.cat( (input, output), dim = 2)
            coeff_input= torch.cat( (emb, output), dim = 2)
            if self.coeff_model == "LSTM":
                #coeff_output, coeff_hidden = self.coeff_rnn(coeff_input, hidden_init)
                coeff_output, coeff_hidden = self.coeff_rnn(coeff_input.detach()) #default hidden state is 0
            elif self.coeff_model == "TRANS":
                hidden_states = coeff_input.detach().permute(1,0,2)
                hidden_states = self.coeff_trans(hidden_states)
                coeff_output = hidden_states[0].permute(1,0,2)
            #coeff_pred = self.coeff_out_linear(coeff_output)
            coeff_pred_1 = F.relu(self.coeff_out_linear_1(coeff_output))
            coeff_pred_2 = F.relu(self.coeff_out_linear_2(coeff_pred_1))
            coeff_pred = self.coeff_out_linear_3(coeff_pred_2)
            coeff_pred_batch_first = coeff_pred.permute(1,0,2)
            return output_batch_first, coeff_pred_batch_first


class RNN_encoder(nn.Module):
    def __init__(self, model_type, ninp, nhid, nlayers, dropout):
        super(RNN_encoder, self).__init__()
        if model_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, model_type)(ninp, nhid, nlayers, dropout=0, bidirectional = True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[model_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=0)
        
        self.use_dropout = True
        self.dropout = dropout
        self.lockdrop = LockedDropout()
        self.nlayers = nlayers
        self.model_type = model_type
    
    def forward(self, emb):
        #output_org, hidden = self.rnn(emb, hidden)
        output_org, hidden = self.rnn(emb)
        #output = self.drop(output_org)
        output = self.lockdrop(output_org, self.dropout if self.use_dropout else 0)
        return output
    
    #def init_hidden(self, bsz):
    #    weight = next(self.parameters())
    #    if self.model_type == 'LSTM':
    #        return (weight.new_zeros(self.nlayers*2, bsz, self.nhid),
    #                weight.new_zeros(self.nlayers*2, bsz, self.nhid))
    #    else:
    #        return weight.new_zeros(self.nlayers, bsz, self.nhid)
    
class RNN_pooler(nn.Module):
    def __init__(self, nhid):
        super(RNN_pooler, self).__init__()
        self.nhid = nhid

    def forward(self, output, bsz):
        output_unpacked = output.view(output.size(0), bsz, 2, self.nhid)
        #print(output_unpacked.size())
        output_last = torch.cat( (output_unpacked[-1,:,0,:], output_unpacked[0,:,1,:]) , dim = 1)
        #print(output_last.size())
        #forward_mean = torch.mean(output_unpacked[:,:,0,:], dim = 0)
        #backward_mean = torch.mean(output_unpacked[:,:,1,:], dim = 0)
        #output_last = torch.cat( (forward_mean, backward_mean), dim = 1 )
        #output_last = output_last / (0.000000000001 + output_last.norm(dim = 1, keepdim=True) )
        return output_last

class TRANS_pooler(nn.Module):
    def __init__(self, method = 'last'):
        super(TRANS_pooler, self).__init__()
        self.method = method

    def forward(self, output):
        if self.method == 'last':
            output_last = output[-1,:,:]
        elif self.method == 'avg':
            output_last = torch.mean(output, dim = 0)
        return output_last


class seq_to_emb(nn.Module):
    def __init__(self, model_type_list, ninp, nhid, nlayers, dropout, max_sent_len, trans_layers, trans_nhid, num_type_feature):
        super(seq_to_emb, self).__init__()
        self.encoder_array = nn.ModuleList()
        input_dim = ninp
        for i, model_type in enumerate(model_type_list):
            if model_type == 'LSTM' or model_type == 'GRU':
                model = RNN_encoder(model_type, input_dim, nhid, nlayers, dropout)
                input_dim = nhid * 2
            elif model_type == 'TRANS':
                self.linear_trans_dim = None
                if input_dim != trans_nhid:
                    self.linear_trans_dim = nn.Linear(input_dim, trans_nhid)
                    
                if i == 0:
                    add_position_emb = True
                else:
                    add_position_emb = False
                #model = model_trans.BertEncoder(model_type = model_type, hidden_size = input_dim, max_position_embeddings = max_sent_len, num_hidden_layers=trans_layers, add_position_emb = add_position_emb)
                #model = model_trans.Transformer(model_type = model_type, hidden_size = input_dim, max_position_embeddings = max_sent_len, num_hidden_layers=trans_layers, add_position_emb = add_position_emb)
                model = model_trans.Transformer(model_type = model_type, hidden_size = trans_nhid, max_position_embeddings = max_sent_len, num_hidden_layers=trans_layers, add_position_emb = add_position_emb, num_type_feature=num_type_feature, dropout_prob=dropout)
                input_dim = trans_nhid
            else:
                print("model type must be either LSTM or TRANS")
                sys.exit(1)
            self.encoder_array.append( model )
        if model_type_list[-1] == 'LSTM' or model_type_list[-1] == 'GRU':
            self.pooler = RNN_pooler(nhid)
        else:
            self.pooler = TRANS_pooler(method = 'last')
        
        self.model_type_list = model_type_list
        #self.dim_adjuster = None
        #if input_dim != nhid * 2:
        #    self.dim_adjuster = nn.Linear(input_dim, nhid * 2)
        self.output_dim = input_dim

    def forward(self, emb, input_type):
        bsz = emb.size(1)
        hidden_states = emb
        for model in self.encoder_array:
            model_type = model.model_type
            if model_type == 'LSTM' or model_type == 'GRU':
                hidden_states = model(hidden_states)
            elif model_type == 'TRANS':
                #If we want to use transformer by default at the end, we will want to reconsider reducing the number of permutes
                if self.linear_trans_dim is not None:
                    hidden_states = self.linear_trans_dim(hidden_states)
                hidden_states = hidden_states.permute(1,0,2)
                hidden_states = model(hidden_states, token_type_ids = input_type)
                hidden_states = hidden_states[0].permute(1,0,2)
        if self.model_type_list[-1] == 'LSTM' or self.model_type_list[-1] == 'GRU':
            output_emb = self.pooler(hidden_states, bsz)
        else:
            output_emb = self.pooler(hidden_states)
        #if self.dim_adjuster is not None:
        #    output_emb = self.dim_adjuster(output_emb)
        return output_emb, hidden_states

#class RNNModel_simple(nn.Module):
class SEQ2EMB(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, model_type_list, ntoken, ninp, nhid, nlayers, dropout, dropouti, dropoute, max_sent_len, external_emb, init_idx = [], trans_layers=2, trans_nhid=300, num_type_feature=6):
        #super(RNNModel_simple, self).__init__()
        super(SEQ2EMB, self).__init__()
        #self.drop = nn.Dropout(dropout)
        self.lockdrop = LockedDropout()
        if len(external_emb) > 1 and ninp == 0:
            ntoken, ninp = external_emb.size()
            scale_factor = math.sqrt(ninp)
            self.encoder = nn.Embedding.from_pretrained(external_emb.clone() * scale_factor, freeze = False)
            #self.encoder = nn.Embedding.from_pretrained(external_emb.clone(), freeze = False)
            if len(init_idx) > 0:
                print("Randomly initializes embedding for ", len(init_idx), " words")
                device = self.encoder.weight.data.get_device()
                extra_init_emb = torch.randn(len(init_idx), ninp, requires_grad = False, device = device)
                #extra_init_emb = extra_init_emb / (0.000000000001 + extra_init_emb.norm(dim = 1, keepdim=True))
                self.encoder.weight.data[init_idx, :] = extra_init_emb
        else:
            self.encoder = nn.Embedding(ntoken, ninp)
        self.use_dropout = True

        #self.init_weights()
        self.seq_summarizer =  seq_to_emb(model_type_list, ninp, nhid, nlayers, dropout, max_sent_len, trans_layers, trans_nhid, num_type_feature)
        self.output_dim = self.seq_summarizer.output_dim

        #self.model_type = model_type
        #self.nhid = nhid
        #self.nlayers = nlayers
        #self.dropout = dropout
        self.dropoute = dropoute
        self.dropouti = dropouti

    #def init_weights(self):
    #    initrange = 0.1
    #    self.encoder.weight.data.uniform_(-initrange, initrange)
    #    self.encoder.weight.data[0,:] = 0
    #    self.decoder.bias.data.zero_()

    def forward(self, input, input_type):
        
        #bsz = input.size(1)
        bsz = input.size(0)
        #print(input.size())
        #hidden = self.init_hidden(bsz)

        #weight = next(self.parameters())
        #input_long = weight.new_tensor(input,dtype = torch.long)

        emb = embedded_dropout(self.encoder, input.t(), dropout=self.dropoute if self.use_dropout else 0)
        emb = self.lockdrop(emb, self.dropouti if self.use_dropout else 0)
        #emb = self.drop(self.encoder(input))
        
        output_last, output = self.seq_summarizer(emb, input_type)
        #return output, hidden, output_last #If we want to output output, hidden, we need to shift the batch dimension to the first dim in order to use nn.DataParallel correctly
        return output_last, output.permute(1,0,2)


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
