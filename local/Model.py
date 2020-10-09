from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class pGRULayer(nn.Module):
    def __init__(self,input_size,hidden_size, dropout_rate, n_layers=1, batch_first=False):
        super(pGRULayer, self).__init__()
        self.pyramid = 2
        self.batch_first = batch_first
        self.dropout_rate = dropout_rate
        self.n_layers = n_layers
        GRU = 'GRU'
        self.GRU = getattr(nn,GRU.upper())
        self.biGRU = self.GRU(input_size * 2,hidden_size,1,
                              dropout=self.dropout_rate,
                              bidirectional=False,
                              batch_first=self.batch_first)
    def forward(self,input_x):
        batch_size = input_x.size(1)
        timestep = input_x.size(0)
        feature_dim = input_x.size(2)
        if timestep%2 != 0:
            timestep = int(timestep-1)
            input_x = input_x[:timestep,:,:].contiguous()
            timestep = input_x.size(0)
        input_x = input_x.contiguous().view(int(timestep/2),batch_size,feature_dim*2)
        output,hidden = self.biGRU(input_x)
        return output, hidden

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1,
                    reduce_dim=1, dropout=0.1,batch_first=False,
                     ):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.reduce_dim=reduce_dim
        self.n_layers = n_layers
        self.batch_first=batch_first
        self.pGRU_layer1 = pGRULayer(input_size,hidden_size, dropout)
        self.pGRU_layer2 = pGRULayer(hidden_size * 2,hidden_size, dropout)
        self.pGRU_layer3 = pGRULayer(hidden_size * 2,hidden_size, dropout)
        for l in range(self.n_layers):
            layer_input_size = input_size if l==0 else hidden_size * 2
            setattr(self, 'layer'+str(l),nn.GRU(layer_input_size,hidden_size,n_layers,
                                                dropout=(0 if self.n_layers==1 else self.dropout),
                                                bidirectional=True,
                                                batch_first=self.batch_first))

    def forward(self, input_seq, input_lengths, hidden=None):
        if self.reduce_dim:
            x_temp= torch.tensor([]).cuda()
            for t in range(0,input_seq.size(0)-1,2):
                x_temp = torch.cat((x_temp,(torch.cat((input_seq[t],input_seq[t+1]),dim=1)).unsqueeze(0)),dim=0)
            input_lengths.div_(2)
            input_seq=x_temp
        output, hidden = self.pGRU_layer1(input_seq)
        output, hidden = self.pGRU_layer2(output)
        output, hidden = self.pGRU_layer3(output)
        return output, hidden

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)

    def las_score(self, query_outputs, key_outputs):
        energy = self.attn(key_outputs)
        return torch.sum(query_outputs * energy, dim=2)

    def forward(self, query_outputs, key_outputs):
        attn_energies = self.las_score(query_outputs, key_outputs)
        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class DecoderRNN(nn.Module):
    def __init__(self, embedding, hidden_size, output_size, n_layers=1, dropout=0.1, batch_first = False):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attn(hidden_size)
        self.key_network = nn.Linear(hidden_size, hidden_size)  
        self.value_network = nn.Linear(hidden_size, hidden_size) 

        for l in range(self.n_layers):
            layer_input_size = hidden_size
            setattr(self, 'layer'+str(l),nn.GRU(layer_input_size,hidden_size,n_layers,
                                                dropout=(0 if self.n_layers==1 else self.dropout),
                                                bidirectional=False,
                                                batch_first=self.batch_first))

    def forward(self, input_step, last_hidden, encoder_outputs):
        input_step = input_step.long()
        embedded = self.embedding(input_step)
        for l in range(self.n_layers):
            query_outputs, last_hidden = getattr(self,'layer'+str(l))(embedded, last_hidden)
        key_outputs = self.key_network(encoder_outputs) 
        value_outputs = self.value_network(encoder_outputs)
        attn_weights = self.attn(query_outputs, key_outputs)
        context = attn_weights.bmm(value_outputs.transpose(0, 1))
        query_outputs = query_outputs.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((query_outputs, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, last_hidden

