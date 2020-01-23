import numpy as np
import argparse
import os
import imp
import re
import pickle
import datetime
import random
import math

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED) #numpy
random.seed(RANDOM_SEED)

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

torch.manual_seed(RANDOM_SEED) # cpu
torch.cuda.manual_seed(RANDOM_SEED) #gpu
torch.backends.cudnn.deterministic=True # cudnn

from utils import utils
from utils.readers import InHospitalMortalityReader
from utils.preprocessing import Discretizer, Normalizer
from utils import metrics
from utils import common_utils

class Attention(nn.Module):
    def __init__(self, attention_type='add', hidden_dim=64, input_dim=128, demographic_dim=12, attention_width=None, history_only=True, time_aware=False, use_demographic=False, device='cuda'):
        super(Attention, self).__init__()
        
        self.attention_type = attention_type
        self.attention_width = attention_width
        self.hidden_dim = hidden_dim
        self.input_dim = 128
        self.history_only = history_only
        self.use_demographic = use_demographic
        self.demographic_dim = demographic_dim
        self.time_aware = time_aware
        self.device = device
        if history_only and attention_width == None:
            self.attention_width = 1e7
            
        if attention_type == 'add':
            if self.time_aware == True:
                self.Wx = nn.Parameter(torch.randn(input_dim+1, hidden_dim))
            else:
                self.Wx = nn.Parameter(torch.randn(input_dim, hidden_dim))
            self.Wt = nn.Parameter(torch.randn(input_dim, hidden_dim))
            self.Wd = nn.Parameter(torch.randn(demographic_dim, hidden_dim))
            self.bh = nn.Parameter(torch.zeros(hidden_dim,))
            self.Wa = nn.Parameter(torch.randn(hidden_dim, 1))
            self.ba = nn.Parameter(torch.zeros(1,))
            
            nn.init.kaiming_uniform_(self.Wd, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == 'mul':
            self.Wa = nn.Parameter(torch.randn(input_dim, input_dim))
            self.ba = nn.Parameter(torch.zeros(1,))
            
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == 'concat':
            self.Wh = nn.Parameter(torch.randn(2*input_dim, hidden_dim))
            self.Wa = nn.Parameter(torch.randn(hidden_dim, 1))
            self.ba = nn.Parameter(torch.zeros(1,))
            
            nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        else:
            raise RuntimeError('Wrong attention type.')
        
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
    
    def forward(self, input, demo=None, time=None):
        #time dim B*T*1
        batch_size, time_step, input_dim = input.size()
        assert(input_dim == self.input_dim)
        
        if self.attention_type == 'add': #B*T*I  @ H*I
            q = torch.matmul(input, self.Wt)
            q = torch.reshape(q, (batch_size, time_step, 1, self.hidden_dim)) #B*T*1*H
            if self.time_aware == True:
                k_input = torch.cat((input, time), dim=-1)
                k = torch.matmul(k_input, self.Wx)
                k = torch.reshape(k, (batch_size, 1, time_step, self.hidden_dim)) #B*1*T*H
            else:
                k = torch.matmul(input, self.Wx)
                k = torch.reshape(k, (batch_size, 1, time_step, self.hidden_dim)) #B*1*T*H
            if self.use_demographic == True:
                d = torch.matmul(demo, self.Wd) #B*H
                d = torch.reshape(d, (batch_size, 1, 1, self.hidden_dim))
            h = q + k + d + self.bh
            h = self.tanh(h) #B*T*T*H
            e = torch.matmul(h, self.Wa) + self.ba #B*T*T*1
            e = torch.reshape(e, (batch_size, time_step, time_step))
        elif self.attention_type == 'mul':
            e = torch.matmul(input, self.Wa)
            e = torch.bmm(e, input.permute(0,2,1)) + self.ba #B*T*I @ B*I*T
        elif self.attention_type == 'concat':
            q = input.unsqueeze(2).repeat(1,1,time_step,1)
            k = input.unsqueeze(1).repeat(1,time_step,1,1)
            c = torch.cat((q, k), dim=-1) #B*T*T*2I
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba #B*T*T*1
            e = torch.reshape(e, (batch_size, time_step, time_step))
        
        e = torch.exp(e - torch.max(e, dim=-1, keepdim=True).values)
        
        if self.attention_width is not None:
            if self.history_only:
                lower = torch.arange(0, time_step).to(self.device) - (self.attention_width - 1)
            else:
                lower = torch.arange(0, time_step).to(self.device) - self.attention_width // 2
            lower = lower.unsqueeze(-1)
            upper = lower + self.attention_width
            indices = torch.arange(0, time_step).unsqueeze(0).to(self.device)
            e = e * (lower <= indices).float() * (indices < upper).float()
        
        s = torch.sum(e, dim=-1, keepdim=True)
        e = e / (s + 1e-7)
        v = torch.bmm(e, input) #B*T*H 

        return v, e

class patient_LM(nn.Module):
    def __init__(self, cell='GRU', use_demo=False, demo_dim=4, input_dim=17, hidden_dim=16, output_dim=1, dropout=0.3, device='cuda'):
        super(patient_LM, self).__init__()
        self.cell = cell
        self.use_demo = use_demo
        self.demo_dim = demo_dim
        self.input_dim = input_dim
        if self.use_demo:
            self.input_dim += demo_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.att_dim = 16
        self.dropout = dropout
        self.device = device
        
        if self.cell == 'lstm':

            self.rnn_context = nn.LSTMCell(self.input_dim, self.hidden_dim)
        else:

            self.rnn_context = nn.GRUCell(self.input_dim, self.hidden_dim)
            
        self.nn_output = nn.Linear(2 * self.hidden_dim, self.output_dim)
        #self.re_output = nn.Linear(self.hidden_dim, self.input_dim)
        
        self.next_mid_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.next_opt_layer = nn.Linear(self.hidden_dim, self.input_dim)

        
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.nn_dropout = nn.Dropout(p=dropout)
                
        self.att_i = nn.Linear(self.hidden_dim, self.att_dim)
        self.att_t = nn.Linear(self.hidden_dim, self.att_dim)
        self.att = nn.Linear(self.att_dim, 1)
        
    def forward(self, input):
        batch_size = input.size(0)
        time_step = input.size(1)
        feature_dim = input.size(2)
        
      
        cur_h_context = Variable(torch.zeros(batch_size, self.hidden_dim)).to(self.device)
        if self.cell == 'LSTM':
            
            cur_c_context = Variable(torch.zeros(batch_size, self.hidden_dim)).to(self.device)
        
        h_task = []
        h_context = []
        for cur_time in range(time_step):
            if self.cell == 'lstm':
               
                rnn_state_context = (cur_h_context, cur_c_context)
                cur_h_context, cur_c_context = self.rnn_context(input[:, cur_time, :], rnn_state_task)
            else:
                
                cur_h_context = self.rnn_context(input[:, cur_time, :], cur_h_context)

          
            h_context.append(cur_h_context) # t b h
        
        # a_t = self.att_t(cur_h)
        h_context_stack = torch.stack(h_context).permute(1,0,2) # b t h
        # a_i = h_stack.contiguous().view(batch_size * time_step, self.hidden_dim)
        # a_i = self.att_i(a_i)
        # a_i = a_i.contiguous().view(batch_size, time_step, self.att_dim)
        # a = a_t.unsqueeze(1) + a_i
        # a = self.tanh(a)
        # a = self.att(a)
        # a = self.softmax(a)
        # cur_h = torch.sum(a * h_stack, dim=1)
        
        # if self.dropout > 0.0:
        #     cur_h = self.nn_dropout(cur_h)

        next_output = self.next_opt_layer(self.relu(self.next_mid_layer(h_context_stack)))
        
        #print(this_output.shape)
    
                
        return next_output, h_context_stack

