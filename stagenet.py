import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

RANDOM_SEED = 12345
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic=True

class SACRNN(nn.Module):
    def __init__(self, input_dim=17, hidden_dim=72, conv_size=10, output_dim=1, levels=2, dropconnect=0., dropout=0., dropres=0.):
        super(SACRNN, self).__init__()
        
        assert hidden_dim % levels == 0
        self.dropout = dropout
        self.dropconnect = dropconnect
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv_dim = hidden_dim
        self.conv_size = conv_size
        self.output_dim = output_dim
        self.levels = levels
        self.chunk_size = hidden_dim // levels
        
        self.kernel = nn.Linear(input_dim+1, hidden_dim*4+levels*2)
        nn.init.xavier_uniform_(self.kernel.weight)
        nn.init.zeros_(self.kernel.bias)
        self.recurrent_kernel = nn.Linear(hidden_dim+1, hidden_dim*4+levels*2)
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.recurrent_kernel.bias)
        
        self.nn_conv = nn.Conv1d(hidden_dim, self.conv_dim, conv_size, 1)
        self.nn_output = nn.Linear(self.conv_dim, output_dim)
        
        if self.dropconnect:
            self.nn_dropconnect = nn.Dropout(p=dropconnect)
            self.nn_dropconnect_r = nn.Dropout(p=dropconnect)
        if self.dropout:
            self.nn_dropout = nn.Dropout(p=dropout)
            self.nn_dropres = nn.Dropout(p=dropres)
    
    def cumax(self, x, mode='l2r'):
        if mode == 'l2r':
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return x
        elif mode == 'r2l':
            x = torch.flip(x, [-1])
            x = torch.softmax(x, dim=-1)
            x = torch.cumsum(x, dim=-1)
            return torch.flip(x, [-1])
        else:
            return x
    
    def step(self, inputs, c_last, h_last, interval):
        x_in = inputs
        interval = interval.unsqueeze(-1)
        x_out1 = self.kernel(torch.cat((x_in, interval), dim=-1))
        x_out2 = self.recurrent_kernel(torch.cat((h_last, interval), dim=-1))
        if self.dropconnect:
            x_out1 = self.nn_dropconnect(x_out1)
            x_out2 = self.nn_dropconnect_r(x_out2)
        x_out = x_out1 + x_out2
        f_master_gate = self.cumax(x_out[:, :self.levels], 'l2r')
        f_master_gate = f_master_gate.unsqueeze(2)
        i_master_gate = self.cumax(x_out[:, self.levels:self.levels*2], 'r2l')
        i_master_gate = i_master_gate.unsqueeze(2)
        x_out = x_out[:, self.levels*2:]
        x_out = x_out.reshape(-1, self.levels*4, self.chunk_size)
        f_gate = torch.sigmoid(x_out[:, :self.levels])
        i_gate = torch.sigmoid(x_out[:, self.levels:self.levels*2])
        o_gate = torch.sigmoid(x_out[:, self.levels*2:self.levels*3])
        c_in = torch.tanh(x_out[:, self.levels*3:])
        c_last = c_last.reshape(-1, self.levels, self.chunk_size)
        overlap = f_master_gate * i_master_gate
        c_out = overlap * (f_gate * c_last + i_gate * c_in) + (f_master_gate - overlap) * c_last + (i_master_gate - overlap) * c_in
        h_out = o_gate * torch.tanh(c_out)
        c_out = c_out.reshape(-1, self.hidden_dim)
        h_out = h_out.reshape(-1, self.hidden_dim)
        out = torch.cat([h_out, f_master_gate[..., 0], i_master_gate[..., 0]], 1)
        return out, c_out, h_out
        
    
    def forward(self, input, time, device):
        batch_size, time_step, feature_dim = input.size()
        c_out = torch.zeros(batch_size, self.hidden_dim).to(device)
        h_out = torch.zeros(batch_size, self.hidden_dim).to(device)
        
        tmp_h = torch.zeros_like(h_out, dtype=torch.float32).view(-1).repeat(self.conv_size).view(self.conv_size, batch_size, self.hidden_dim).to(device)
        tmp_dis = torch.zeros((self.conv_size, batch_size)).to(device)
        h = []
        origin_h = []
        distance = []
        for t in range(time_step):
            out, c_out, h_out = self.step(input[:, t, :], c_out, h_out, time[:, t])
            cur_distance = 1 - torch.mean(out[..., self.hidden_dim:self.hidden_dim+self.levels], -1)
            cur_distance_in = torch.mean(out[..., self.hidden_dim+self.levels:], -1)
            origin_h.append(out[..., :self.hidden_dim])
            tmp_h = torch.cat((tmp_h[1:], out[..., :self.hidden_dim].unsqueeze(0)), 0)
            tmp_dis = torch.cat((tmp_dis[1:], cur_distance.unsqueeze(0)), 0)
            distance.append(cur_distance)
            
            local_dis = tmp_dis.permute(1, 0)
            local_dis = torch.flip(torch.cumsum(local_dis, dim=1), dims=[1])
            local_dis = torch.softmax(local_dis, dim=1)
            local_h = tmp_h.permute(1, 2, 0)

            local_h = local_h * local_dis.unsqueeze(1)
            local_h = self.nn_conv(local_h).squeeze(-1) 
            h.append(local_h)  

        origin_h = torch.stack(origin_h).permute(1, 0, 2)
        rnn_outputs = torch.stack(h).permute(1, 0, 2)
        rnn_outputs = rnn_outputs + origin_h
        rnn_outputs = rnn_outputs.contiguous().view(-1, rnn_outputs.size(-1)) #B*T c
        if self.dropout > 0.0:
            rnn_outputs = self.nn_dropout(rnn_outputs)
        output = self.nn_output(rnn_outputs)
        output = output.contiguous().view(batch_size, time_step, self.output_dim)
        output = torch.sigmoid(output)

        return output, torch.stack(distance)
