import numpy as np
import argparse
import os
import imp
import re
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

RANDOM_SEED = 12345
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True

from utils import utils
from utils.readers import DecompensationReader
from utils.preprocessing import Discretizer, Normalizer
from utils import metrics
from utils import common_utils
from model import AdaCare
from LM import patient_LM
from concare_new import vanilla_transformer_encoder
from check_clu import getCluster

import json
import tornado
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options, parse_config_file
from tornado.web import Application, RequestHandler
from tornado.escape import json_decode, json_encode, utf8

device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
print("available device: {}".format(device))
order = ['cl', 'co2', 'wbc', 'hgb', 'urea', 'ca', 'k', 'na', 'cre', 'p', 'alb', 'crp', 'glu', 'amount', 'weight','sys','dia']

mu = {
    'cl': 98.01925300471626,
    'wbc' : 8.28950174958162,
    'co2': 27.38820021299255,  
    'hgb' : 114.67975886201126,
    'urea' : 19.97359577057660,
    'ca' : 2.39811197322379, 
    'k' : 4.31358892438765, 
    'na' : 138.51236117450175, 
    'cre' : 857.37942492012780, 
    'p' : 1.59916020082154, 
    'alb' : 37.64694888178914, 
    'crp' : 8.70053095998783, 
    'glu' : 6.66997261524418, 
    'amount' : 3435.23214518674882, 
    'weight' : 62.17573065591684,
    'sys': 133.49089456869010,
    'dia' : 77.64016430853492
}

sigma = {
    'cl': 4.835784044260248,
    'wbc' : 55.38874579878202,
    'co2': 3.641592765723632,  
    'hgb' : 16.997849288372166,
    'urea' : 5.47088344532125,
    'ca' : 0.3721553027810553, 
    'k' : 0.7237871536552675, 
    'na' : 4.412943954176232, 
    'cre' : 283.59559363802396, 
    'p' : 0.4325994378061795, 
    'alb' : 4.4367471661194875, 
    'crp' : 15.973240032371143, 
    'glu' : 3.0444014878379098, 
    'amount' : 1262.535925457892, 
    'weight' : 11.479039918250724,
    'sys' : 24.512420744080206,
    'dia' : 13.971425472954335
}

def genData(raw):
    static = raw['patient']
    lab = raw['lab']
    vis_num = len(lab)
    lab_mat = np.zeros(shape=(1, vis_num)) # 1 t
    for idx in order:
        tmp = list()
        for cur in lab:
            tmp.append(float(cur[idx]))
        tmp_arr = np.array(tmp)
        tmp_arr = ((tmp_arr - mu[idx])/sigma[idx]).reshape(1, vis_num)
        lab_mat = np.r_[lab_mat, tmp_arr]            # f+1 t
    lab_mat = np.transpose(lab_mat[1:, :])           # t f
    lab_mat = np.expand_dims(lab_mat, axis=0)        # b(1) t f
    return lab_mat


def runConcare(data):
    with torch.no_grad():
        data = data[0]
        data_len = np.size(data, axis=0)
        if data_len > 400:
            data = data[:400, :]
            data_len = 400

        if data_len == 1:
            test_x = np.stack((data, data), axis=0)
            test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
            test_len = np.array([test_x.size()[1], test_x.size()[1]])
            test_len = torch.tensor(test_len, dtype=torch.float32).to(device).int()
            test_output, context, attn = model_concare(test_x, test_len) 
            output = test_output.cpu().detach().numpy().squeeze().tolist()
            context = context.cpu().detach().numpy().squeeze().tolist()
            attn = attn.cpu().detach().numpy().squeeze().tolist()
            return output[0], context[0], attn[0]

        test_x = []
        test_len = []
        for i in range(data_len):
            cur_data = np.zeros((data_len, 17))
            idx = data_len - i
            cur_data[:idx] = data[:idx]
            test_x.append(cur_data)
            test_len.append(idx)
        test_x = np.array(test_x)
        #print(test_x)
        test_x = torch.tensor(test_x, dtype=torch.float32).to(device)
        test_len = np.array(test_len)
        test_len = torch.tensor(test_len, dtype=torch.float32).to(device).int()
        test_output, context, attn = model_concare(test_x, test_len) 
        output = test_output.cpu().detach().numpy().squeeze()
        output = np.flip(output, axis=0).tolist()
        context = context.cpu().detach().numpy().squeeze().tolist()
        attn = attn.cpu().detach().numpy()
        attn = np.flip(attn, axis=0)
        attn_dict = {}
        idx = 0
        for name in order:
            attn_dict[name] = attn[:, idx].tolist()
            idx += 1
        return output, context[0], attn_dict


class IndexHandler(RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "*")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def options(self):
        self.set_status(204)
        self.finish()
        
    def get(self, *args, **kwargs):
        result = {
            "predict": '0',
            "attention": '0'
        }
        self.write(json_encode(result))

    def post(self, *args, **kwargs):
        jsonbyte = self.request.body
        jsonstr = jsonbyte.decode('utf8')
        raw_data = json.loads(jsonstr)
        print("json resolved...")
        data = genData(raw_data)
        print("data generated...")

        con_output, con_context, con_attn = runConcare(data)
        cluster_id, top_pdid = getCluster(con_context, top_num=6)
        concare_res = {
            "predict": con_output,
            "attention": con_attn,
            "cluster_id": cluster_id,
            "cluster_top6_pdid": top_pdid
        }

        print("output generated...")
        self.write(json_encode(concare_res))


model_concare =  vanilla_transformer_encoder(device=device).to(device)
optimizer_concare = torch.optim.Adam(model_concare.parameters(), lr=0.001)

checkpoint_concare = torch.load('./saved_weights/concare-ckd', map_location=lambda storage, loc: storage)
model_concare.load_state_dict(checkpoint_concare['net'])
optimizer_concare.load_state_dict(checkpoint_concare['optimizer'])
model_concare.eval()

define('port', type=int, default=10407, multiple=False)

app = Application([('/',IndexHandler)])
server = HTTPServer(app)
server.listen(options.port)
IOLoop.current().start()
