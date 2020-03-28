import numpy as np
import argparse
import os
import imp
import re
import pickle
import random
import datetime
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
from concare import vanilla_transformer_encoder
from check_clu import getCluster
from stagenet import SACRNN

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
    'dia' : 77.64016430853492,
    'date' : 34.5249
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
    'dia' : 13.971425472954335,
    'date' : 29.5434
}


def calcDays(str1,str2):
    date1=datetime.datetime.strptime(str1[0:10],"%Y-%m-%d")
    date2=datetime.datetime.strptime(str2[0:10],"%Y-%m-%d")
    num=(date1-date2).days
    return num

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
    
    time_mat = np.zeros(shape=(1, vis_num))
    start = 0
    if vis_num > 400:
        start = vis_num - 400
    for i in range(start, vis_num):
        if i == start:
            continue
        time_mat[0][i] = (calcDays(lab[i]['date'], lab[i-1]['date'])/30 - mu['date'])/sigma['date']

    return lab_mat, time_mat


def runStageNet(x, t):
    with torch.no_grad():
        test_time = torch.tensor(t, dtype=torch.float32).to(device)
        test_x = torch.tensor(x, dtype=torch.float32).to(device)
        if test_x.size()[1] > 400:
            test_x = test_x[:, :400, :]
        test_output, test_stage = model_stage(test_x, test_time, device)  #output: 1 t 1, att: 1 t f
    ret_output = test_output.cpu().detach().numpy().squeeze().tolist()
    ret_stage = test_stage.cpu().detach().numpy().squeeze().tolist()
    return ret_output, ret_stage


class IndexHandler(RequestHandler):
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
        x, t = genData(raw_data)
        print("data generated...")

        sn_output, sn_stage = runStageNet(x, t)
        sn_res = {
            "predict": sn_output, 
            "stage": sn_stage
        }
        self.write(json_encode(sn_res))


model_stage = SACRNN().to(device)
optimizer_stage = torch.optim.Adam(model_stage.parameters(), lr=0.001)

checkpoint_stage = torch.load('./saved_weights/SA-CRNN_4_10', map_location=lambda storage, loc: storage)
model_stage.load_state_dict(checkpoint_stage['net'])
optimizer_stage.load_state_dict(checkpoint_stage['optimizer'])
model_stage.eval()

define('port', type=int, default=10409, multiple=False)

app = Application([('/',IndexHandler)])
server = HTTPServer(app)
server.listen(options.port)
IOLoop.current().start()
