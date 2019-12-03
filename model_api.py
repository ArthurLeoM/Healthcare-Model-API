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

import json
import tornado
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
from tornado.options import define, options, parse_config_file
from tornado.web import Application, RequestHandler
from tornado.escape import json_decode, json_encode, utf8

def genData(raw):
    static = raw['patient']
    lab = raw['lab']
    vis_num = len(lab)
    lab_mat = np.zeros(shape=(1, vis_num)) # 1 t
    order = ['cl', 'co2', 'wbc', 'hgb', 'urea', 'ca', 'k', 'na', 'scr', 'p', 'alb', 'crp', 'glu', 'amount', 'weight','sys','dia']
    for idx in order:
        tmp = list()
        for cur in lab:
           tmp.append(cur[idx])
        tmp_arr = np.array(tmp)
        mu = np.mean(tmp_arr)
        sigma = np.std(tmp_arr)
        tmp_arr = ((tmp_arr - mu)/sigma).reshape(1, vis_num)
        lab_mat = np.r_[lab_mat, tmp_arr]            # f+1 t
    lab_mat = np.transpose(lab_mat[1:, :])           # t f
    lab_mat = np.expand_dims(lab_mat, axis=0)        # b(1) t f
    return lab_mat


def runAda(data):
    device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
    print("available device: {}".format(device))

    model = AdaCare(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint = torch.load('./saved_weights/AdaCare')
    save_chunk = checkpoint['chunk']
    print("last saved model is in chunk {}".format(save_chunk))
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()

    with torch.no_grad():
        test_x = torch.tensor(data, dtype=torch.float32).to(device)
        if test_x.size()[1] > 400:
            test_x = test_x[:, :400, :]
        test_output, test_att = model(test_x, device)  #output: 1 t 1, att: 1 t f
    test_output = test_output.numpy()
    test_att = test_att.numpy()
    return [test_output, test_att]


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
        data = genData(raw_data)
        output, att = runAda(data)
        output = output.squeeze().tolist()
        att = att.squeeze().tolist()
        result = {
            "predict": output,
            "attention": att
        }
        self.write(json_encode(result))

define('port', type=int, default=8888, multiple=False)
parse_config_file('config')

app = Application([('/',IndexHandler)])
server = HTTPServer(app)
server.listen(options.port)
IOLoop.current().start()

        