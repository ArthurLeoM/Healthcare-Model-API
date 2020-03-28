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
fn = './concare2web/'
center_dict = pickle.load(open(fn + 'clu_id2centers', 'rb'))
clu_pdid = pickle.load(open(fn + 'clu_id2pdid', 'rb'))
hidden_dict = pickle.load(open(fn + 'pdid2hiddens', 'rb'))
MAX = 99999


def getCluster(context, top_num=5):
    cluster_dist = MAX
    cluster_id = 0
    for k, v in center_dict.items():
        cur_dist = np.sqrt(np.sum(np.square(context - v)))
        if cluster_dist > cur_dist:
            cluster_dist = cur_dist
            cluster_id = k
    dist_dict = {}
    for pdid in clu_pdid[cluster_id]:
        dist_dict[pdid] = np.sqrt(np.sum(np.square(context - hidden_dict[pdid])))
    sort_dist = sorted(dist_dict.items(), key=lambda x: x[1])
    top_pdid = []
    for i in range(top_num):
        top_pdid.append(sort_dist[i][0])
    return cluster_id, top_pdid