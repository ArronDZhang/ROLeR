import pickle
import torch
import os
import pandas as pd
from vis import svd_reduction_and_visualized_with_color, just_svd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import seaborn as sns
from collections import Counter

from src.core.configs import get_training_data, get_features
from environments.KuaiRec.env.KuaiEnv import KuaiEnv
import argparse
import random
from src.core.user_model_ensemble import EnsembleModel

from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance

y_path = 'environments/RL4Rec/data/coat_pseudoGT_ratingM.ascii'
y_mat = pd.read_csv(y_path, sep="\s+", header=None, dtype=str).to_numpy(dtype=int)

with open('/home/s4715423/DORL-codes/saved_models/KuaiRand-v0/DeepFM/matsPre/[pointneg]_matPre.pickle', 'rb') as f:
    matpre = pickle.load(f)    

def load_mat():
    filename_GT = os.path.join(DATAPATH, "..", "RL4Rec", "data", "coat_pseudoGT_ratingM.ascii")
    mat = pd.read_csv(filename_GT, sep="\s+", header=None, dtype=str).to_numpy(dtype=int)

    # mat_distance = get_distance_mat(mat)

    num_item = mat.shape[1]
    distance = np.zeros([num_item, num_item])
    mat_distance = get_distance_mat1(mat, distance)

    df_item = CoatEnv.load_item_feat()

    return mat, df_item, mat_distance