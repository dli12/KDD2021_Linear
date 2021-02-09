#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import sys
import time
from copy import deepcopy

import numpy as np
from scipy import sparse
import pandas as pd
import bottleneck as bn
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib.layers import apply_regularization, l2_regularizer


# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# In[3]:


from utils import *


# In[4]:


def load_train_data(csv_file):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    
    train_df = pd.DataFrame()
    train_df['user'] = np.asarray(rows)
    train_df['item'] = np.asarray(cols)
    
    return data, train_df

def load_tr_te_data(csv_file_tr, csv_file_te):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                             (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te),
                             (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    
#     train_df = pd.DataFrame()
#     train_df['user'] = np.asarray(rows_tr)
#     train_df['item'] = np.asarray(cols_tr)
    
    return data_tr, data_te


# In[5]:


# change to the location of the data
DATA_DIR = '/users/kent/dli12/data/netflix/'

#itemId='movieId'   # for ml-20m data

pro_dir = os.path.join(DATA_DIR, 'pro_sg')


# In[6]:


pro_dir = os.path.join(DATA_DIR, 'pro_sg')


# In[7]:


unique_sid = list()
with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())

n_items = len(unique_sid)


# In[8]:


# load training data
train_data, _ = load_train_data(os.path.join(pro_dir, 'train.csv'))


# In[9]:


t_user = train_data.shape[0]



uorder = np.arange(t_user)
np.random.seed(1008)
np.random.shuffle(uorder)


# In[12]:


keepu = 70000
keeprows = uorder[:keepu]


# In[13]:


sub_train = train_data[keeprows]


# In[14]:


subu = sub_train.tocoo().row
subi = sub_train.tocoo().col

train_df = pd.DataFrame()
train_df['user'] = subu
train_df['item'] = subi


# In[15]:


test_data_tr, test_data_te = load_tr_te_data(
    os.path.join(pro_dir, 'test_tr.csv'),
    os.path.join(pro_dir, 'test_te.csv'))

vali_data_tr, vali_data_te = load_tr_te_data(
    os.path.join(pro_dir, 'validation_tr.csv'),
    os.path.join(pro_dir, 'validation_te.csv'))


# In[16]:


data_reader = DataReader(train_df, n_items)


# In[17]:


z = np.load(DATA_DIR + 'svd.npz')




s = z['s']
vh = z['vh']
u = z['u']




class PCA(nn.Module):
    
    def __init__(self, num_item, UI, k, u, s, vh, lamd0 = 8000., C = 1000., p = 0.5):
        
        super().__init__()
              
        self.UI = UI
        self.num_item = num_item
        
        self.lamd = nn.Embedding(k, 1)
        #self.D = nn.Embedding(num_item, 1)
        self.Db = nn.Embedding(1, num_item)
        
 
        self.s = s
        self.u = u
        self.vh = vh
        
        self.lamd0 = lamd0
        self.C = C
        self.drop = nn.Dropout(p = p)
        
        self._init_weight()
        
    def _init_weight(self):

        nn.init.zeros_(self.lamd.weight)
        #nn.init.zeros_(self.D.weight)
        nn.init.zeros_(self.Db.weight)
        
    def getW(self):
        
        ss = self.s/(self.s + self.lamd0 + self.C * torch.tanh(self.lamd.weight.view(-1)))
                
        W = torch.matmul(self.u, torch.diag(ss))
        W = torch.matmul(W, self.vh)
        
        #W = torch.sigmoid(self.D.weight) * W
        W = W * torch.sigmoid(self.Db.weight)
        W[torch.arange(self.num_item), torch.arange(self.num_item)] = 0
        #W[torch.arange(self.num_item), torch.arange(self.num_item)] = 0
        #W = W - torch.diag(torch.diag(W))
        
        return W    
    
    def bpr_loss(self, user, pos_item, neg_item, decay = 1e-3):
        
        W = self.getW()
        c = torch.arange(user.shape[0])
        
        #A = torch.tensor(self.UI[user].toarray()).float().to(device)
        A = self.UI[user]
        A[c, pos_item] = 0
        A = self.drop(A)
        
        AW = torch.matmul(A, W)
        
        pos = AW[c, pos_item]
        neg = AW[c, neg_item]
        
        loss = - torch.log(torch.sigmoid(100*(pos - neg))).mean()
        
        return loss 
    
    def bpr_loss2(self, user, pos_item, neg_item):
        
        #A = torch.tensor(self.UI[user].toarray()).float().to(device)
        A = self.UI[user]

        W = self.getW()
        
        posW = W[:, pos_item] 
        negW = W[:, neg_item]
        
        posW = torch.transpose(posW, 0, 1)
        negW = torch.transpose(negW, 0, 1)
        
        c = torch.arange(user.shape[0])
        A[c, pos_item] = 0
        
        A = self.drop(A)

        #AW = torch.matmul(A, W)
        pos = (A * posW).sum(1)
        neg = (A * negW).sum(1)
        #pos = AW[c, pos_item]
        #neg = AW[c, neg_item]

        loss = - torch.log(torch.sigmoid(100*(pos - neg))).mean()

        return loss


# In[20]:


def evaluate(f, BB, test_data_tr = test_data_tr, test_data_te = test_data_te):
    
    print("evaluating ...")
    N_test = test_data_tr.shape[0]
    idxlist_test = range(N_test)

    batch_size_test = 5000
    n100_list, r20_list, r50_list = [], [], []
    for bnum, st_idx in enumerate(range(0, N_test, batch_size_test)):
        end_idx = min(st_idx + batch_size_test, N_test)
        X = test_data_tr[idxlist_test[st_idx:end_idx]]

        if sparse.isspmatrix(X):
            X = X.toarray()
        X = X.astype('float32')
        

        pred_val = X.dot(BB)
        # exclude examples from training and validation (if any)
        pred_val[X.nonzero()] = -np.inf
        n100_list.append(NDCG_binary_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=100))
        r20_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=20))
        r50_list.append(Recall_at_k_batch(pred_val, test_data_te[idxlist_test[st_idx:end_idx]], k=50))

    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)
    r50 = (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list)))
    print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
    print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
    print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))

    f.write("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
    f.write("\n")
    f.write("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
    f.write("\n")
    f.write("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))
    f.write("\n")

    return r50


# In[21]:


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(101)


# In[22]:


k = 2000
lr = 0.001
epochs = 5
train_batch_size = 2048
N = 1 # negative sampling

p = 0.5 # dropout rate
C = 5000. # for netflix
lamd0 = 40000. # for netflix


# In[23]:


UI = torch.tensor(sub_train.toarray()).float().to(device)


# In[24]:


#UI = torch.tensor(np.asarray(train_data.todense(), dtype = np.float32)).float().to(device)
#UI = train_data
tu = torch.tensor(u[:,:k]).float().to(device)
ts = torch.tensor(s[:k]).float().to(device)
tvh = torch.tensor(vh[:k,:]).float().to(device)


# In[25]:


model = PCA(n_items, UI, k, tu, ts, tvh, lamd0 = lamd0, C = C, p =p).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)


# In[26]:


fname = "[used]NETFLIX_PCA_2000_LT_RMD_SUBSET"
f = open("save/" + fname + ".txt", "a")
print('initialization:')
f.write('initialization:')
f.write("\n")
t0 = time.time()
model.eval()
BB = model.getW().detach().cpu().numpy()
print('validation data')
f.write('validation data')
f.write("\n")
evaluate(f, BB, test_data_tr = vali_data_tr, test_data_te = vali_data_te)
print('test data')
f.write('test data')
f.write("\n")
evaluate(f, BB, test_data_tr = test_data_tr, test_data_te = test_data_te)
t1 = time.time()
f.write('evaluation take time %f'%(t1-t0))
f.write("\n")
f.close()


rmax = 0
emax = 0
for epoch in range(epochs):
    
    t0 = time.time()
    train_dataloader = data_reader.bpr_getTrain(train_batch_size, N)
    t1 = time.time()
    print('sample time: %f'%(t1-t0))
    
    for idx, batch_data in enumerate(train_dataloader):
        model.train()
     
        user = batch_data['user'].long().to(device)
        pos_item = batch_data['pos_item'].long().to(device)
        neg_item = batch_data['neg_item'].long().to(device)
        
        model.zero_grad()
        loss = model.bpr_loss2(user, pos_item, neg_item)
            
        loss.backward()
        optimizer.step()
        

        
        if (idx+1)%2000 == 0:
            
            f = open("save/"+fname + ".txt", "a")
            model.eval()
            BB = model.getW().detach().cpu().numpy()

            print('validation data:')
            f.write('validation data:')
            f.write("\n")
            rcurr,_ = evaluate(f, BB, test_data_tr = vali_data_tr, test_data_te = vali_data_te)

            if rcurr > rmax:
                rmax = rcurr
                emax = epoch
                f.write("-------------------")
                f.write("\n")
                f.write("max epoch%d, max idx%d"%(emax, idx+1))
                f.write("\n")
                torch.save(model.state_dict(), "save_model/" + fname + ".pt")

            print('test data:')
            f.write('test data:')
            f.write("\n")
            evaluate(f, BB, test_data_tr = test_data_tr, test_data_te = test_data_te)
            f.close()



