


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



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class DataReader(object):
    
    def __init__(self, train_df, num_item):
        
        self.train_df = train_df
        self.train_u_dict = train_df.groupby('user')['item'].apply(list).to_dict()
        self.trainUser = train_df['user'].values
        self.trainItem = train_df['item'].values
        self.num_item = num_item
    
    def bpr_getTrain(self, train_batch_size, N = 2):

        train_u = []
        train_pos_i = []
        train_neg_i = []

        u_list = self.trainUser
        i_list = self.trainItem

        for index in range(len(u_list)):

            u = u_list[index]
            i = i_list[index]
            train_u.extend([u]*(N))
            train_pos_i.extend([i]*(N))

            PositiveSet = set(self.train_u_dict[u]) 

            for t in range(N):# sample negative items
                neg_i = np.random.randint(0, self.num_item)
                while neg_i in PositiveSet:
                    neg_i = np.random.randint(0, self.num_item)
                train_neg_i.append(neg_i)

        train_dataset = BPRDataset(train_u, train_pos_i, train_neg_i)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size = train_batch_size, 
                                      shuffle = True,
                                      num_workers = 4,
                                      pin_memory = True,
                                     )

        return train_dataloader
    
class BPRDataset(Dataset):

    def __init__(self, users, pos_items, neg_items):

        self.users = users
        self.pos_items = pos_items
        self.neg_items = neg_items

    def __len__(self):

        return len(self.users)

    def __getitem__(self, idx):

        user = self.users[idx]
        pos_item = self.pos_items[idx]
        neg_item = self.neg_items[idx]

        sample = {'user':user, 'pos_item':pos_item, 'neg_item':neg_item}

        return sample  

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    return DCG / IDCG

def Recall_at_k_batch(X_pred, heldout_batch, k=100):
    batch_users = X_pred.shape[0]

    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    X_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall










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
    
    
    return data_tr, data_te



#%%%%%%-----------------Load Data-----------------

# change to the location of the data
DATA_DIR = 'path'

pro_dir = os.path.join(DATA_DIR, 'pro_sg')



unique_sid = list()
with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())

n_items = len(unique_sid)


# load training data
train_data, train_df = load_train_data(os.path.join(pro_dir, 'train.csv'))



test_data_tr, test_data_te = load_tr_te_data(
    os.path.join(pro_dir, 'test_tr.csv'),
    os.path.join(pro_dir, 'test_te.csv'))

vali_data_tr, vali_data_te = load_tr_te_data(
    os.path.join(pro_dir, 'validation_tr.csv'),
    os.path.join(pro_dir, 'validation_te.csv'))





data_reader = DataReader(train_df, n_items)



# Load SVD
z = np.load(DATA_DIR + 'svd.npz')


s = z['s']
vh = z['vh']
u = z['u']




class LRR(nn.Module):
    
    def __init__(self, num_item, UI, k, u, s, vh, lamd0 = 8000., C = 1000., p = 0.5):
        
        super().__init__()
              
        self.UI = UI
        self.num_item = num_item
        
        self.lamd = nn.Embedding(k, 1)
        self.D = nn.Embedding(num_item, 1)
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
        nn.init.zeros_(self.D.weight)
        nn.init.zeros_(self.Db.weight)
        
    def getW(self):
        
        ss = self.s/(self.s + self.lamd0 + self.C * torch.tanh(self.lamd.weight.view(-1)))
                
        W = torch.matmul(self.u, torch.diag(ss))
        W = torch.matmul(W, self.vh)
        
        W = torch.sigmoid(self.D.weight) * W
        W = W * torch.sigmoid(self.Db.weight)

        W[torch.arange(self.num_item), torch.arange(self.num_item)] = 0
       
        return W    
        
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

        pos = (A * posW).sum(1)
        neg = (A * negW).sum(1)

        loss = - torch.log(torch.sigmoid(100*(pos - neg))).mean()

        return loss





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




device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(101)


k = 2000
lr = 0.001
epochs = 10
train_batch_size = 2048
N = 8 # negative sampling

p = 0.5 # dropout rate
C = 5000. # for netflix
lamd0 = 40000. # for netflix


#If you dont have enough memory, try to pass user-item row to GPU when necessary 
UI = torch.tensor(train_data.toarray()).float().to(device)


tu = torch.tensor(u[:,:k]).float().to(device)
ts = torch.tensor(s[:k]).float().to(device)
tvh = torch.tensor(vh[:k,:]).float().to(device)




model = LRR(n_items, UI, k, tu, ts, tvh, lamd0 = lamd0, C = C, p =p).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = lr)




fname = "NETFLIX_LRR_Lambda_HT_RMD"
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
           
        if (idx+1)%5000 == 0:
            
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



