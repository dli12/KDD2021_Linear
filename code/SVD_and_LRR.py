
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


    
    
    
class MyClock:
    startTime = time.time()
    def tic(self):
        self.startTime = time.time()
    def toc(self):
        secs = time.time() - self.startTime 
        print("... elapsed time: {} min {} sec".format(int(secs//60), secs%60) )




def load_train_data(csv_file):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows),
                             (rows, cols)), dtype='float64',
                             shape=(n_users, n_items))
    return data

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



DATA_DIR = 'path/yourprocesseddata'





pro_dir = os.path.join(DATA_DIR, 'pro_sg')


# ## Load the pre-processed training and test data


unique_sid = list()
with open(os.path.join(pro_dir, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())

n_items = len(unique_sid)





# load training data
# user-item binary matrix
train_data = load_train_data(os.path.join(pro_dir, 'train.csv'))





from numpy.linalg import svd as npsvd



#Pre-computation of the item-item training-data (used by analytic solutions)
XtX= np.asarray(train_data.T.dot(train_data).todense(), dtype = np.float32)  
XtXdiag = deepcopy(np.diag(XtX))   
ii_diag = np.diag_indices(XtX.shape[0])



# compute SVD
u,s,vh = npsvd(XtX, hermitian = True)


np.savez(DATA_DIR + 'svd', u=u, s=s, vh=vh)



# load SVD

z = np.load(DATA_DIR + 'svd.npz')
u = z['u']
s = z['s']
vh = z['vh']




k = 1000
lamd = 5000



def getW(lamd = 5000, k = 1000, u = u,s = s,vh = vh):
    
    ss = s[:k]/(s[:k] + lamd)
    W = np.matmul(u[:,:k], (np.diag(ss)))
    W = np.matmul(W, vh[:k,:])
    
    return W


test_data_tr, test_data_te = load_tr_te_data(
    os.path.join(pro_dir, 'test_tr.csv'),
    os.path.join(pro_dir, 'test_te.csv'))

vali_data_tr, vali_data_te = load_tr_te_data(
    os.path.join(pro_dir, 'vali_tr.csv'),
    os.path.join(pro_dir, 'vali_te.csv'))



def evaluate(BB, test_data_tr = test_data_tr, test_data_te = test_data_te):
    
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
    print("Test NDCG@100=%.5f (%.5f)" % (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
    print("Test Recall@20=%.5f (%.5f)" % (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
    print("Test Recall@50=%.5f (%.5f)" % (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))



# Fast Grid Search LRR
ks = [1000, 2000, 3000]
lamds = [1000, 3000, 5000]
for i in ks:
    for j in lamds:
        W = getW(j, i)
        evaluate(BB, test_data_tr = vali_data_tr, test_data_te = vali_data_te)
        evaluate(BB, test_data_tr = test_data_tr, test_data_te = test_data_te)






