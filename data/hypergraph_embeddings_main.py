
import pandas as pd
import scipy as sp
import scipy.sparse.linalg
from numpy import *
import numpy as np
import os
from args import args

def hypergraph_embeddings(hypergraph, phi):
    H = hypergraph
    epsilon = 0.1 ** 10
    tolerant = 0.1 ** 5

    num_nodes = hypergraph.shape[0]
    num_edges = hypergraph.shape[1]
    Du = np.sum(hypergraph,axis=1).reshape(-1)
    Di = np.sum(hypergraph,axis=0).reshape(-1)
    I = sp.sparse.lil_matrix((num_nodes, num_nodes))
    for u in range(num_nodes): I[u, u] = 1

    Dn = sp.sparse.lil_matrix((num_nodes, num_nodes))
    De= sp.sparse.lil_matrix((num_edges, num_edges))
    W = sp.sparse.lil_matrix((num_edges, num_edges))

    for u in range(num_nodes):
        Dn[u, u] = 1.0 / max(sqrt(Du[u]), epsilon)
    for i in range(num_edges):
        De[i, i] = 1.0 / max(Di[i], epsilon)
        W[i, i] = 1
    W = np.diag([0.665, 1.072, 0.365, 0.786, 1.051, 1.062, 0.648, 0.522, -1.391])
    L = I - Dn @ H @ W @ De @ H.T @ Dn

    edge = Dn @ H @ W @ De @ H.T @ Dn

    if phi > num_nodes: phi = num_nodes

    [Lamda, hypergraph_embeddings] = sp.sparse.linalg.eigsh(L, k=phi, which='SM', tol=tolerant)

    return hypergraph_embeddings


if __name__ == '__main__':
    train_hypergraph = pd.read_csv(f"cropped_data/Training set/Training.csv").iloc[:, 2:].to_numpy()
    train_size = train_hypergraph.shape[0]
    train_hypergraph_embeddings = pd.DataFrame(hypergraph_embeddings(train_hypergraph, phi=args.phi))

    Inter_test_hypergraph = pd.read_csv(f"cropped_data/Internal test set/Internal test.csv").iloc[:, 2:].to_numpy()
    Inter_size = Inter_test_hypergraph.shape[0]
    Inter_test_hypergraph_embeddings = pd.DataFrame(hypergraph_embeddings(Inter_test_hypergraph, phi=args.phi))

    Exter_test_hypergraph = pd.read_csv(f"cropped_data/External test set/External test.csv").iloc[:, 2:].to_numpy()
    Exter_size = Exter_test_hypergraph.shape[0]
    Exter_test_hypergraph_embeddings = pd.DataFrame(hypergraph_embeddings(Exter_test_hypergraph, phi=args.phi))
    # 保存到 CSV 文件
    hypergraph_embeddings = hypergraph_embeddings(np.vstack((train_hypergraph, Inter_test_hypergraph, Exter_test_hypergraph)), phi=args.phi)

    # train_hypergraph_embeddings = pd.DataFrame(hypergraph_embeddings[:train_size, :])
    # Inter_test_hypergraph_embeddings = pd.DataFrame(hypergraph_embeddings[train_size:-Exter_size, :])
    # Exter_test_hypergraph_embeddings = pd.DataFrame(hypergraph_embeddings[-Exter_size:,:])

    #
    # train_hypergraph_embeddings.to_csv(os.path.join(args.pretrain_fold, "Training_hypergraph_embeddings.csv"), index=False)
    # Inter_test_hypergraph_embeddings.to_csv(os.path.join(args.pretrain_fold, "Internal test_hypergraph_embeddings.csv"), index=False)
    # Exter_test_hypergraph_embeddings.to_csv(os.path.join(args.pretrain_fold, "External test_hypergraph_embeddings.csv"), index=False)
