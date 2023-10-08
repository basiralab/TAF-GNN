import numpy as np
import pickle
import torch
from sklearn.model_selection import KFold
from torch_geometric.data import Data

def create_edge_index_attribute(adj_matrix):
    """
    Given an adjacency matrix, this function creates the edge index and edge attribute matrix
    suitable to graph representation in PyTorch Geometric.
    """

    rows, cols = adj_matrix.shape[0], adj_matrix.shape[1]
    edge_index = torch.zeros((2, rows * cols), dtype=torch.long)
    edge_attr = torch.zeros((rows * cols, 1), dtype=torch.float)
    counter = 0

    for src, attrs in enumerate(adj_matrix):
        for dest, attr in enumerate(attrs):
            edge_index[0][counter], edge_index[1][counter] = src, dest
            edge_attr[counter] = attr
            counter += 1

    return edge_index, edge_attr, rows, cols


def swap(data):
    # Swaps the x & y values of the given graph
    edge_i, edge_attr, _, _ = create_edge_index_attribute(data.y)
    data_s = Data(x=data.y, edge_index=edge_i, edge_attr=edge_attr, y=data.x)
    return data_s


def cross_val_indices(folds, num_samples, new=False):
    """
    Takes the number of inputs and number of folds.
    Determines indices to go into validation split in each turn.
    Saves the indices on a file for experimental reproducibility and does not overwrite
    the already determined indices unless new=True.
    """

    kf = KFold(n_splits=folds, shuffle=True)
    train_indices = list()
    val_indices = list()

    try:
        if new == True:
            raise IOError
        with open("../data/" + str(folds) + "_" + str(num_samples) + "cv_train", "rb") as f:
            train_indices = pickle.load(f)
        with open("../data/" + str(folds) + "_" + str(num_samples) + "cv_val", "rb") as f:
            val_indices = pickle.load(f)
    except IOError:
        for tr_index, val_index in kf.split(np.zeros((num_samples, 1))):
            train_indices.append(tr_index)
            val_indices.append(val_index)
        with open("../data/" + str(folds) + "_" + str(num_samples) + "cv_train", "wb") as f:
            pickle.dump(train_indices, f)
        with open("../data/" + str(folds) + "_" + str(num_samples) + "cv_val", "wb") as f:
            pickle.dump(val_indices, f)

    return train_indices, val_indices

def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{}".format(int(hours), int(minutes), seconds))
