import copy
import os.path as osp
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.io

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn.pool import *
from torch_geometric.utils import normalized_cut
from torch_cluster import nearest


class GraphPyramid():
    """Construct a graph for a given heart along with a graph hierarchy.
    For graph construction: Nodes are converted to vertices, edges are added between every node
    and it K nearest neighbor (criteria can be modified) and edge attributes between any two vertices
    is the normalized differences of Cartesian coordinates if an edge exists between the nodes
    , i.e., normalized [x1-x2, y1-y2, z1-z2] and 0 otherwise.
    
    For graph hierarchy, graph clustering method is used.
    
    Args:
        heart: name of the cardiac anatomy on which to construct the  graph and its hierarchy
        K: K in KNN for defining edge connectivity in the graph
    """

    def __init__(self, heart='case3', structure='EC', num_mesh=1230, seq_len=201, graph_method='bipartite', K=6):
        """
        """
        self.path_in = osp.join(osp.dirname(osp.realpath('__file__')), 'data', 'signal', heart)
        self.method = graph_method
        self.path_structure = osp.join(osp.dirname(osp.realpath('__file__')), 'data', 'structure', structure)
        self.pre_transform = T.KNNGraph(k=K)
        self.transform = T.Cartesian(cat=False)
        self.filename = osp.join(self.path_in, heart)
        self.heart_name = structure
        self.num_mesh = num_mesh
        self.seq_len = seq_len

    def normalized_cut_2d(self, edge_index, pos):
        """ calculate the normalized cut 2d 
        """
        row, col = edge_index
        if pos.size(1) == 3:
            edge_attr = torch.norm(pos[row] - pos[col], dim=1)
        else:
            u = pos[row] - pos[col]
            edge_attr = torch.norm(u[:, 0:3], dim=1)
        return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

    def save_graph(self, h_gs, h_Ps, t_gs, t_Ps, Hs, Ps):
        """save the graphs and the pooling matrices in a file
        """
        with open('{}_{}.pickle'.format(self.filename, self.method), 'wb') as f:
            for h_g in h_gs:
                pickle.dump(h_g, f)

            for h_P in h_Ps:
                pickle.dump(h_P, f)

            for t_g in t_gs:
                pickle.dump(t_g, f)

            for t_P in t_Ps:
                pickle.dump(t_P, f)

            if self.method == 'bipartite':
                pickle.dump(Hs, f)
                pickle.dump(Ps, f)
            else:
                raise NotImplementedError
    
    def cluster_mesh(self, g, cluster, cor, edge_index):
        m = len(cor)
        n = len(cluster)
        P = np.zeros((n, m))
        for i in range(n):
            j = cluster[i] - 1
            P[i, j] = 1
        # P = cluster
        
        Pn = P / P.sum(axis=0)
        PnT = torch.from_numpy(np.transpose(Pn)).float()

        m, s = g.x.shape
        x = g.x.view(m, s)
        x = torch.mm(PnT, x)
        x = x.view(-1, s)
        
        edge_index = torch.tensor(edge_index)
        cor = torch.tensor(cor).float()
        g_coarse = Data(x=x, y=g.y, pos=cor, edge_index=edge_index)
        g_coarse = self.transform(g_coarse)
        return P, g_coarse


    # def clus_heart(self, d, method='graclus'):
    #     """Use graph clustering method to make a hierarchy of coarser-finer graphs
        
    #     Args:
    #         method: graph clustering method to use (options: graclus or voxel)
    #         d: a instance of Data class (a graph object)
        
    #     Output:
    #         P: transformation matrix from coarser to finer scale
    #         d_coarser: graph for the coarser scale
    #     """
    #     # clustering
    #     if (method == 'graclus'):
    #         weight = self.normalized_cut_2d(d.edge_index, d.pos)
    #         cluster = graclus(d.edge_index, weight, d.x.size(0))
    #     elif (method == 'voxel'):
    #         cluster = voxel_grid(d.pos, torch.tensor(np.zeros(d.pos.shape[0])), size=10)
    #     else:
    #         print('this clustering method has not been implemented')

    #     # get clusters assignments with consequitive numbers
    #     cluster, perm = self.consecutive_cluster(cluster)
    #     unique_cluster = np.unique(cluster)
    #     n, m = cluster.shape[0], unique_cluster.shape[0]  # num nodes, num clusters

    #     # transformaiton matrix that consists of num_nodes X num_clusters
    #     P = np.zeros((n, m))
    #     # P_{ij} = 1 if ith node in the original cluster was merged to jth node in coarser scale
    #     for j in range(m):
    #         i = np.where(cluster == int(unique_cluster[j]))
    #         P[i, j] = 1
    #     Pn = P / P.sum(axis=0)  # column normalize P
    #     PnT = torch.from_numpy(np.transpose(Pn)).float()  # PnT tranpose
    #     # the coarser scale features =  Pn^T*features
    #     # this is done for verification purpose only
    #     m, _, s = d.x.shape
    #     x = d.x.view(m, s)
    #     x = torch.mm(PnT, x)  # downsampled features
    #     pos = torch.mm(PnT, d.pos)  # downsampled coordinates (vertices)
    #     x = x.view(-1, 1, s)

    #     # convert into a new object of data class (graphical format)
    #     d_coarser = Data(x=x, pos=pos, y=d.y)
    #     d_coarser = self.pre_transform(d_coarser)
    #     d_coarser = self.transform(d_coarser)
    #     return P, d_coarser

    def consecutive_cluster(self, src):
        """
        Args:
            src: cluster
        """
        unique, inv = torch.unique(src, sorted=True, return_inverse=True)
        perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
        perm = inv.new_empty(unique.size(0)).scatter_(0, inv, perm)
        return inv, perm
    
    def get_embeddings(self, g, heart_name, index, is_heart=True):
        if is_heart:
            matFiles = scipy.io.loadmat(osp.join(self.path_structure, '{}_{}.mat'.format(heart_name, index)), squeeze_me=True, struct_as_record=False)
        else:
            matFiles = scipy.io.loadmat(osp.join(self.path_structure, '{}_t{}.mat'.format(heart_name, index)), squeeze_me=True, struct_as_record=False)
        face = matFiles['face']
        cor = matFiles['cor']
        cluster = matFiles['cluster']
        edge_index = self.face2edge(face)
        P, g_coarse = copy.deepcopy(self.cluster_mesh(g, cluster, cor, edge_index))

        # if is_heart:
        #     self.save_connection(g_coarse, name='h{}'.format(index), face=face)
        # else:
        #     self.save_connection(g_coarse, name='t{}'.format(index), face=face)

        return P, g_coarse
    
    def bipartite_graph(self, left, right):
        left = copy.deepcopy(left)
        right = copy.deepcopy(right)

        row = np.arange(left.pos.shape[0])
        col = np.arange(right.pos.shape[0])
        row_len = row.shape[0]
        col_len = col.shape[0]
        col += row_len
        edge_index = []
        for i in row:
            for j in col:
                edge_index.append([i, j])
        edge_index = np.array(edge_index)
        edge_index = edge_index.transpose()
        combine_x = torch.cat((left.x, right.x), 0)
        combine_pos = torch.cat((left.pos, right.pos), 0)
        H_inv = Data(x=combine_x, y=left.y, pos=combine_pos)
        H_inv.edge_index = torch.tensor(edge_index)
        H_inv = self.transform(H_inv)

        P = np.zeros((row_len, col_len))
        edge_attr = H_inv.edge_attr.numpy()
        for i in range(row_len):
            for j in range(col_len):
                P[i, j] = np.sqrt(edge_attr[i * col_len + j, :] ** 2)[0]
        
        return H_inv, P
    
    def face2edge(self, face):
        edge_index = []
        for triangle in face:
            a, b, c = triangle
            if [a, b] not in edge_index:
                edge_index.append([a, b])
            if [b, a] not in edge_index:
                edge_index.append([b, a])
            if [a, c] not in edge_index:
                edge_index.append([a, c])
            if [c, a] not in edge_index:
                edge_index.append([c, a])
            if [b, c] not in edge_index:
                edge_index.append([b, c])
            if [c, b] not in edge_index:
                edge_index.append([c, b])
        edge_index = sorted(edge_index, key=lambda x: x[0])
        edge_index = np.array(edge_index).transpose()
        edge_index = edge_index - 1
        return edge_index.astype(np.int64)
    
    def save_connection(self, g, name, face=0):
        edge_index = g.edge_index
        pos = g.pos

        edge_index = edge_index.numpy()
        edge_index = edge_index.tolist()

        pos = pos.numpy()
        pos = pos.tolist()
        file_name = self.filename + '{}.mat'.format(name)
        scipy.io.savemat(file_name, {'pos': pos, 'edge_index': edge_index, 'face': face})

    def make_graph(self, K=6):
        """Main function for constructing the graph and its hierarchy
        """

        # Create a graph on a subset of datapoints with pre-transform and transform properties 
        h_gs, h_Ps = [], []
        matFiles = scipy.io.loadmat(osp.join(self.path_structure, '{}_0.mat'.format(self.heart_name)), squeeze_me=True, struct_as_record=False)
        cor = matFiles['cor']
        face = matFiles['face']
        cor = torch.from_numpy(cor)
        heart = Data(x=torch.zeros([cor.shape[0], self.seq_len]), y=torch.zeros(2), pos=cor)

        edge_index = self.face2edge(face)
        heart.edge_index = torch.tensor(edge_index)
        heart = self.transform(heart)
        h_g = copy.deepcopy(heart)  # graph at the meshfree nodes level
        # self.save_connection(h_g, name='h0', face=face)  # plot the graph
        h_gs.append(h_g)
        
        h_P1, h_g1 = self.get_embeddings(h_g, self.heart_name, 1)
        h_gs.append(h_g1)
        h_Ps.append(h_P1)

        h_P2, h_g2 = self.get_embeddings(h_g1, self.heart_name, 2)
        h_gs.append(h_g2)
        h_Ps.append(h_P2)
        
        h_P3, h_g3 = self.get_embeddings(h_g2, self.heart_name, 3)
        h_gs.append(h_g3)
        h_Ps.append(h_P3)
        
        h_P4, h_g4 = self.get_embeddings(h_g3, self.heart_name, 4)
        h_gs.append(h_g4)
        h_Ps.append(h_P4)

        t_gs, t_Ps = [], []
        matFiles = scipy.io.loadmat(osp.join(self.path_structure, '{}_t0.mat'.format(self.heart_name)), squeeze_me=True, struct_as_record=False)
        cor = matFiles['cor']
        face = matFiles['face']
        cor = torch.from_numpy(cor)
        torso = Data(x=torch.zeros([cor.shape[0], self.seq_len]), y=torch.zeros(2), pos=cor)
        
        edge_index = self.face2edge(face)
        torso.edge_index = torch.tensor(edge_index)
        torso = self.transform(torso)
        t_g = copy.deepcopy(torso)  # graph at the meshfree nodes level
        # self.save_connection(t_g, name='t0')  # plot the graph
        t_gs.append(t_g)

        t_P1, t_g1 = self.get_embeddings(t_g, self.heart_name, 1, is_heart=False)
        t_gs.append(t_g1)
        t_Ps.append(t_P1)
        
        t_P2, t_g2 = self.get_embeddings(t_g1, self.heart_name, 2, is_heart=False)
        t_gs.append(t_g2)
        t_Ps.append(t_P2)
        
        t_P3, t_g3 = self.get_embeddings(t_g2, self.heart_name, 3, is_heart=False)
        t_gs.append(t_g3)
        t_Ps.append(t_P3)

        if self.method == 'bipartite':
            H_inv, P = self.bipartite_graph(h_g4, t_g3)
            self.save_graph(h_gs, h_Ps, t_gs, t_Ps, H_inv, P)
        else:
            raise NotImplementedError
