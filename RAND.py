import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Parameter
from torch.utils.data import DataLoader
import numpy as np
import math

import dgl
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader, GraphDataLoader
from dgld.utils.early_stopping import EarlyStopping, EarlyStopping_ASampler
from dgl.nn.pytorch import GATConv, GraphConv
import scipy.sparse as sp
from torch.nn.functional import normalize
from numpy.linalg import inv
from tqdm import tqdm
from sklearn import metrics


def adj_normalize(mx):
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def eigenvector(L):
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    return torch.tensor(EigVec[:, 1:11], dtype = torch.float32)


def column_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = mx.dot(r_mat_inv)
    return mx


def normalize_tensor_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, mat):
        return torch.mean(mat, 1)


class MaxReadout(nn.Module):
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, mat):
        val, idx = torch.max(mat, 1)
        return val


class MinReadout(nn.Module):
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, mat):
        val, idx = torch.min(mat, 1)
        return val


def adaptive_sampler(adj, features, num_node, eigen_adj, hop1_adj, hop2_adj, knn_adj, p=None, total_sample_size=20):
    data_list = []
    for id in range(num_node):
        s_ppr = eigen_adj[id]
        s_ppr[id] = -1000.0
        top_neighbor_index = s_ppr.argsort()[-total_sample_size:] 

        s_ppr = eigen_adj[id]
        s_ppr[id] = 0
        s_ppr = np.maximum(s_ppr, 0) 
        if p is not None:
            s_hop1 = hop1_adj[id]
            s_hop2 = hop2_adj[id]
            s_knn = knn_adj[id]
            s_hop1[id] = 0
            s_hop2[id] = 0
            s_knn[id] = 0
            s_hop1 = np.maximum(s_hop1, 0)
            s_hop2 = np.maximum(s_hop2, 0)
            s_knn = np.maximum(s_knn, 0)
            s = p[0]*s_ppr/(s_ppr.sum()+1e-5) + p[1]*s_hop1/(s_hop1.sum()+1e-5) + p[2]*s_hop2/(s_hop2.sum()+1e-5) + p[3]*s_knn/(s_knn.sum()+1e-5)
        
        sampled_num = np.minimum(total_sample_size, (s > 0).sum())
        if sampled_num > 0:
            sampled_index = np.random.choice(a=np.arange(num_node), size=sampled_num, replace=False, p=s/s.sum())
        else:
            sampled_index = np.array([], dtype=int)

        data_list.append(sampled_ids)
    
    return data_list


def get_reward(device, p, ppr_adj, hop1_adj, hop2_adj, knn_adj, num_nodes, sampled_nodes, cost_mat):
    
    r = [[], [], [], []]

    v_ppr, v_hop1, v_hop2, v_knn = [], [], [], []

    reward = np.zeros(4)
    with torch.no_grad():
        for id in range(num_nodes):
            sampled_ids = sampled_nodes[id]
            
            sampled_cost = cost_mat[sampled_ids]
            center_cost = cost_mat[id].unsqueeze(0)
            score_diff = F.softmax(1/(torch.abs(sampled_cost-center_cost)+1e-5), dim=0).to(device)

            s_ppr = ppr_adj[id]
            s_hop1 = hop1_adj[id]
            s_hop2 = hop2_adj[id]
            s_knn = knn_adj[id]
            s_ppr[id], s_hop1[id], s_hop2[id], s_knn[id] = 0, 0, 0, 0
            s_ppr = torch.tensor(np.maximum(s_ppr, 0)).to(device)
            s_ppr = s_ppr/(s_ppr.sum()+1e-5)
            s_hop1 = torch.tensor(np.maximum(s_hop1, 0)).to(device)
            s_hop1 = s_hop1/(s_hop1.sum()+1e-5)
            s_hop2 = torch.tensor(np.maximum(s_hop2, 0)).to(device)
            s_hop2 = s_hop2/(s_hop2.sum()+1e-5)
            s_knn = torch.tensor(np.maximum(s_knn, 0)).to(device)
            s_knn = s_knn / (s_knn.sum() + 1e-5)
            phi = p[0]*s_ppr + p[1]*s_hop1 + p[2]*s_hop2 + p[3]*s_knn + 1e-5

            r[0].append(p[0] * s_ppr[sampled_ids] * score_diff / phi[sampled_ids])
            r[1].append(p[1] * s_hop1[sampled_ids] * score_diff / phi[sampled_ids])
            r[2].append(p[2] * s_hop2[sampled_ids] * score_diff / phi[sampled_ids])
            r[3].append(p[3] * s_knn[sampled_ids] * score_diff / phi[sampled_ids])

        reward[0] = torch.mean(torch.cat([i.unsqueeze(0) for i in r[0]])).cpu().numpy()
        reward[1] = torch.mean(torch.cat([i.unsqueeze(0) for i in r[1]])).cpu().numpy()
        reward[2] = torch.mean(torch.cat([i.unsqueeze(0) for i in r[2]])).cpu().numpy()
        reward[3] = torch.mean(torch.cat([i.unsqueeze(0) for i in r[3]])).cpu().numpy()
    return reward



class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class StructureDecoder(nn.Module):
    def __init__(self, in_dim, dropout):
        super(StructureDecoder, self).__init__()
        self.gcn = GraphConvolution(in_dim, in_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.gcn(x, adj))
        return x @ x.T


class StructureMLPDecoder(nn.Module):
    def __init__(self, in_dim, dropout):
        super(StructureMLPDecoder, self).__init__()
        self.w1 = nn.Linear(in_dim, in_dim)
        self.dropout = dropout
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act(self.w1(x))
        return x @ x.T


class AttributeDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(AttributeDecoder, self).__init__()
        self.gcn1 = GraphConvolution(in_dim, in_dim)
        self.gcn2 = GraphConvolution(in_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.gcn1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.gcn2(x, adj))
        return x


class AttributeMLPDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(AttributeMLPDecoder, self).__init__()
        self.w1 = nn.Linear(in_dim, in_dim)
        self.w2 = nn.Linear(in_dim, out_dim)
        self.dropout = dropout
        self.act = nn.LeakyReLU()
    
    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act(self.w1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.w2(x)
        return x


class MLP_proj(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.1):
        super(MLP_proj, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.w1 = nn.Linear(in_dim, hid_dim, bias=False)
        self.w2 = nn.Linear(hid_dim, out_dim, bias=False)
        self.drop = nn.Dropout(p=dropout)
        self.act = nn.LeakyReLU()
    
    def forward(self, x):
        x_hid = self.drop(self.act(self.w1(x)))
        x_out = self.w2(x_hid)
        return x_out


class AdaptiveMessageAggregator(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, sample_num, ano_rate=0.1, device='cpu'):
        super(AdaptiveMessageAggregator, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sample_num = sample_num
        self.act = nn.Tanh()
        self.ano_rate = ano_rate
        self.device = device
        self.dropout = dropout
        self.msg = MessageAggregationLayer(self.in_dim, self.out_dim, self.dropout)

    def forward(self, center_feat, neighbor_feats):
        batch_center = torch.mean(center_feat, dim=-1)
        diff_center = torch.sum(center_feat - batch_center.unsqueeze(-1), dim=-1)
        sorted_idx = torch.argsort(diff_center, dim=-1, descending=False)
        bs = sorted_idx.shape[0]
        ano_num = int(bs * self.ano_rate)

        pos_idx = sorted_idx[:bs - ano_num]
        neg_idx = sorted_idx[-ano_num:]

        agg_info = torch.zeros(center_feat.shape).to(self.device)
        agg_info[pos_idx] = self.msg(center_feat[pos_idx], neighbor_feats[pos_idx])
        if self.ano_rate > 0.0:
            agg_info[neg_idx] = center_feat[neg_idx]
        return agg_info, neg_idx


class MessageAggregationLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super(MessageAggregationLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features

        self.W1 = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.W2 = nn.Parameter(torch.empty(size=(out_features, out_features)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

        self.scorer = nn.Tanh()
        self.act = nn.LeakyReLU()

    def forward(self, h_center, h_neighbor):
        h_concated = torch.cat((h_center.unsqueeze(1), h_neighbor), 1)
        scores = self.scorer(torch.matmul(h_concated, self.W1))
        agg_h = torch.matmul(torch.sum(scores*h_concated, dim=1), self.W2)
        return agg_h


class RAND_model(nn.Module):

    def __init__(self, in_dim, out_dim, sample_num, dropout, mask_rate, device='cpu', decoder_type='mlp'):
        super(RAND_model, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sample_num = sample_num
        self.dropout = dropout
        self.device = device
        self.info_proj = MLP_proj(in_dim, 2 * out_dim, out_dim)
        self.encoder = AdaptiveMessageAggregator(out_dim, out_dim, dropout, sample_num, mask_rate, device=self.device)
        self.decoder_type = decoder_type
        if self.decoder_type == 'mlp':
            self.attribute_reconstructor = AttributeMLPDecoder(out_dim, in_dim, dropout)
            self.structure_reconstructor = StructureMLPDecoder(out_dim, dropout)
        elif self.decoder_type == 'gcn':
            self.attribute_reconstructor = AttributeDecoder(out_dim, in_dim, dropout)
            self.structure_reconstructor = StructureDecoder(out_dim, dropout)
        else:
            print("incorrect decoder selected!")
            self.attribute_reconstructor = None
            self.structure_reconstructor = None

    def anomaly_scoring(self, attribute_origin, attribute_rebuild, structure_origin, structure_rebuild):
        diff_attribute = torch.pow(attribute_origin - attribute_rebuild, 2)
        attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        diff_structure = torch.pow(structure_origin - structure_rebuild, 2)
        structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))

        return attribute_reconstruction_errors, structure_reconstruction_errors

    def rebuild(self, node_emb, adj):
        if self.decoder_type == 'mlp':
            rebuild_attribute = self.attribute_reconstructor(node_emb)
            rebuild_structure = self.structure_reconstructor(node_emb)
        elif self.decoder_type == 'gcn':
            rebuild_attribute = self.attribute_reconstructor(node_emb, adj)
            rebuild_structure = self.structure_reconstructor(node_emb, adj)
        return rebuild_attribute, rebuild_structure

    def forward(self, features, ada_neighbor_idx):
        center_rep = self.info_proj(features)
        ada_neighbor_rep = center_rep[ada_neighbor_idx]

        agg_emb, neg_idx = self.encoder(center_rep, ada_neighbor_rep)

        return agg_emb


class RAND(nn.Module):
    def __init__(self, in_dim, out_dim, sample_num, balance_param, mask_rate, dropout, num_nodes, device='cpu', decoder_type='mlp'):
        super(RAND, self).__init__()
        self.balance_param = balance_param
        self.mask_rate = mask_rate
        self.sample_num = sample_num
        self.dropout = dropout
    
        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!', flush=True)
        else:
            device = torch.device("cpu")
            print('Using cpu!!!', flush=True)
        
        self.model = RAND_model(in_dim, out_dim, sample_num, dropout, mask_rate, device, decoder_type)
        self.early_stopper = None
        self.ppr_adj = None
        self.knn_adj = None
        self.hop1_adj = None
        self.hop2_adj = None

    def fit(self, g, lr=0.005, num_epoch=100, weight_decay=0., device='cpu', warm_up_epoch=3, update_internal=3, patience=10):
        
        print('*' * 20, 'training', '*' * 20, flush=True)

        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!', flush=True)
        else:
            device = torch.device("cpu")
            print('Using cpu!!!', flush=True)

        features = g.ndata['feat']

        self.model = self.model.to(device)

        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g = g.to(device)
        features = features.to(device)
        features.requires_grad = False
        adj_sp = g.adj(scipy_fmt='coo')
        adj_full = torch.FloatTensor(adj_sp.toarray()).to(device)
        adj_feed = torch.FloatTensor(adj_normalize(adj_sp).toarray()).to(device)
        sampling_ways = 4
        
        normalized_adj = adj_normalize(adj_sp)
        column_normalized_adj = column_normalize(adj_sp)
        ppr_c = 0.15
        power_adj_list = [normalized_adj]
        for m in range(2):
            power_adj_list.append(power_adj_list[0]*power_adj_list[m])
        self.ppr_adj = ppr_c * inv((sp.eye(adj_sp.shape[0]) - (1 - ppr_c) * column_normalized_adj).toarray()) #PPR
        self.hop1_adj = power_adj_list[0].toarray()
        self.hop2_adj = power_adj_list[1].toarray()
        x = normalize(features, dim=1).cpu()
        self.knn_adj = np.array(torch.matmul(x, x.transpose(1, 0)))

        print('features shape:', features.shape, flush=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.early_stopper = EarlyStopping_ASampler(early_stopping_rounds=10, patience=patience)

        sampling_weight = np.ones(4)
        p_min = 0.05
        p = (1 - 4 * p_min) * sampling_weight / sum(sampling_weight) + p_min

        nid = np.arange(g.num_nodes())
        nid = torch.LongTensor(nid).to(device)
        warm_up_epoch = warm_up_epoch
        update_internal = update_internal
        update_day = -1

        torch.autograd.set_detect_anomaly(True)
        print("training!!!", flush=True)
        print("initial sampling weight: ", sampling_weight, flush=True)
        print("initial strategy probability: ", p, flush=True)
        for epoch in range(num_epoch):
            self.model.train()
            optimizer.zero_grad()
            sampled_result = adaptive_sampler(adj_sp, features, g.num_nodes(), self.ppr_adj, self.hop1_adj, self.hop2_adj, self.knn_adj, p=p, total_sample_size=self.sample_num)
            ada_neighbor_nodes = torch.stack(sampled_result).to(device).detach()
                        
            center_rep = self.model(features, ada_neighbor_nodes)
            
            rebuild_attribute, rebuild_structure = self.model.rebuild(center_rep, adj_feed)

            attribute_reconstruction_errors, structure_reconstruction_errors = self.model.anomaly_scoring(features, rebuild_attribute, adj_full, rebuild_structure)
            anomaly_loss = self.balance_param * attribute_reconstruction_errors + (1-self.balance_param) * structure_reconstruction_errors

            with torch.no_grad():
                anomaly_score = self.balance_param * attribute_reconstruction_errors + (1-self.balance_param) * structure_reconstruction_errors
            

            if epoch >= warm_up_epoch and (epoch-update_day) >= update_internal:
                r = get_reward(device, p, self.ppr_adj, self.hop1_adj, self.hop2_adj, self.knn_adj, g.num_nodes(), ada_neighbor_nodes, cost_mat=anomaly_score)
                print('****** reward:', r, flush=True)
                updated_param = np.exp((p_min/2.0)*(r+0.01/p)*100*np.sqrt(np.log(self.sample_num/0.01)/(sampling_ways*update_internal)))
                sampling_weight = sampling_weight*updated_param
                p = (1 - 4 * p_min) * sampling_weight / sum(sampling_weight) + p_min
                p_list.append(p.tolist())
                print('****** updated strategy probability:', p, flush=True)
                update_day = epoch


            epoch_loss = torch.mean(anomaly_loss)
            epoch_loss.backward()
            optimizer.step()
            print("# Epoch:", '%04d' % epoch, "mean anomaly score=", "{:.5f}".format(anomaly_score.mean().item()), "mean attribute score=", "{:.5f}".format(attribute_reconstruction_errors.mean().item()), "mean structure score=", "{:.5f}".format(structure_reconstruction_errors.mean().item()), flush=True)

            print("$ Epoch:", '%04d' % epoch, "train loss=", "{:.5f}".format(epoch_loss.item()), flush=True)

            self.early_stopper(epoch_loss.cpu().detach(), self.model, p)
            if self.early_stopper.isEarlyStopping():
                print(f"Early stopping in round {epoch}", flush=True)
                break
        
        return

    
    def predict(self, g, device='cpu'):
        print('*' * 20, 'predict', '*' * 20, flush=True)

        if torch.cuda.is_available() and device != 'cpu':
            device = torch.device("cuda:" + device)
            print('Using gpu!!!', flush=True)
        else:
            device = torch.device("cpu")
            print('Using cpu!!!', flush=True)
            
        features = g.ndata['feat']
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g = g.to(device)
        features = features.to(device)
        adj_sp = g.adj(scipy_fmt='coo')
        adj_full = torch.FloatTensor(adj_sp.toarray()).to(device)
        adj_feed = torch.FloatTensor(adj_normalize(adj_sp).toarray()).to(device)
        
        nid = np.arange(g.num_nodes())
        nid = torch.LongTensor(nid).to(device)

        self.model.load_state_dict(self.early_stopper.best_paramenters, strict=True)
        self.model.eval()
        print("start testing...", flush=True)
        with torch.no_grad():
            sampled_result = adaptive_sampler(adj_sp, features, g.num_nodes(), self.ppr_adj, self.hop1_adj, self.hop2_adj, self.knn_adj, p=self.early_stopper.best_sampling_probability, total_sample_size=self.sample_num)
            ada_neighbor_nodes = torch.stack(sampled_result).to(device).detach()
            
            center_rep = self.model(features, ada_neighbor_nodes)
            
            rebuild_attribute, rebuild_structure = self.model.rebuild(center_rep, adj_feed)

            attribute_reconstruction_errors, structure_reconstruction_errors = self.model.anomaly_scoring(features, rebuild_attribute, adj_full, rebuild_structure)
            anomaly_score = self.balance_param * attribute_reconstruction_errors + (1-self.balance_param) * structure_reconstruction_errors

            score = anomaly_score.detach().cpu().numpy()
        
        return score
