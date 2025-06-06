import torch
import pandas as pd
import argparse
from _Support.Graph_Construction_Beijing_12 import calculate_the_distance_matrix
from _Support.Graph_Construction_Beijing_12 import calculate_the_neighbor_matrix
from _Support.Graph_Construction_Beijing_12 import calculate_the_similarity_matrix
from _Support.Graph_Construction_Beijing_12 import calculate_the_correlation_matrix
from _Support.Graph_Construction_Beijing_12 import calculate_adjacency_matrix
from model.model import model
from exp.exp import Exp_model



parser = argparse.ArgumentParser(description='model parameters')
parser.add_argument('--in_channels', type=int, default=10, help='feature dimension')
parser.add_argument('--T_in', type=int, default=24, help='input sequence length')
parser.add_argument('--T_out', type=int, default=1, help='output sequence length')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden size of the model')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the model')
parser.add_argument('--alpha', type=float, default=0.2, help='leaky relu alpha')
parser.add_argument('--num_heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--kernel_size', type=int, default=4, help='kernel size for TCN')
parser.add_argument('--dropout', type=float, default=0.25, help='dropout rate')
parser.add_argument('--block_num', type=int, default=2, help='number of spatiotemporal blocks')
parser.add_argument('alpha', type=float, default=0.2, help='leaky relu alpha')
parser.add_argument('--apt_size', type=int, default=10, help='size of the node attributes')
parser.add_argument('--num_channels', type=int, nargs='+', default=[64, 64, 64, 64], help='number of channels for TCN')
parser.add_argument('--K', type=int, default=3, help='number of GCN layers')
parser.add_argument('--dataset', type=str, default='Beijing_12', help='dataset name')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for optimizer')
parser.add_argument('--ratio', type=str, default='6:2:2', help='train/val/test split ratio')
parser.add_argument('--num_workers', type=int, default=3, help='number of workers for data loading')
parser.add_argument('--gated_TCN_bool', type=bool, default=True, help='use gated TCN or not')
parser.add_argument('--gcn_bool', type=bool, default=True, help='use GCN or not')
parser.add_argument('--gat_bool', type=bool, default=True, help='use GAT or not')
parser.add_argument('--ASTAM_bool', type=bool, default=True, help='use ASTAM or not')

args = parser.parse_args()

# Beijing_12距离图
adj_matrix_D, edge_index_D, edge_weight_D = calculate_the_distance_matrix(threshold=0.4)
# adj_matrix_D = torch.tensor(adj_matrix_D, dtype=torch.float).cuda()
edge_index_D = torch.tensor(edge_index_D, dtype=torch.long).cuda()
edge_weight_D = torch.tensor(edge_weight_D, dtype=torch.float).cuda()

# Beijing_12邻居图
adj_matrix_N, edge_index_N = calculate_the_neighbor_matrix()
# adj_matrix_N = torch.tensor(adj_matrix_N, dtype=torch.float).cuda()
edge_index_N = torch.tensor(edge_index_N, dtype=torch.long).cuda()
edge_weight_N = None

# Beijing_12相关图
adj_matrix_C, edge_index_C, edge_weight_C = calculate_the_correlation_matrix(threshold=0.85, target='PM25')
adj_matrix_C = torch.tensor(adj_matrix_C, dtype=torch.float).cuda()
edge_index_C = torch.tensor(edge_index_C, dtype=torch.long).cuda()
edge_weight_C = torch.tensor(edge_weight_C, dtype=torch.float).cuda()

# Beijing_12功能相似图
adj_matrix_S = pd.read_csv('./dataset/Beijing_12/POI/adjacency_matrix.csv', header=None)
adj_matrix_S = torch.tensor(adj_matrix_S.values, dtype=torch.float).cuda()

models = {
    'model_D': model(adj_matrix_D, args.in_channels, args.hidden_size, args.dropout, args.alpha, args.num_heads, kernel_size=args.kernel_size, num_channels=args.num_channels, apt_size=args.apt_size, num_nodes=args.num_nodes, num_block=args.block_num, predict_len=args.predict_len,
    gated_TCN_bool=args.gated_TCN_bool, gcn_bool=args.gcn_bool, gat_bool=args.gat_bool, ASTAM_bool=args.ASTAM_bool),
    'model_N': model(adj_matrix_N, args.in_channels, args.hidden_size, args.dropout, args.alpha, args.num_heads, kernel_size=args.kernel_size, num_channels=args.num_channels, apt_size=args.apt_size, num_nodes=args.num_nodes, num_block=args.block_num, predict_len=args.predict_len,
    gated_TCN_bool=args.gated_TCN_bool, gcn_bool=args.gcn_bool, gat_bool=args.gat_bool, ASTAM_bool=args.ASTAM_bool),
    'model_C': model(adj_matrix_C, args.in_channels, args.hidden_size, args.dropout, args.alpha, args.num_heads, kernel_size=args.kernel_size, num_channels=args.num_channels, apt_size=args.apt_size, num_nodes=args.num_nodes, num_block=args.block_num, predict_len=args.predict_len,
    gated_TCN_bool=args.gated_TCN_bool, gcn_bool=args.gcn_bool, gat_bool=args.gat_bool, ASTAM_bool=args.ASTAM_bool),
    'model_S': model(adj_matrix_S, args.in_channels, args.hidden_size, args.dropout, args.alpha, args.num_heads, kernel_size=args.kernel_size, num_channels=args.num_channels, apt_size=args.apt_size, num_nodes=args.num_nodes, num_block=args.block_num, predict_len=args.predict_len,
    gated_TCN_bool=args.gated_TCN_bool, gcn_bool=args.gcn_bool, gat_bool=args.gat_bool, ASTAM_bool=args.ASTAM_bool),
        
}

for model_name, model_instance in models.items():
    exp = Exp_model(model_name, model_instance, args.epoch, args.learning_rate, args.target, args.batch_size, args.num_workers, args.dataset, args.predict_len, args.run, args.ratio)
    print(model_name + "训练开始！")
    exp.train()
    print(model_name + "训练结束！")
    print(model_name + "测试开始！")
    exp.test()
    print(model_name + "测试结束！")
    print("================================================================================================================================================")
