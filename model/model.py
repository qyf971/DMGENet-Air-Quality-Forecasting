import torch
import torch.nn as nn
import torch.nn.functional as F
from _Support.TemporalConvNet import TemporalConvNet
from GNN.GATLayer import GAT
from GNN.GCNLayer import GCNLayer


class PredictionLayer(nn.Module):
    def __init__(self, T_dim, output_T_dim, embed_size):
        super(PredictionLayer, self).__init__()

        self.conv1 = nn.Conv2d(T_dim, output_T_dim, 1)
        self.conv2 = nn.Conv2d(embed_size, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, input_prediction_layer):
        """
        :param input_prediction_layer: [B, T, N, D]
        :return: [B, N, out_T]
        """
        out = self.relu(self.conv1(input_prediction_layer))
        out = out.permute(0, 3, 2, 1)
        out = self.conv2(out)
        out = out.permute(0, 2, 3, 1)

        return out


class gated_TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.num_channels = num_channels
        self.TCN1 = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.TCN2 = TemporalConvNet(input_size, num_channels, kernel_size, dropout)

    def forward(self, x):
        """
        :param x:(batch_size, seq_len, num_nodes, features)
        :return: (batch_size, seq_len, num_nodes, num_channels[-1])
        """
        x = x.transpose(1, 2)
        B, N, t, d = x.shape
        x = x.reshape(B * N, t, d)
        x = x.transpose(1, 2)
        TCN1_output = F.tanh(self.TCN1(x))
        TCN2_output = F.sigmoid(self.TCN2(x))
        output = (TCN1_output * TCN2_output).transpose(1, 2)
        output = output.contiguous().view(B, N, t, self.num_channels[-1]).permute(0, 2, 1, 3)
        return output


class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super().__init__()
        self.num_channels = num_channels
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        B, N, t, d = x.shape
        x = x.reshape(B * N, t, d)
        x = x.transpose(1, 2)
        output = self.tcn(x).transpose(1, 2)
        output = output.contiguous().view(B, N, t, self.num_channels[-1]).permute(0, 2, 1, 3)
        return output



class SpitalBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, adj, dropout, alpha, n_heads, gcn_bool, gat_bool):
        super(SpitalBlock, self).__init__()
        self.gcn_bool = gcn_bool
        self.gat_bool = gat_bool
        self.adj = adj
        self.gcn = GCNLayer(in_channels, hidden_size, hidden_size, adj, bias=True)
        self.gat = GAT(in_channels, hidden_size, dropout, alpha, n_heads)

        self.f_gcn = nn.Linear(hidden_size, hidden_size)
        self.f_gat = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        if self.gcn_bool and self.gat_bool:
            out_gcn = self.gcn(x)
            out_gat = self.gat(x, self.adj)
            gate = torch.sigmoid(self.f_gcn(out_gcn) + self.f_gat(out_gat))
            out = gate * out_gcn + (1 - gate) * out_gat
        elif self.gcn_bool:
            out = self.gcn(x)
        elif self.gat_bool:
            out = self.gat(x, self.adj)
        return out


class ComputeAttentionScore(nn.Module):
    def __init__(self):
        super(ComputeAttentionScore, self).__init__()

    def forward(self, x, node_vec):
        n_q = node_vec.unsqueeze(dim=-1)
        x_t_a = torch.einsum('btnd,ndl->btnl', (x, n_q)).contiguous()
        return x_t_a


class model_block(nn.Module):
    def __init__(self, in_channels, hidden_size, num_channels, dropout, alpha, n_heads, kernel_size, num_nodes, apt_size, adj, gated_TCN_bool, gcn_bool, gat_bool, ASTAM_bool):
        super(model_block, self).__init__()
        self.gated_TCN_bool = gated_TCN_bool
        self.ASTAM_bool = ASTAM_bool
        self.start_conv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=(1, 1))
        self.gated_TCN = gated_TCN(hidden_size, num_channels, kernel_size, dropout)
        self.TCN = TCN(hidden_size, num_channels, kernel_size, dropout)
        self.spital_block = SpitalBlock(hidden_size, hidden_size, adj, dropout, alpha, n_heads, gcn_bool, gat_bool)
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, apt_size).cuda(), requires_grad=True).cuda()
        self.ComputeAttentionScore = ComputeAttentionScore()
        self.w_s = (nn.Conv2d(in_channels=apt_size, out_channels=hidden_size, kernel_size=(1, 1)))
        self.w_t = (nn.Conv2d(in_channels=apt_size, out_channels=hidden_size, kernel_size=(1, 1)))

    def forward(self, x):
        """
        :param x: (batch_size, num_nodes, seq_len, in_dim)
        :return: (batch_size, num_nodes, seq_len. hidden_size)
        """
        x = x.permute(0, 3, 1, 2)
        x = self.start_conv(x).permute(0, 3, 2, 1)
        if self.gated_TCN_bool:
            x_t = self.gated_TCN(x)
        else:
            # x_t = self.TCN(x)
            x_t = x
        x_s = self.spital_block(x_t)
        if self.ASTAM_bool:
            n_q_t = self.w_t(self.node_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)).squeeze()
            x_t_a = self.ComputeAttentionScore(x_t, n_q_t)

            n_q_s = self.w_s(self.node_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)).squeeze()
            x_s_a = self.ComputeAttentionScore(x_s, n_q_s)
            # node-level adaptation tendencies
            x_a = torch.cat((x_t_a, x_s_a), -1)
            x_att = F.softmax(x_a, dim=-1)

            # Add Temporal, Spatial attention
            x = x_att[:, :, :, 0].unsqueeze(dim=-1) * x_t + x_att[:, :, :, 1].unsqueeze(dim=-1) * x_s
            x = x.transpose(1, 2)
        else:
            x = x_t.transpose(1, 2) + x_s.transpose(1, 2)
        return x


class model(nn.Module):
    def __init__(self, adj, input_size, hidden_size, dropout, alpha, n_heads, kernel_size, num_channels, apt_size, num_nodes, num_block, predict_len, gated_TCN_bool, gcn_bool, gat_bool, ASTAM_bool):
        super(model, self).__init__()
        self.hidden_size = hidden_size
        self.predict_len = predict_len
        self.Blocklist = nn.ModuleList([
            model_block(in_channels=input_size, hidden_size=hidden_size, num_channels=num_channels, dropout=dropout, alpha=alpha, n_heads=n_heads, kernel_size=kernel_size, num_nodes=num_nodes, apt_size=apt_size, adj=adj,
                                             gated_TCN_bool=gated_TCN_bool, gcn_bool=gcn_bool, gat_bool=gat_bool, ASTAM_bool=ASTAM_bool)
            if i == 0 else
            model_block(in_channels=hidden_size, hidden_size=hidden_size, num_channels=num_channels, dropout=dropout, alpha=alpha, n_heads=n_heads, kernel_size=kernel_size, num_nodes=num_nodes, apt_size=apt_size, adj=adj,
                                             gated_TCN_bool=gated_TCN_bool, gcn_bool=gcn_bool, gat_bool=gat_bool, ASTAM_bool=ASTAM_bool)
            for i in range(num_block)
        ])
        self.linear = nn.Linear(hidden_size, 1)
        self.prediction_layer = PredictionLayer(T_dim=T_in, output_T_dim=T_out, embed_size=hidden_size)

    def forward(self, x):
        # B, N, t, d = x.shape
        for i in range(len(self.Blocklist)):
            x = self.Blocklist[i](x)
        B, N, t, _ = x.size()
        x = x.contiguous().view(B * N, t, self.hidden_size)
        x = self.linear(x)
        x = x.contiguous().view(B, N, t, 1)
        x = x.permute(0, 2, 1, 3)
        x = self.prediction_layer(x)
        return x