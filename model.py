import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt
import pdb


class GTN(nn.Module):

    #num_edge = 5，num_class = 4， norm = true，
    # w_in = D, w_out = 64
    def __init__(self, num_edge, num_channels, w_in, w_out, num_class,num_layers,norm):
        super(GTN, self).__init__()
        # k
        self.num_edge = num_edge
        self.num_channels = num_channels
        self.w_in = w_in
        self.w_out = w_out
        self.num_class = num_class
        self.num_layers = num_layers
        self.is_norm = norm
        layers = []
        for i in range(num_layers):
            if i == 0:
                # k, c
                layers.append(GTLayer(num_edge, num_channels, first=True))
            else:
                layers.append(GTLayer(num_edge, num_channels, first=False))
        #         专门用于存储module的list。
        self.layers = nn.ModuleList(layers)
        self.weight = nn.Parameter(torch.Tensor(w_in, w_out))
        self.bias = nn.Parameter(torch.Tensor(w_out))
        self.loss = nn.CrossEntropyLoss()
        # w_out = 64, num_channels = c
        self.linear1 = nn.Linear(self.w_out * self.num_channels, self.w_out)
        # num_class = 3 + 1 = 4
        self.linear2 = nn.Linear(self.w_out, self.num_class)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    # X, H[i]
    def gcn_conv(self, X, H):
        X = torch.mm(X, self.weight)
        H = self.norm(H, add=True)
        return torch.mm(H.t(),X)

    def normalization(self, H):
        for i in range(self.num_channels):
            if i==0:
                H_ = self.norm(H[i,:,:]).unsqueeze(0)
            else:
                H_ = torch.cat((H_,self.norm(H[i,:,:]).unsqueeze(0)), dim=0)
        return H_

    def norm(self, H, add=False):
        H = H.t()
        if add == False:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor))
        else:
            H = H*((torch.eye(H.shape[0])==0).type(torch.FloatTensor)) + torch.eye(H.shape[0]).type(torch.FloatTensor)
        deg = torch.sum(H, dim=1)
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        deg_inv = deg_inv*torch.eye(H.shape[0]).type(torch.FloatTensor)
        H = torch.mm(deg_inv,H)
        H = H.t()
        return H

    # A, node_features = (18405, 334), train_node = 前800个label节点的索引
    # train_target = 前800个节点的0，1，2，3
    def forward(self, A, X, target_x, target):
        # 函数用来增加某个维度。在PyTorch中维度是从0开始的。
        # A = (1, 18405, 18405, 5) => (1, 5, 18405, 18405)
        A = A.unsqueeze(0).permute(0,3,1,2) 
        Ws = []
        # 2
        for i in range(self.num_layers):
            if i == 0:
                H, W = self.layers[i](A)
            else:
                # H是上一层输出A, 对H归一化
                # 当前H包含了C个邻接矩阵。
                H = self.normalization(H)
                H, W = self.layers[i](A, H)
            Ws.append(W)
        
        #H,W1 = self.layer1(A)
        #H = self.normalization(H)
        #H,W2 = self.layer2(A, H)
        #H = self.normalization(H)
        #H,W3 = self.layer3(A, H)
        for i in range(self.num_channels):
            if i==0:
                X_ = F.relu(self.gcn_conv(X, H[i]))
            else:
                X_tmp = F.relu(self.gcn_conv(X, H[i]))
                X_ = torch.cat((X_, X_tmp), dim=1)
        # nn.Linear(self.w_out * self.num_channels, self.w_out)
        X_ = self.linear1(X_)
        X_ = F.relu(X_)
        # train_node = 前800个label节点的索引
        y = self.linear2(X_[target_x])
        loss = self.loss(y, target)
        return loss, y, Ws

class GTLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, first=True):
        # k, c
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first = first
        if self.first == True:
            self.conv1 = GTConv(in_channels, out_channels)
            self.conv2 = GTConv(in_channels, out_channels)
        else:
            self.conv1 = GTConv(in_channels, out_channels)
    
    def forward(self, A, H_ = None):
        if self.first == True:
            # 获得一个Q
            a = self.conv1(A)
            b = self.conv2(A)
            # 矩阵乘法
            H = torch.bmm(a, b)
            # 经过detach返回的张量将不会进行反向传播计算梯度
            # 这个W在这里起到什么作用？
            # 卷积层weight在这里是：nn.Parameter(torch.Tensor(out_channels,in_channels,1,1))
            W = [(F.softmax(self.conv1.weight, dim=1)).detach(), (F.softmax(self.conv2.weight, dim=1)).detach()]
        else:
            # 后续层每次获得一个Q 再与上一层的输出A 相乘
            a = self.conv1(A)
            H = torch.bmm(H_,a)
            W = [(F.softmax(self.conv1.weight, dim=1)).detach()]
        return H, W

class GTConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(GTConv, self).__init__()
        # k, c
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 1 1 K C 卷积向量
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, 1, 1))
        self.bias = None
        self.scale = nn.Parameter(torch.Tensor([0.1]), requires_grad=False)
        self.reset_parameters()
    def reset_parameters(self):
        n = self.in_channels
        # 常量初始化为0.1
        nn.init.constant_(self.weight, 0.1)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            # 均匀分布
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, A):
        # F = torch.nn.functional
        # 获得一个Q
        A = torch.sum(A*F.softmax(self.weight, dim=1), dim=1)
        return A
