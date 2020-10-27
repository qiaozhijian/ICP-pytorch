#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation
from torch_geometric.nn import knn_graph
import pynvml

pynvml.nvmlInit()
handle0 = pynvml.nvmlDeviceGetHandleByIndex(0)
if torch.cuda.device_count() > 1:
    handle1 = pynvml.nvmlDeviceGetHandleByIndex(1)
ratio = 1024 ** 2

def print_gpu(s=""):
    if torch.cuda.device_count() > 1:
        meminfo0 = pynvml.nvmlDeviceGetMemoryInfo(handle0)
        meminfo1 = pynvml.nvmlDeviceGetMemoryInfo(handle0)
        used = (meminfo0.used + meminfo1.used) / ratio
    else:
        meminfo0 = pynvml.nvmlDeviceGetMemoryInfo(handle0)
        used = meminfo0.used / ratio
    print(s+" used: ", used)

class GlobalVar():
    """A class for recording intermediate data in attention for plotting usage.

    """
    def __init__(self):
        self.self_att_src = None
        self.cross_self_att_src = None
        self.cross_att_src = None
        self.self_att_tgt = None
        self.cross_self_att_tgt = None
        self.cross_att_tgt = None
        self.src = None
        self.tgt = None


    def transform_np(self):
        """transform all the attributes into numpy.array

        Returns:
            none
        """
        print(self.self_att_src.shape)
        self.self_att_src = self.format(self.self_att_src)
        self.cross_self_att_src = self.format(self.cross_self_att_src)
        self.cross_att_src = self.format(self.cross_att_src)
        self.self_att_tgt = self.format(self.self_att_tgt)
        self.cross_self_att_tgt = self.format(self.cross_self_att_tgt)
        self.cross_att_tgt = self.format(self.cross_att_tgt)
        self.src = self.format(self.src)
        self.tgt = self.format(self.tgt)


    def format(self, data):
        """reformat the tensor into numpy array

        Args:
            data: input Tensor or numpy array

        Returns:
            reformatted numpy array
        """
        if type(data) is torch.Tensor:  data=data.detach().cpu().numpy()
        data= np.squeeze(data)
        return data


# Part of the code is referred from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py

def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rotation, translation):
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, 3, N]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    xyz = xyz.transpose(2, 1)
    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # 采样点矩阵（B, npoint）
    distance = torch.ones(B, N).to(device) * 1e10  # 采样点到所有点距离（B, N）

    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # batch_size 数组

    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)  # 初始时随机选择一点

    barycenter = torch.sum((xyz), 1)  # 计算重心坐标 及 距离重心最远的点
    barycenter = barycenter / xyz.shape[1]
    barycenter = barycenter.view(B, 1, 3)

    dist = torch.sum((xyz - barycenter) ** 2, -1)
    farthest = torch.max(dist, 1)[1]  # 将距离重心最远的点作为第一个点

    for i in range(npoint):
        centroids[:, i] = farthest  # 更新第i个最远点
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # 取出这个最远点的xyz坐标
        dist = torch.sum((xyz - centroid) ** 2, -1)  # 计算点集中的所有点到这个最远点的欧式距离
        mask = dist < distance
        distance[mask] = dist[mask]  # 更新distance，记录样本中每个点距离所有已出现的采样点的最小距离
        farthest = torch.max(distance, -1)[1]  # 求最小距离里的最大值（避免两个点互相为最远的情况），返回最远点索引

    return centroids

def knn(x, k):
    """get k nearest neighbors based on distance in feature space

    Args:
        x: [b,dims(=3),num]
        k: number of neighbors to select

    Returns:
        k nearest neighbors (batch_size, num_points, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x) # [b,num,num]
    # 求坐标（维度空间）的平方和
    xx = torch.sum(x ** 2, dim=1, keepdim=True) # [b,1,num] #x ** 2 表示点平方而不是x*x
    # 2x1x2+2y1y2+2z1z2-x1^2-y1^2-z1^2-x2^2-y2^2-z2^2=-[(x1-x2)^2+(y1-y2)^2+(z1-z2)^2]
    pairwise_distance = -xx - inner
    pairwise_distance = pairwise_distance - xx.transpose(2, 1).contiguous() # [b,num,num]
    idx = pairwise_distance.topk(k=k+1, dim=-1)[1][:,:,1:]  # (batch_size, num_points, k)
    return idx

def knn_efficient(x, k):
    """get k nearest neighbors based on distance in feature space

    Args:
        x: [b,dims(=3),num]
        k: number of neighbors to select

    Returns:
        k nearest neighbors (batch_size, num_points, k)
    """
    batch_size = x.shape[0]
    num_points = x.shape[2]
    fea_dims = x.shape[1]
    x_ = x.transpose(1, 2).reshape(-1, fea_dims)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = torch.arange(0,batch_size,device=device)\
        .reshape(batch_size,1).repeat(1,num_points).reshape(-1)
    idx = knn_graph(x_, k=k, batch=batch, loop=False)
    idx = idx[0].view(batch_size,num_points,k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx - idx_base
    return idx

# input x [B,dims,num]
# output [B, dims*2, num, k] 邻域特征tensor
    """

    Args:
        x: [B,dims,num]
        k:
        idx:

    Returns:
        邻域特征tensor [B, dims*2, num, k]
    """
def get_graph_feature(x, k=20, idx=None):
    batch_size, dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn_efficient(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 获得索引阶梯数组
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points  # (batch_size, 1, 1) [0 num_points ... num_points*(B-1)]
    # 以batch为单位，加到索引上
    idx = idx + idx_base  # (batch_size, num_points, k)
    # 展成一维数组，方便后续索引
    idx = idx.view(-1)  # (batch_size * num_points * k)
    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, dims)
    # 改变x的shape，方便索引。被索引数组是所有batch的所有点的特征，索引数组idx为所有临近点对应的序号，从而索引出所有邻域点的特征
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size * num_points * k,dims)
    # 统一数组形式
    feature = feature.view(batch_size, num_points, k, dims)  # (batch_size, num_points, k, dims)
    # 重复k次，以便k个邻域点每个都能和中心点做运算
    x = x.view(batch_size, num_points, 1, dims).repeat(1, 1, k, 1)  # [B, num, k, dims]
    # 邻域特征的表示，为(feature - x, x)，这种形式可以详尽参见dgcnn论文
    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)     # [B, dims*2, num, k]

    return feature

# input x [B,dims,num]
# output [B, dims*2, num, k] 邻域特征tensor
def get_graph_featureNew(x, k=20, idx=None):
    # 代码更少，但会占用更多显存
    batch_size, dims, num_points = x.size()
    if idx is None:
        idx = knn_efficient(x, k=k)  # (batch_size, num_points, k)
    idx = idx.view(batch_size,num_points*k).unsqueeze(1).repeat(1,dims,1)
    feature = torch.gather(x,index=idx,dim=2).view(batch_size,dims, num_points, k)
    x = x.unsqueeze(3).repeat(1, 1, 1, k)
    feature = torch.cat((feature, x),dim=1)     # [B, dims*2, num, k]

    return feature