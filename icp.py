from scipy.spatial.transform import Rotation
from util import transform_point_cloud
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from time import time


class ICP(nn.Module):
    def __init__(self, max_iterations=10, tolerance=0.001):
        super(ICP, self).__init__()

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    # [B,dims,num]
    def forward(self, srcInit, dst):
        icp_start = time()
        src = srcInit
        prev_error = 0
        for i in range(self.max_iterations):
            # find the nearest neighbors between the current source and destination points
            mean_error, src_corr = self.nearest_neighbor(src, dst)
            # compute the transformation between the current source and nearest destination points
            rotation_ab, translation_ab = self.best_fit_transform(src, src_corr)
            src = transform_point_cloud(src, rotation_ab, translation_ab)

            if torch.abs(prev_error - mean_error) < self.tolerance:
                # print('iteration: '+str(i))
                break
            prev_error = mean_error


        # calculate final transformation
        rotation_ab, translation_ab = self.best_fit_transform(srcInit, src)

        rotation_ba = rotation_ab.transpose(2, 1).contiguous()
        translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)

        print("icp: ",time() - icp_start)
        return srcInit,src,rotation_ab, translation_ab, rotation_ba, translation_ba

    def nearest_neighbor(self,src, dst):

        batch_size = src.size(0)
        num_points = src.size(2)

        inner = -2 * torch.matmul(src.transpose(2, 1).contiguous(), dst)
        xx = torch.sum(src ** 2, dim=1, keepdim=True).transpose(2, 1).contiguous()
        yy = torch.sum(dst ** 2, dim=1, keepdim=True)

        pairwise_distance = -xx - inner
        pairwise_distance = pairwise_distance - yy

        idx = pairwise_distance.topk(k=1, dim=-1)[1]
        val = pairwise_distance.topk(k=1, dim=-1)[0]

        idx_base = torch.arange(0, batch_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))\
                       .view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        dst = dst.transpose(2, 1).contiguous()
        candidates = dst.view(batch_size * num_points, -1)[idx, :]
        # 这里的第三维是指tgt的索引，第二维是src的索引
        candidates = candidates.view(batch_size, num_points, 1, 3).squeeze(-2)  # (batch_size,num, tgtK, 3)

        return val.mean(), candidates.transpose(2, 1).contiguous()

    def best_fit_transform(self, src, src_corr):

        batch_size = src.size(0)

        src_centered = src - src.mean(dim=2, keepdim=True)

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3)



def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def getDateset(batch_size = 8,gaussian_noise = False, angle = 4,num_points = 512, dims = 3, tTld=0.5):

    pointcloud1All=[]
    pointcloud2All=[]
    R_abAll=[]
    translation_abAll=[]

    for b in range(batch_size):
        pointcloud = np.random.rand(num_points, dims)-0.5

        if gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)

        np.random.seed(b)
        anglex = np.random.uniform() * angle / 180.0 * np.pi
        angley = np.random.uniform() * angle / 180.0 * np.pi
        anglez = np.random.uniform() * angle / 180.0 * np.pi

        cosx = np.cos(anglex);cosy = np.cos(angley);cosz = np.cos(anglez)
        sinx = np.sin(anglex);siny = np.sin(angley);sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                   [0, cosx, -sinx],
                   [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                   [0, 1, 0],
                   [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                   [sinz, cosz, 0],
                   [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)

        translation_ab = np.array([np.random.uniform(-tTld, tTld), np.random.uniform(-tTld, tTld),
                                   np.random.uniform(-tTld, tTld)])

        pointcloud1 = pointcloud.T  # [3,num]

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        # pointcloud1 = np.random.permutation(pointcloud1.T).T
        # pointcloud2 = np.random.permutation(pointcloud2.T).T

        pointcloud1All.append(pointcloud1)
        pointcloud2All.append(pointcloud2)
        R_abAll.append(R_ab)
        translation_abAll.append(translation_ab)

    pointcloud1All = np.asarray(pointcloud1All).reshape(batch_size,dims,num_points)
    pointcloud2All = np.asarray(pointcloud2All).reshape(batch_size,dims,num_points)
    R_abAll = np.asarray(R_abAll).reshape(batch_size,3,3)
    translation_abAll = np.asarray(translation_abAll).reshape(batch_size,3)

    # [3,num_points]
    return pointcloud1All.astype('float32'), pointcloud2All.astype('float32'), R_abAll.astype('float32'), translation_abAll.astype('float32')



if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Only run on GPU!")
        exit(-1)
    pointcloud1, pointcloud2, R_ab, t_ab = getDateset(batch_size=1,angle=45,tTld=0.5,num_points=512)
    net = ICP().cuda()
    src = torch.tensor(pointcloud1).cuda()
    target = torch.tensor(pointcloud2).cuda()
    rotation_ab = torch.tensor(R_ab).cuda()
    translation_ab = torch.tensor(t_ab).cuda()
    batch_size = src.size(0)

    src, src_corr, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)
    identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
    loss = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) + F.mse_loss(translation_ab_pred, translation_ab)
    print(loss)




